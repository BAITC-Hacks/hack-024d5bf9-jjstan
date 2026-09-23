from copy import deepcopy
from io import BytesIO
from pathlib import Path

import pandas as pd
import pytest

from engine_examples import make_examples
from quality_review import (BLOCKING, REVIEW, OUTSIDE, build_issue_report, build_issue_csv,
                            reconciliation_details, summarize_issues)
from ui_integration import run_calculation


def case_with_issues(*issues):
    case = make_examples()['control_72']
    case['dataset']['quality_report'] = pd.DataFrame([
        dict(supplier='IEK', sku='SYNTH-001', severity='error', issue=issue, source='sales.xlsx') for issue in issues])
    return case['dataset'], case['settings']


def test_period_exceptions_stay_bound_to_exact_issue_and_preserve_severity():
    old, current = 'monthly_transaction_mismatch:2024-01', 'monthly_transaction_mismatch:2026-03'
    data, settings = case_with_issues(old, current)
    original = deepcopy(data)
    result, prepared = run_calculation(data, settings)
    report = build_issue_report(prepared, result)
    past = report.loc[report.issue.eq(old)].iloc[0]
    assert past.applicability == OUTSIDE
    assert past.severity == 'error'
    assert '2026-03' in past.evidence
    assert past.months == '2024-01'
    assert report.loc[report.issue.eq(current), 'applicability'].tolist() == [BLOCKING]
    assert result['orders'].data_quality.tolist() == ['insufficient']
    assert 'sales.xlsx' in past.source and 'engine.py' in past.source
    for name in data:
        pd.testing.assert_frame_equal(data[name], original[name])


@pytest.mark.parametrize('issue', ['monthly_transaction_mismatch:2024-01,2026-03',
                                 'monthly_transaction_mismatch:', 'conflicting_units'])
def test_only_exact_engine_exception_can_mark_a_finding_historical(issue):
    data, settings = case_with_issues(issue)
    result, prepared = run_calculation(data, settings)
    report = build_issue_report(prepared, result)
    assert report.loc[report.issue.eq(issue), 'applicability'].tolist() == [BLOCKING]


def test_error_not_reached_by_engine_is_pending_instead_of_inferred_from_date():
    issue = 'monthly_transaction_mismatch:2024-01'
    data, settings = case_with_issues(issue)
    data['products']['unit'] = None
    result, prepared = run_calculation(data, settings)
    report = build_issue_report(prepared, result)
    assert report.loc[report.issue.eq(issue), 'applicability'].tolist() == [REVIEW]
    assert report.loc[report.issue.eq('Не подтверждена единица учёта'), 'applicability'].tolist() == [BLOCKING]


def test_supplier_findings_deduplicate_and_do_not_cross_suppliers_or_invent_skus():
    data, settings = case_with_issues('unit_missing')
    product = data['products'].iloc[0].to_dict()
    data['products'] = pd.DataFrame([dict(product, sku='001_'), dict(product, sku='002_'),
                                   dict(product, supplier='Systeme Electric', sku='001_')])
    data['quality_report'] = pd.DataFrame([
        dict(supplier='IEK', sku=None, severity='error', issue='current_stock_missing', source='stock.xlsx'),
        dict(supplier='IEK', sku=None, severity='error', issue='current_stock_missing', source='stock.xlsx'),
        dict(supplier='IEK', sku='UNKNOWN', severity='warning', issue='sku_only_in_one_source_type', source='one.xlsx'),
        dict(supplier='Other source', sku=None, severity='error', issue='unreadable_workbook', source='broken.xlsx'),
    ])
    result = dict(orders=pd.DataFrame(), diagnostics=pd.DataFrame([
        dict(supplier='IEK', sku='001_', severity='error', issue='current_stock_missing')]))
    report = build_issue_report(data, result)
    stocks = report.loc[report.issue.eq('current_stock_missing')]
    assert stocks.sku.tolist() == ['001_', '002_']
    assert stocks.applicability.eq(REVIEW).all()
    summary = summarize_issues(report).set_index('Тип проблемы')
    assert summary.loc['Актуальный остаток', 'Товаров'] == 2
    assert not report.loc[report.sku.eq('UNKNOWN'), 'linked'].any()
    assert report.loc[report.supplier.eq('Other source'), 'sku'].isna().all()


def test_exception_does_not_apply_to_same_sku_from_another_supplier():
    issue = 'monthly_transaction_mismatch:2024-01'
    data, settings = case_with_issues(issue)
    result, prepared = run_calculation(data, settings)
    prepared['quality_report'] = pd.concat([prepared['quality_report'],
        pd.DataFrame([dict(supplier='Systeme Electric', sku='SYNTH-001', severity='error', issue=issue, source='se.xlsx')])])
    report = build_issue_report(prepared, result)
    assert report.loc[report.supplier.eq('IEK') & report.issue.eq(issue), 'applicability'].tolist() == [OUTSIDE]
    assert report.loc[report.supplier.eq('Systeme Electric'), 'applicability'].tolist() == [REVIEW]


def test_csv_and_reconciliation_keep_unknowns_codes_evidence_and_safe_text():
    issue = 'monthly_transaction_mismatch:2026-03'
    data, settings = case_with_issues(issue)
    data['products']['sku'] = '001_'
    data['products']['name'] = '=dangerous()'
    for name in ['quality_report', 'monthly_sales', 'stock', 'transit']:
        data[name]['sku'] = '001_'
    data['monthly_sales']['reconciled_transaction_quantity'] = float('nan')
    data['monthly_sales']['reconciliation_difference'] = float('nan')
    data['monthly_sales']['reconciliation_status'] = 'incomplete_quantity'
    result, prepared = run_calculation(data, settings)
    report = build_issue_report(prepared, result)
    details = reconciliation_details(data['monthly_sales'], report.loc[report.issue.eq(issue)].iloc[0])
    assert details.reconciled_transaction_quantity.isna().all()
    assert details.reconciliation_difference.isna().all()
    assert details.quantity.tolist() == [310]
    payload = build_issue_csv(report, calculation_date=settings['as_of'], source_mode='Демонстрация')
    assert payload.startswith(b'\xef\xbb\xbf')
    exported = pd.read_csv(BytesIO(payload), sep=';', dtype=str)
    assert exported['Код 1С'].eq('001_').all()
    assert exported['Наименование'].eq("'=dangerous()").all()
    assert exported['Исходное сообщение'].tolist() == [issue]
    assert exported['Исходный уровень'].tolist() == ['error']
    assert exported['Основание применимости'].str.contains(issue, regex=False).all()


def test_empty_report_has_stable_schema():
    report = build_issue_report({}, {})
    assert report.empty
    assert summarize_issues(report).empty
    assert 'Применимость' in build_issue_csv(report, calculation_date='2026-09-22', source_mode='Мои файлы').decode('utf-8-sig')


def open_app():
    from streamlit.testing.v1 import AppTest
    return AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'), default_timeout=30).run()


def test_ui_filters_preserve_approval_and_stale_report_cannot_be_downloaded():
    at = open_app()
    assert not at.exception
    at.button(key='select_visible').click().run()
    at.checkbox(key='confirm_review').check().run()
    at.button(key='approve').click().run()
    payload = at.session_state['approval']['payload']
    at.selectbox(key='quality_filter_status').set_value(BLOCKING).run()
    at.text_input(key='quality_filter_search').set_value('DEMO-007').run()
    assert not at.exception
    assert at.session_state['approval']['payload'] == payload
    assert at.selectbox(key='quality_product').value[1] == 'DEMO-007'
    at.number_input(key='lead_se').set_value(10).run()
    assert not at.exception
    downloads = [item for item in at.get('download_button') if item.proto.label == 'Скачать полный список проблем (CSV)']
    assert len(downloads) == 1 and downloads[0].proto.disabled
    assert 'approval' not in at.session_state


def test_ui_shows_historical_exception_and_monthly_evidence(monkeypatch):
    import demo_data
    data = demo_data.make_demo_dataset()
    data['quality_report'] = pd.DataFrame([
        dict(supplier='Systeme Electric', sku='DEMO-001', severity='error',
             issue='monthly_transaction_mismatch:' + month, source='synthetic.xlsx')
        for month in ['2024-01', '2026-08']])
    data['monthly_sales']['reconciled_transaction_quantity'] = 300.0
    data['monthly_sales']['reconciliation_difference'] = data['monthly_sales'].quantity - 300
    data['monthly_sales']['reconciliation_status'] = 'mismatch'
    monkeypatch.setattr(demo_data, 'make_demo_dataset', lambda: deepcopy(data))
    at = open_app()
    assert not at.exception
    at.selectbox(key='quality_filter_category').set_value('Расхождения продаж').run()
    at.selectbox(key='quality_filter_status').set_value(OUTSIDE).run()
    assert not at.exception
    assert any('2024-01' in item.value for item in at.text)
    at.selectbox(key='quality_filter_status').set_value(BLOCKING).run()
    assert not at.exception
    evidence = [item.value for item in at.dataframe if 'Разница: отчёт − операции' in item.value.columns]
    assert len(evidence) == 1
    assert evidence[0]['Месячный отчёт'].tolist() == [310]
    assert evidence[0]['Сумма операций'].tolist() == [300]
    assert evidence[0]['Разница: отчёт − операции'].tolist() == [10]
