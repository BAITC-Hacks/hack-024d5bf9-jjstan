"""Controlled synthetic examples; no commercial Excel data is required."""
from copy import deepcopy
import pandas as pd
import pytest
from engine import calculate_orders, round_order, ORDER_COLUMNS


@pytest.fixture
def case():
    dataset = {
        'products': pd.DataFrame([dict(supplier='IEK', sku='001_', name='Test', category='A',
            unit='шт', min_order_qty=24, pack_multiple=12)]),
        'monthly_sales': pd.DataFrame([dict(supplier='IEK', sku='001_', month=pd.Timestamp('2026-03-01'),
            quantity=310, is_complete=True)]),
        'stock': pd.DataFrame([dict(supplier='IEK', sku='001_', warehouse='__all__',
            as_of=pd.Timestamp('2026-04-01'), free_stock=50, is_current=True)]),
        'transit': pd.DataFrame([dict(supplier='IEK', sku='001_', warehouse='__all__',
            expected_date=pd.Timestamp('2026-04-02'), quantity=40)]),
        'seasonality': pd.DataFrame([dict(supplier='IEK', category='__all__', month_number=m, factor=1.0)
            for m in range(1, 13)]),
    }
    settings = dict(as_of='2026-04-01', lead_time_days={'IEK': 7}, review_period_days=7,
        default_safety_days=2, safety_days_by_category={}, exclude_anomalies=False, restore_stockouts=False)
    return dataset, settings


def first(case):
    return calculate_orders(*case)['orders'].iloc[0]


def test_known_answer_and_contract(case):
    result = calculate_orders(*case)
    row = result['orders'].iloc[0]
    assert list(result['orders']) == ORDER_COLUMNS
    assert row['sku'] == '001_'
    assert row['forecast_qty'] == pytest.approx(140)
    assert row['safety_stock'] == pytest.approx(20)
    assert row['raw_order_qty'] == pytest.approx(70)
    assert row['recommended_qty'] == 72
    assert len(result['forecast']) == 14
    assert result['forecast']['date'].min() == pd.Timestamp('2026-04-02')
    assert result['forecast']['date'].max() == pd.Timestamp('2026-04-15')


@pytest.mark.parametrize('table,column,value', [('stock','free_stock',74), ('transit','quantity',64)])
def test_more_supply_reduces_need(case, table, column, value):
    original = first(case)['raw_order_qty']
    case[0][table].loc[0, column] = value
    assert first(case)['raw_order_qty'] == pytest.approx(original-24)


def test_zero_need_does_not_create_minimum_order(case):
    case[0]['stock'].loc[0, 'free_stock'] = 500
    assert first(case)['recommended_qty'] == 0


def test_arrivals_after_horizon_excluded(case):
    case[0]['transit'].loc[0, 'expected_date'] = pd.Timestamp('2026-04-16')
    assert first(case)['incoming_in_horizon'] == 0
    assert first(case)['raw_order_qty'] == pytest.approx(110)


def test_arrival_on_horizon_included_but_early_deficit_visible(case):
    case[0]['transit'].loc[0, 'expected_date'] = pd.Timestamp('2026-04-15')
    case[0]['transit'].loc[0, 'quantity'] = 200
    row = first(case)
    assert row['recommended_qty'] == 0
    assert row['urgency'] == 'urgent'


@pytest.mark.parametrize('change', ['stale','missing','unknown_scope','missing_lead','missing_moq'])
def test_missing_required_inputs_are_not_zero_orders(case, change):
    data, settings = case
    if change == 'stale': data['stock'].loc[0, 'is_current'] = False
    if change == 'missing': data['stock'] = pd.DataFrame()
    if change == 'unknown_scope': data['stock'].loc[0, 'warehouse'] = None
    if change == 'missing_lead': settings['lead_time_days'] = {}
    if change == 'missing_moq': data['products'].loc[0, 'min_order_qty'] = float('nan')
    row = first(case)
    assert pd.isna(row['recommended_qty'])
    assert row['data_quality'] == 'insufficient'


def test_incomplete_and_future_history_not_used(case):
    extra = pd.DataFrame([
        dict(supplier='IEK', sku='001_', month=pd.Timestamp('2026-02-01'), quantity=100000, is_complete=False),
        dict(supplier='IEK', sku='001_', month=pd.Timestamp('2026-05-01'), quantity=100000, is_complete=True)])
    case[0]['monthly_sales'] = pd.concat([case[0]['monthly_sales'], extra], ignore_index=True)
    assert first(case)['forecast_qty'] == pytest.approx(140)


def test_supplied_seasonal_peak(case):
    case[0]['seasonality'].loc[case[0]['seasonality'].month_number.eq(4), 'factor'] = 2
    assert first(case)['forecast_qty'] == pytest.approx(280)


def test_category_safety_days(case):
    case[1]['safety_days_by_category'] = {'A': 5}
    assert first(case)['safety_stock'] == pytest.approx(50)


def test_inputs_not_mutated(case):
    data, settings = case
    original = deepcopy(data)
    calculate_orders(data, settings)
    for key in data:
        pd.testing.assert_frame_equal(data[key], original[key])


def test_unimplemented_correction_blocks_instead_of_silent_skip(case):
    case[1]['exclude_anomalies'] = True
    case[0]['transactions'] = pd.DataFrame([dict(supplier='IEK', sku='001_', quantity=9999)])
    assert pd.isna(first(case)['recommended_qty'])
    assert 'ещё не реализовано' in first(case)['reason']


@pytest.mark.parametrize('raw,minimum,multiple,expected', [(70,24,12,72),(1,24,12,24),(0,24,12,0),(0.3,0,0.1,0.3)])
def test_rounding(raw, minimum, multiple, expected):
    assert round_order(raw, minimum, multiple) == pytest.approx(expected)


def test_empty_dataset_has_stable_output_schema():
    result = calculate_orders({}, {})
    assert list(result['orders']) == ORDER_COLUMNS
    assert result['orders'].empty
    assert not result['diagnostics'].empty


def test_supplier_level_quality_error_blocks_recommendation(case):
    case[0]['quality_report'] = pd.DataFrame([dict(supplier='IEK', sku=None,
        severity='error', issue='Не подтверждён склад отчёта', source='synthetic')])
    assert pd.isna(first(case)['recommended_qty'])


def test_other_supplier_quality_error_does_not_block(case):
    case[0]['quality_report'] = pd.DataFrame([dict(supplier='Systeme Electric', sku=None,
        severity='error', issue='Other supplier', source='synthetic')])
    assert first(case)['recommended_qty'] == 72


def test_transit_with_different_scope_blocks(case):
    case[0]['transit'].loc[0, 'warehouse'] = 'Other'
    assert pd.isna(first(case)['recommended_qty'])


def test_overdue_transit_requires_reconciliation(case):
    case[0]['transit'].loc[0, 'expected_date'] = pd.Timestamp('2026-03-31')
    assert pd.isna(first(case)['recommended_qty'])


def test_loader_seasonality_alias(case):
    case[0]['seasonality']['category'] = '*all*'
    case[0]['seasonality'].loc[case[0]['seasonality'].month_number.eq(4), 'factor'] = 2
    assert first(case)['forecast_qty'] == 280


def test_historical_unknown_scope_does_not_poison_current_snapshot(case):
    history = dict(supplier='IEK', sku='001_', warehouse=None,
        as_of=pd.Timestamp('2026-03-01'), free_stock=float('nan'), is_current=False)
    case[0]['stock'] = pd.concat([case[0]['stock'], pd.DataFrame([history])], ignore_index=True)
    assert first(case)['recommended_qty'] == 72


def test_explicit_scope_fills_unknown_transit_without_mutation(case):
    case[0]['stock']['warehouse'] = '*all*'
    case[0]['transit']['warehouse'] = None
    case[1]['confirmed_warehouse_scope'] = {'IEK': '__all__'}
    assert first(case)['recommended_qty'] == 72
    assert case[0]['transit']['warehouse'].isna().all()


def test_explicit_scope_does_not_override_known_conflicting_scope(case):
    case[1]['confirmed_warehouse_scope'] = {'IEK': '__all__'}
    case[0]['transit']['warehouse'] = 'Other'
    assert pd.isna(first(case)['recommended_qty'])


def test_loader_unconfirmed_scope_requires_setting(case):
    case[0]['quality_report'] = pd.DataFrame([dict(supplier='IEK', sku=None,
        severity='warning', issue='warehouse_scope_unconfirmed', source='test')])
    assert pd.isna(first(case)['recommended_qty'])
    case[1]['confirmed_warehouse_scope'] = {'IEK': '__all__'}
    assert first(case)['recommended_qty'] == 72


def test_explicit_missing_constraint_default_with_diagnostics(case):
    case[0]['products']['pack_multiple'] = float('nan')
    case[1]['order_constraints_by_supplier'] = {'IEK': {'min_order_qty': 999, 'pack_multiple': 12}}
    result = calculate_orders(*case)
    assert result['orders'].iloc[0]['recommended_qty'] == 72  # existing minimum 24 wins
    assert result['diagnostics'].issue.str.contains('pack_multiple отсутствует').any()
    assert pd.isna(case[0]['products'].iloc[0].pack_multiple)


def test_loader_workbooks_to_engine(tmp_path):
    """Runs when feat/data is integrated; actual adapter, no mocked tables."""
    loader = pytest.importorskip('data_loader')
    from openpyxl import Workbook
    def book(name, rows):
        path = tmp_path / name
        wb = Workbook()
        for row in rows:
            wb.active.append(row)
        wb.save(path)
        wb.close()
        return str(path)
    paths = [
        book('Ежемесячные продажи Systeme.xlsx', [
            ['Номенклатура', 'Номенклатура.Код', 'март 2026', 'апр. 2026'],
            [None, None, 'Количество', 'Количество'], ['Test', '001_', 310, None]]),
        book('Динамика Systeme.xlsx', [
            ['Дата', 'Номер', 'Документ', 'Код', 'Номенклатура', 'Ед.', 'Склад', 'Количество'],
            ['10.03.2026', '1', 'Расходная накладная 1', '001_', 'Test', 'шт', 'Алматы', 310]]),
        book('Товар в пути Systeme на 01.04.2026.xlsx', [
            ['Код 1с', 'Наименование', 'Категория 2026', 'Свободный остаток', 'СЭ в пути 02.04'],
            ['001_', 'Test', 'A', 50, 40]]),
        book('MOQ Systeme.xlsx', [['Номенклатура.Код', 'Номенклатура', 'Кратность'], ['001_', 'Test', 12]]),
        book('Ежемесячные остатки Systeme.xlsx', [
            ['Номенклатура', 'Ед.', 'Номенклатура.Код', 'март 2026'],
            [None, None, None, 'Количество'], [None, None, None, 'нач. остаток'], ['Test', 'шт', '001_', 100]])]
    data = loader.load_data(paths)
    original = deepcopy(data)
    settings = dict(as_of='2026-04-01', lead_time_days={'Systeme Electric': 7},
        review_period_days=7, default_safety_days=2, safety_days_by_category={},
        exclude_anomalies=False, restore_stockouts=False)
    assert pd.isna(calculate_orders(data, settings)['orders'].iloc[0].recommended_qty)
    settings['confirmed_warehouse_scope'] = {'Systeme Electric': '__all__'}
    settings['order_constraints_by_supplier'] = {'Systeme Electric': {'min_order_qty': 0}}
    result = calculate_orders(data, settings)
    row = result['orders'].iloc[0]
    assert row.forecast_qty == pytest.approx(140), row.reason
    assert row.incoming_in_horizon == 40, row.reason
    assert row.recommended_qty == 72, row.reason
    for key in data:
        pd.testing.assert_frame_equal(data[key], original[key])
