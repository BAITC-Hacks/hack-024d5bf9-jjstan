from copy import deepcopy
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import pandas as pd
import pytest

from demo_data import make_demo_dataset, default_settings
from order_review import make_review, merge_visible_edits, review_signature, validate_selection, row_key
from ui_integration import prepare_for_engine, run_calculation, load_uploads, input_signature, upload_key


@pytest.fixture
def review():
    data = make_demo_dataset()
    result, _ = run_calculation(data, default_settings())
    return make_review(result['orders'], data['products'])


def test_real_engine_demo_known_answer():
    result, _ = run_calculation(make_demo_dataset(), default_settings())
    orders = result['orders'].set_index('sku')
    assert orders.loc['DEMO-001', 'recommended_qty'] == 72
    assert orders.loc['DEMO-003', 'recommended_qty'] == 0
    assert pd.isna(orders.loc['DEMO-007', 'recommended_qty'])


def test_edits_follow_sku_across_reordering_and_filters(review):
    subset = review.iloc[[4, 0]].copy()
    subset.iloc[0, subset.columns.get_loc('approved_qty')] = 84
    subset.iloc[1, subset.columns.get_loc('approved_qty')] = 0
    changed = merge_visible_edits(review, subset[['approved_qty', 'selected']])
    assert changed.loc[row_key('IEK', 'DEMO-005'), 'approved_qty'] == 84
    assert changed.loc[row_key('Systeme Electric', 'DEMO-001'), 'approved_qty'] == 0
    assert changed.loc[review.index[1], 'approved_qty'] == review.iloc[1].approved_qty


@pytest.mark.parametrize('qty', [-1, np.inf, np.nan, 73])
def test_invalid_quantities_cannot_be_approved(review, qty):
    review.iloc[0, review.columns.get_loc('selected')] = True
    review.iloc[0, review.columns.get_loc('approved_qty')] = qty
    assert validate_selection(review)[1]


def test_manually_entered_quantity_does_not_bypass_missing_data(review):
    key = row_key('IEK', 'DEMO-007')
    review.loc[key, ['selected', 'approved_qty']] = [True, 5]
    assert 'недостаточно' in validate_selection(review)[1][0]


def test_changes_invalidate_signature_but_sorting_does_not(review):
    original = review_signature(review, 'v1')
    assert review_signature(review.sort_index(ascending=False), 'v1') == original
    review.iloc[0, review.columns.get_loc('approved_qty')] = 84
    assert review_signature(review, 'v1') != original
    assert review_signature(review, 'v2') != review_signature(review, 'v1')


def test_alias_adapter_preserves_sources_and_unknown_scopes():
    data = make_demo_dataset()
    data['seasonality'].category = '*all*'
    data['stock'].warehouse = '*all*'
    data['transit'].warehouse = None
    before = deepcopy(data)
    ready = prepare_for_engine(data, default_settings())
    assert ready['seasonality'].category.eq('__all__').all()
    assert ready['stock'].warehouse.eq('__all__').all()
    assert ready['transit'].warehouse.isna().all()
    for name in data:
        pd.testing.assert_frame_equal(data[name], before[name])


def test_historical_rows_do_not_hide_current_snapshot():
    data = make_demo_dataset()
    old = data['stock'].iloc[[0]].assign(is_current=False, as_of=pd.Timestamp('2026-08-01'), warehouse=None)
    data['stock'] = pd.concat([data['stock'], old], ignore_index=True)
    result, prepared = run_calculation(data, default_settings())
    assert len(prepared['stock']) == len(data['stock']) - 1
    assert result['orders'].iloc[0].recommended_qty == 72


def test_old_snapshot_cannot_be_used_on_new_date():
    settings = default_settings()
    settings['as_of'] = pd.Timestamp('2026-09-23')
    result, _ = run_calculation(make_demo_dataset(), settings)
    assert result['orders'].recommended_qty.isna().all()


def test_bad_zip_is_explained_without_demo_fallback():
    with pytest.raises(ValueError, match='исправным'):
        load_uploads([('IEK.zip', b'broken')])


def test_uploads_keep_supplier_name_and_stay_in_temporary_folder(monkeypatch):
    seen = []
    def fake_loader(paths):
        seen.extend(paths)
        assert all(Path(p).is_file() for p in paths)
        assert Path(paths[0]).name == 'IEK.zip'
        return {'ok': pd.DataFrame()}
    monkeypatch.setattr('data_loader.load_data', fake_loader)
    buf = BytesIO()
    with ZipFile(buf, 'w') as z:
        z.writestr('IEK/readme.txt', 'test')
    load_uploads([('../../IEK.zip', buf.getvalue())])
    assert all(not Path(p).exists() for p in seen)


def test_changed_upload_or_settings_change_calculation_identity():
    base = input_signature('real', [('IEK.zip', b'a')], {'days': 7})
    assert base != input_signature('real', [('IEK.zip', b'b')], {'days': 7})
    assert base != input_signature('real', [('IEK.zip', b'a')], {'days': 8})


def test_explicit_conditions_used_for_missing_values_only():
    data = make_demo_dataset()
    data['products'].loc[0, 'min_order_qty'] = np.nan
    settings = default_settings()
    settings['order_constraints_by_supplier'] = {'Systeme Electric': {'min_order_qty': 24, 'pack_multiple': 5}}
    result, _ = run_calculation(data, settings)
    review = make_review(result['orders'], data['products'], settings)
    review.iloc[0, review.columns.get_loc('selected')] = True
    assert review.iloc[0].min_order_qty == 24
    assert review.iloc[0].pack_multiple == 12
    assert not validate_selection(review)[1]
    assert pd.isna(data['products'].iloc[0].min_order_qty)
    review.iloc[0, review.columns.get_loc('approved_qty')] = 73
    assert validate_selection(review)[1]


def app_test():
    from streamlit.testing.v1 import AppTest
    return AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'), default_timeout=30).run()


def test_app_demo_and_approval_lifecycle():
    at = app_test()
    assert not at.exception
    assert len(at.tabs) == 3
    assert at.metric[0].value == '4'
    assert at.session_state['calculated_settings']['confirmed_warehouse_scope'] == {}
    assert at.session_state['calculated_settings']['order_constraints_by_supplier'] == {}
    at.button(key='select_visible').click().run()
    assert not at.exception
    at.checkbox(key='confirm_review').check().run()
    at.button(key='approve').click().run()
    assert not at.exception
    assert at.session_state['approval']['count'] == 4
    assert b'DEMO-001' in at.session_state['approval']['payload']
    at.number_input(key='lead_se').set_value(10).run()
    assert not at.exception
    assert 'approval' not in at.session_state
    assert at.button(key='approve').disabled
    assert any('пересчёта' in item.value for item in at.warning)


def test_app_mode_change_blocks_previous_demo_export():
    at = app_test()
    at.radio(key='source_mode').set_value('Мои файлы').run()
    assert not at.exception
    assert at.button(key='approve').disabled
    at.button(key='calculate').click().run()
    assert not at.exception
    assert any('Добавьте' in item.value for item in at.error)


def test_app_empty_filters_and_source_view():
    at = app_test()
    at.text_input(key='search').set_value('no-such-sku').run()
    assert not at.exception
    assert any('ничего не найдено' in item.value for item in at.info)
    at.selectbox(key='source_table').set_value('seasonality').run()
    assert not at.exception


def ambiguous_workbook():
    from openpyxl import Workbook
    wb = Workbook()
    for row in [['Номенклатура', 'Номенклатура.Код', 'янв. 2026', 'фев. 2026'],
                [None, None, 'Количество', 'Количество'], ['Systeme in product name', '0007', 12, 20]]:
        wb.active.append(row)
    content = BytesIO()
    wb.save(content)
    wb.close()
    return 'Ежемесячные продажи в количественном выражении за последние 2 года.xlsx', content.getvalue()


def test_ambiguous_xlsx_supplier_is_explicit_and_not_inferred_from_cells():
    name, payload = ambiguous_workbook()
    with pytest.raises(ValueError, match='Выберите поставщика'):
        load_uploads([(name, payload)])
    identity = upload_key(name, payload)
    data = load_uploads([(name, payload)], {identity: 'IEK'})
    assert data['products'].supplier.tolist() == ['IEK']
    assert data['products'].sku.tolist() == ['0007']
    assert not data['quality_report'].issue.eq('unrecognized_supplier_or_file_type').any()
    assert data['products'].name.iloc[0] == 'Systeme in product name'
    with pytest.raises(ValueError, match='списка'):
        load_uploads([(name, payload)], {identity: '../../Systeme Electric'})
    one = input_signature('real', [(name, payload)], {}, {identity: 'IEK'})
    two = input_signature('real', [(name, payload)], {}, {identity: 'Systeme Electric'})
    assert one != two


def test_app_supplier_change_invalidates_calculation_approval_and_ai(monkeypatch):
    import streamlit
    from types import SimpleNamespace
    name, payload = ambiguous_workbook()
    monkeypatch.setattr(streamlit, 'file_uploader', lambda *a, **kw: [SimpleNamespace(name=name, getvalue=lambda: payload)])
    at = app_test()
    at.radio(key='source_mode').set_value('Мои файлы').run()
    key = 'upload_supplier_' + upload_key(name, payload)
    at.selectbox(key=key).set_value('IEK').run()
    at.button(key='calculate').click().run()
    assert not at.exception
    previous_id = at.session_state['calculation_id']
    assert at.session_state['dataset']['products'].supplier.eq('IEK').all()
    at.session_state['approval'] = {'signature': 'previous'}
    at.session_state['ai_cache'] = {'old': {'status': 'ok'}}
    at.selectbox(key=key).set_value('Systeme Electric').run()
    assert not at.exception
    assert 'approval' not in at.session_state
    assert at.session_state['ai_cache'] == {}
    assert at.button(key='approve').disabled
    assert at.button(key='explain_sku').disabled
    at.button(key='calculate').click().run()
    assert not at.exception
    assert at.session_state['calculation_id'] != previous_id
    assert at.session_state['dataset']['products'].supplier.eq('Systeme Electric').all()
