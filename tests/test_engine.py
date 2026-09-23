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


def test_malformed_transactions_block_instead_of_silent_skip(case):
    case[1]['exclude_anomalies'] = True
    case[0]['transactions'] = pd.DataFrame([dict(supplier='IEK', sku='001_', quantity=9999)])
    assert pd.isna(first(case)['recommended_qty'])
    assert 'недостаточно полей' in first(case)['reason']


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


def test_direct_engine_rejects_old_snapshot(case):
    case[0]['stock']['as_of'] = pd.Timestamp('2026-03-31')
    row = first(case)
    assert pd.isna(row.recommended_qty)
    assert 'снимка' in row.reason and 'as_of' in row.reason


@pytest.fixture
def demand_case(case):
    data, settings = deepcopy(case)
    months = pd.date_range('2025-10-01', periods=6, freq='MS')
    tx = []
    for month in months:
        for day in range(1, 21):
            tx.append(dict(supplier='IEK', sku='001_', date=month+pd.Timedelta(days=day-1),
                document_id=f'{month:%Y%m}-{day}', warehouse='__all__',
                quantity=10.0, transaction_type='sale', customer_id=None, unit='шт'))
    data['transactions'] = pd.DataFrame(tx)
    data['monthly_sales'] = pd.DataFrame([dict(supplier='IEK', sku='001_', month=m,
        quantity=200.0, is_complete=True) for m in months])
    data['stock']['free_stock'] = 0.0
    data['products']['min_order_qty'] = 0.0
    data['products']['pack_multiple'] = 1.0
    data['transit'] = data['transit'].iloc[:0]
    settings['exclude_anomalies'] = True
    settings['enable_trend'] = False
    return data, settings


def reconcile_months(data):
    tx = data['transactions']
    totals = tx.assign(month=tx.date.dt.to_period('M').dt.to_timestamp()).groupby('month').quantity.sum()
    data['monthly_sales']['quantity'] = data['monthly_sales'].month.map(totals)


def add_spike(data, quantity=10000.0, when='2026-03-15', document='bulk', **extra):
    entry = dict(supplier='IEK', sku='001_', date=pd.Timestamp(when), document_id=document,
        warehouse='__all__', quantity=quantity, transaction_type='sale', customer_id=None, unit='шт')
    entry.update(extra)
    data['transactions'] = pd.concat([data['transactions'], pd.DataFrame([entry])], ignore_index=True)
    reconcile_months(data)


def test_split_document_spike_does_not_inflate_regular_demand(demand_case):
    data, settings = demand_case
    original = first(demand_case).forecast_qty
    for _ in range(20):
        add_spike(data, quantity=500)
    result = calculate_orders(data, settings)
    assert result['orders'].iloc[0].forecast_qty == pytest.approx(original)
    assert len(result['anomalies']) == 1
    assert result['anomalies'].iloc[0].quantity == 10000
    assert result['anomalies'].iloc[0].excluded
    assert result['diagnostics'].issue.str.contains('Нет customer_id').any()


def test_unreconciled_spike_is_not_removed(demand_case):
    data, settings = demand_case
    add_spike(data)
    data['monthly_sales'].loc[data['monthly_sales'].month.eq(pd.Timestamp('2026-03-01')), 'quantity'] += 1
    result = calculate_orders(data, settings)
    assert pd.isna(result['orders'].iloc[0].recommended_qty)
    assert not result['anomalies'].excluded.any()
    assert 'не совпадают' in result['orders'].iloc[0].reason


def test_return_stays_negative_and_is_not_anomaly(demand_case):
    data, settings = demand_case
    add_spike(data, quantity=-100, document='return', transaction_type='return')
    baseline = first(demand_case).forecast_qty
    add_spike(data)
    result = calculate_orders(data, settings)
    assert result['orders'].iloc[0].forecast_qty == pytest.approx(baseline)
    assert result['anomalies'].document_id.tolist() == ['bulk']
    assert data['transactions'].loc[data['transactions'].document_id.eq('return'), 'quantity'].iloc[0] == -100


def test_repeated_large_demand_is_retained(demand_case):
    data, settings = demand_case
    for i, when in enumerate(['2026-03-01','2026-03-10','2026-03-20']):
        add_spike(data, quantity=100, when=when, document=f'repeat-{i}')
    enabled = calculate_orders(data, settings)
    assert len(enabled['anomalies']) == 3
    assert not enabled['anomalies'].excluded.any()
    disabled = calculate_orders(data, dict(settings, exclude_anomalies=False))
    assert enabled['orders'].iloc[0].forecast_qty == disabled['orders'].iloc[0].forecast_qty


def test_growth_survives_anomaly_detection_and_is_bounded(demand_case):
    data, settings = demand_case
    for i, month in enumerate(data['monthly_sales'].month):
        mask = data['transactions'].date.dt.to_period('M').eq(month.to_period('M'))
        data['transactions'].loc[mask,'quantity'] = (10+i*3)*month.days_in_month/20
    reconcile_months(data)
    flat = first(demand_case).forecast_qty
    settings['enable_trend'] = True
    result = calculate_orders(data, settings)
    grown = result['orders'].iloc[0].forecast_qty
    assert flat < grown <= flat*1.5 + 1e-8
    assert not result['anomalies'].excluded.any()
    off = calculate_orders(data, dict(settings, exclude_anomalies=False))
    assert off['orders'].iloc[0].forecast_qty == pytest.approx(grown)


def test_seasonal_growth_coefficient_not_counted_twice(demand_case):
    data, settings = demand_case
    for i, month in enumerate(data['monthly_sales'].month):
        factor = 1+i
        data['seasonality'].loc[data['seasonality'].month_number.eq(month.month),'factor'] = factor
        data['monthly_sales'].loc[data['monthly_sales'].month.eq(month),'quantity'] = 10*month.days_in_month*factor
    settings.update(exclude_anomalies=False, enable_trend=True)
    data['products']['growth_coefficient_raw'] = 999
    assert first(demand_case).forecast_qty == pytest.approx(140)


def stockout_rows(*intervals, **extra):
    return pd.DataFrame([dict(supplier='IEK', sku='001_', warehouse='__all__',
        start_date=pd.Timestamp(start), end_date=pd.Timestamp(end), **extra) for start,end in intervals])


def test_stockout_increases_intensity_and_overlaps_count_once(case):
    data, settings = case
    data['monthly_sales']['quantity'] = 160.0  # 16 in-stock days at 10/day
    raw = first(case).forecast_qty
    settings['restore_stockouts'] = True
    data['stockouts'] = stockout_rows(('2026-03-01','2026-03-10'), ('2026-03-06','2026-03-15'))
    corrected = first(case).forecast_qty
    assert corrected == pytest.approx(140)
    assert corrected > raw
    data['stockouts'] = stockout_rows(('2026-03-01','2026-03-15'))
    assert first(case).forecast_qty == corrected


def test_stockout_future_is_ignored_and_inputs_unchanged(case):
    data, settings = case
    settings['restore_stockouts'] = True
    data['stockouts'] = stockout_rows(('2026-04-02','2026-06-01'))
    before = deepcopy(data)
    assert first(case).recommended_qty == 72
    for key in data:
        pd.testing.assert_frame_equal(data[key],before[key])


@pytest.mark.parametrize('field,value', [('warehouse','Other'),('unit','м'),('confirmed',False)])
def test_stockout_bad_scope_units_or_confirmation_blocks(case, field, value):
    data, settings = case
    settings['restore_stockouts'] = True
    data['stockouts'] = stockout_rows(('2026-03-01','2026-03-15'))
    data['stockouts'][field] = value
    assert pd.isna(first(case).recommended_qty)


def test_empty_stockouts_preserve_baseline_and_report_absence(case):
    case[1]['restore_stockouts'] = True
    result = calculate_orders(*case)
    assert result['orders'].iloc[0].recommended_qty == 72
    assert result['diagnostics'].issue.str.contains('Нет подтверждённых stockout').any()


def test_algorithms_do_not_mutate_input(demand_case):
    add_spike(demand_case[0])
    before = deepcopy(demand_case)
    calculate_orders(*demand_case)
    assert demand_case[1] == before[1]
    for key in before[0]:
        pd.testing.assert_frame_equal(before[0][key], demand_case[0][key])


def test_spike_in_future_or_incomplete_month_does_not_affect_fit(demand_case):
    baseline = first(demand_case).forecast_qty
    add_spike(demand_case[0], when='2026-04-10')
    assert first(demand_case).forecast_qty == pytest.approx(baseline)


def test_invalid_return_sign_blocks(demand_case):
    add_spike(demand_case[0], quantity=100, transaction_type='return')
    assert pd.isna(first(demand_case).recommended_qty)
    assert 'знак' in first(demand_case).reason


def test_scope_does_not_mix_warehouses(demand_case):
    data, settings = demand_case
    data['stock']['warehouse'] = 'Almaty'
    data['transactions']['warehouse'] = 'Astana'
    assert pd.isna(first(demand_case).recommended_qty)


def test_fully_out_of_stock_month_uses_observed_intensity(case):
    data, settings = case
    settings['restore_stockouts'] = True
    data['monthly_sales'] = pd.concat([data['monthly_sales'], pd.DataFrame([
        dict(supplier='IEK',sku='001_',month=pd.Timestamp('2026-02-01'),quantity=0,is_complete=True)])], ignore_index=True)
    data['stockouts'] = stockout_rows(('2026-02-01','2026-02-28'))
    assert first(case).forecast_qty == pytest.approx(140)


def test_no_observed_stockout_days_cannot_create_demand(case):
    data, settings = case
    settings['restore_stockouts'] = True
    data['monthly_sales']['quantity'] = 0.0
    data['stockouts'] = stockout_rows(('2026-03-01','2026-04-20'))
    assert pd.isna(first(case).recommended_qty)
    assert 'Нет дней доступности' in first(case).reason


def test_sales_during_full_month_stockout_require_reconciliation(case):
    case[1]['restore_stockouts'] = True
    case[0]['stockouts'] = stockout_rows(('2026-03-01','2026-03-31'))
    assert pd.isna(first(case).recommended_qty)


def test_quality_mismatch_is_not_waived_by_algorithm_flags(demand_case):
    data, settings = demand_case
    settings.update(restore_stockouts=True, enable_trend=True)
    data['quality_report'] = pd.DataFrame([dict(supplier='IEK',sku='001_',severity='error',
        issue='monthly_transaction_mismatch:2026-03',source='cross_source')])
    row = first(demand_case)
    assert pd.isna(row.recommended_qty)
    assert 'monthly_transaction_mismatch:2026-03' in row.reason


def test_capabilities_are_explicit():
    from engine import ENGINE_CAPABILITIES
    for feature in ['exclude_anomalies','restore_stockouts','trend','strict_snapshot_date']:
        assert ENGINE_CAPABILITIES[feature] is True


def test_known_customer_split_purchase_is_detected_once(demand_case):
    data, settings = demand_case
    data['transactions']['customer_id'] = 'regular'
    baseline = first(demand_case).forecast_qty
    for i in range(20):
        add_spike(data, quantity=10, document=f'client-bulk-{i}', customer_id='bulk-client')
    result = calculate_orders(data, settings)
    assert len(result['anomalies']) == 20
    assert result['anomalies'].excluded.all()
    assert result['anomalies'].quantity.sum() == 200
    assert result['orders'].iloc[0].forecast_qty == pytest.approx(baseline)


def test_unknown_customer_documents_are_not_falsely_clustered(demand_case):
    data, settings = demand_case
    baseline = first(demand_case).forecast_qty
    for i in range(20):
        add_spike(data, quantity=10, document=f'unknown-{i}')
    result = calculate_orders(data, settings)
    assert result['anomalies'].empty
    assert result['orders'].iloc[0].forecast_qty > baseline


def test_document_and_customer_detection_do_not_double_subtract(demand_case):
    data, settings = demand_case
    data['transactions']['customer_id'] = 'regular'
    baseline = first(demand_case).forecast_qty
    add_spike(data, customer_id='bulk-client')
    result = calculate_orders(data, settings)
    assert len(result['anomalies']) == 1
    assert result['orders'].iloc[0].forecast_qty == pytest.approx(baseline)


@pytest.mark.parametrize('months,allowed', [
    ('2024-01,2025-09',True), ('2026-04',True),
    ('2025-09,2026-03',False), ('2026-03',False),
    ('',False), ('2026-13',False), ('unknown',False), ('2024-01,',False)])
def test_mismatch_applies_only_to_used_months(demand_case, months, allowed):
    data, settings = demand_case
    settings.update(enable_trend=True, restore_stockouts=True)
    issue = 'monthly_transaction_mismatch:' + months
    data['quality_report'] = pd.DataFrame([dict(supplier='IEK',sku='001_',severity='error',issue=issue,source='cross_source')])
    original = deepcopy(data)
    result = calculate_orders(data, settings)
    assert bool(result['orders'].recommended_qty.notna().iloc[0]) == allowed
    assert ((result['diagnostics'].issue == issue) & (result['diagnostics'].severity == 'error')).any()
    if allowed:
        assert result['diagnostics'].issue.str.contains('неприменима').any()
        assert result['orders'].iloc[0].data_quality == 'warning'
    for key in data:
        pd.testing.assert_frame_equal(data[key], original[key])


def test_period_exception_never_covers_unit_errors(demand_case):
    data, settings = demand_case
    data['quality_report'] = pd.DataFrame([
        dict(supplier='IEK',sku='001_',severity='error',issue='monthly_transaction_mismatch:2024-01'),
        dict(supplier='IEK',sku='001_',severity='error',issue='conflicting_units')])
    assert pd.isna(first(demand_case).recommended_qty)
    assert 'conflicting_units' in first(demand_case).reason


def test_incomplete_september_is_not_used_even_with_quality_exception(demand_case):
    data, settings = demand_case
    data['monthly_sales'] = pd.concat([data['monthly_sales'],pd.DataFrame([
        dict(supplier='IEK',sku='001_',month=pd.Timestamp('2026-04-01'),quantity=999999,is_complete=False)])],ignore_index=True)
    baseline = first(demand_case).forecast_qty
    data['quality_report'] = pd.DataFrame([dict(supplier='IEK',sku='001_',severity='error',issue='monthly_transaction_mismatch:2026-04')])
    assert first(demand_case).forecast_qty == baseline
