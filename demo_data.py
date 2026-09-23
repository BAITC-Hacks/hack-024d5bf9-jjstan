"""Explicit synthetic data, always calculated by the real teammate engine."""
import pandas as pd

DEMO_DATE = pd.Timestamp('2026-09-22')
SCHEMAS = {
    'products': 'supplier sku supplier_article name category unit min_order_qty pack_multiple'.split(),
    'monthly_sales': 'supplier sku month quantity is_complete'.split(),
    'transactions': 'supplier sku date document_id customer_id warehouse quantity transaction_type'.split(),
    'stock': 'supplier sku warehouse as_of free_stock is_current'.split(),
    'transit': 'supplier sku warehouse expected_date quantity'.split(),
    'seasonality': 'supplier category month_number factor'.split(),
    'stockouts': 'supplier sku warehouse start_date end_date'.split(),
    'quality_report': 'supplier sku severity issue source'.split(),
}


def make_demo_dataset():
    # All names, codes, quantities and conditions here are fictional.
    specs = [
        ('Systeme Electric', 'DEMO-001', 'Розетка Atlas · белая', 'A', 'шт', 10, 50, 40, 24, 12),
        ('Systeme Electric', 'DEMO-002', 'Выключатель Atlas · графит', 'A', 'шт', 4, 8, 0, 10, 10),
        ('Systeme Electric', 'DEMO-003', 'Рамка на 2 поста', 'B', 'шт', 2, 180, 0, 10, 10),
        ('Systeme Electric', 'DEMO-004', 'Розетка USB · алюминий', 'B', 'шт', 1, 10, 30, 1, 1),
        ('IEK', 'DEMO-005', 'Автоматический выключатель C16', 'A', 'шт', 6, 20, 24, 12, 12),
        ('IEK', 'DEMO-006', 'Кабель монтажный', 'B', 'м', 3.5, 30, 0, 5, 0.5),
        ('IEK', 'DEMO-007', 'Светильник аварийный', 'C', 'шт', 0.6, None, 0, 1, 1),
        ('IEK', 'DEMO-008', 'Клеммная колодка', 'C', 'шт', 0, 60, 0, 10, 10),
    ]
    rows = {key: [] for key in SCHEMAS}
    for supplier, sku, name, category, unit, daily, stock, incoming, minimum, pack in specs:
        rows['products'].append(dict(supplier=supplier, sku=sku, supplier_article=sku,
            name=name, category=category, unit=unit, min_order_qty=minimum, pack_multiple=pack))
        for month in pd.date_range('2026-03-01', '2026-08-01', freq='MS'):
            factor = 1.25 if month.month in [10, 11] else 1.0
            rows['monthly_sales'].append(dict(supplier=supplier, sku=sku, month=month,
                quantity=daily * month.days_in_month * factor, is_complete=True))
        if stock is not None:
            rows['stock'].append(dict(supplier=supplier, sku=sku, warehouse='__all__',
                as_of=DEMO_DATE, free_stock=stock, is_current=True))
        if incoming:
            rows['transit'].append(dict(supplier=supplier, sku=sku, warehouse='__all__',
                expected_date=DEMO_DATE + pd.Timedelta(days=2), quantity=incoming))
    for supplier in ['Systeme Electric', 'IEK']:
        for month in range(1, 13):
            rows['seasonality'].append(dict(supplier=supplier, category='__all__',
                month_number=month, factor=1.25 if supplier == 'IEK' and month in [10, 11] else 1.0))
    return {key: pd.DataFrame(values, columns=SCHEMAS[key]) for key, values in rows.items()}


def default_settings():
    return dict(as_of=DEMO_DATE.date(), lead_time_days={'Systeme Electric': 7, 'IEK': 7},
        review_period_days=7, default_safety_days=2, safety_days_by_category={},
        exclude_anomalies=False, restore_stockouts=False)


DEMO_SCENARIOS = {
    'overview': dict(label='Обзор · 8 товаров',
        description='8 вымышленных товаров двух поставщиков: расчёт, правка, утверждение и CSV.'),
    'anomaly': dict(label='Разовый крупный заказ', example='anomaly_off', flag='exclude_anomalies',
        description='К регулярным продажам добавлена одна сделка на 10 000 единиц. Сравните заказ с исключением этой сделки и без него.',
        before='Без исключения аномалии', after='С исключением аномалии'),
    'stockout': dict(label='Отсутствие товара (stockout)', example='stockout_off', flag='restore_stockouts',
        description='В марте товар отсутствовал 15 дней. Сравните заказ по фактическим продажам и с восстановлением упущенного спроса.',
        before='Без компенсации stockout', after='С компенсацией stockout'),
    'trend': dict(label='Устойчивый рост спроса', example='trend_off', flag='enable_trend',
        description='Среднедневные продажи растут шесть месяцев подряд. Сравните заказ по среднему спросу и с учётом тренда.',
        before='Без учёта тренда', after='С учётом тренда'),
}


def make_demo_case(scenario='overview'):
    """Reuse verified engine examples; each selection gets fresh synthetic data."""
    spec = DEMO_SCENARIOS[scenario]
    if scenario == 'overview':
        return make_demo_dataset(), dict(default_settings(), enable_trend=True)
    from engine_examples import make_examples
    case = make_examples()[spec['example']]
    case['dataset']['products']['name'] = 'Синтетический товар · ' + spec['label']
    return case['dataset'], case['settings']
