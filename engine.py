"""StockPilot: first integration milestone, using the TEAM_PLAN contract.

No I/O, UI, or network calls. Anomalies, trend and stockout restoration are
explicitly outside this first milestone; requested unsupported corrections
block affected recommendations rather than silently being ignored.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd


ORDER_COLUMNS = [
    'supplier', 'sku', 'name', 'category', 'unit', 'stock_as_of',
    'free_stock', 'incoming_in_horizon', 'forecast_qty', 'safety_stock',
    'raw_order_qty', 'recommended_qty', 'urgency', 'reason', 'data_quality',
]
FORECAST_COLUMNS = ['supplier', 'sku', 'date', 'predicted_qty']
ANOMALY_COLUMNS = ['supplier', 'sku', 'date', 'document_id', 'quantity', 'excluded', 'reason']
DIAGNOSTIC_COLUMNS = ['supplier', 'sku', 'severity', 'issue']


def _rows(dataset, table, supplier, sku=None):
    frame = dataset.get(table)
    if frame is None or frame.empty:
        return pd.DataFrame()
    if 'supplier' not in frame or (sku is not None and 'sku' not in frame):
        raise ValueError(f'{table}: отсутствуют ключи supplier/sku')
    mask = frame['supplier'].eq(supplier)
    if sku is not None:
        mask &= frame['sku'].eq(sku)
    return frame.loc[mask].copy()


def _number(value, label, positive=False, integer=False):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f'{label}: требуется число, не bool')
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError(f'{label}: значение отсутствует или не является числом') from None
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f'{label}: недопустимое значение')
    if integer and not value.is_integer():
        raise ValueError(f'{label}: требуется целое число дней')
    return value


def _date(value, label):
    result = pd.to_datetime(value, errors='coerce')
    if pd.isna(result):
        raise ValueError(f'{label}: дата отсутствует или некорректна')
    if result.tzinfo is not None:
        raise ValueError(f'{label}: контракт требует локальную дату без timezone')
    return result.normalize()


def round_order(raw_qty, min_order_qty, pack_multiple):
    """Return zero for no need; otherwise respect minimum and pack multiple."""
    raw_qty = _number(raw_qty, 'raw_order_qty')
    if raw_qty == 0:
        return 0.0
    minimum = _number(min_order_qty, 'min_order_qty')
    multiple = _number(pack_multiple, 'pack_multiple', positive=True)
    packs = max(raw_qty, minimum) / multiple
    nearest = round(packs)
    if math.isclose(packs, nearest, rel_tol=0, abs_tol=1e-10):
        packs = nearest
    return float(math.ceil(packs) * multiple)


def calculate_orders(dataset: dict, settings: dict) -> dict:
    """Calculate a baseline order with diagnostics, preserving input tables.

    Dates in the forecast are (as_of, as_of + lead + review], inclusive on
    the right. Pending inbound on as_of is included and assumed not to be
    already reflected in free_stock. Earlier overdue inbound is blocked.
    """
    orders, forecasts, diagnostics = [], [], []
    products = dataset.get('products', pd.DataFrame())

    def note(supplier, sku, severity, issue):
        diagnostics.append(dict(supplier=supplier, sku=sku, severity=severity, issue=issue))

    if not isinstance(products, pd.DataFrame):
        raise ValueError('products должен быть DataFrame')
    if products.empty:
        note(None, None, 'warning', 'Нет товаров для расчёта')
    elif not {'supplier', 'sku'}.issubset(products.columns):
        raise ValueError('products: отсутствуют supplier/sku')

    for _, product in products.iterrows():
        supplier, sku = product['supplier'], product['sku']
        record = {column: np.nan for column in ORDER_COLUMNS}
        record.update({key: product.get(key) for key in ['supplier', 'sku', 'name', 'category', 'unit']})
        record.update(stock_as_of=pd.NaT, urgency='unknown', data_quality='insufficient', reason='')
        start_notes = len(diagnostics)
        try:
            if not isinstance(supplier, str) or not supplier or not isinstance(sku, str) or not sku:
                raise ValueError('supplier и sku должны быть непустыми строками')
            if len(products.loc[products.supplier.eq(supplier) & products.sku.eq(sku)]) != 1:
                raise ValueError('Повторный ключ (supplier, sku) в products')
            if pd.isna(product.get('unit')) or not str(product.get('unit')).strip():
                raise ValueError('Не подтверждена единица учёта')
            as_of = _date(settings.get('as_of'), 'as_of')
            lead = int(_number(settings.get('lead_time_days', {}).get(supplier), 'lead_time_days', integer=True))
            review = int(_number(settings.get('review_period_days'), 'review_period_days', integer=True))
            horizon = lead + review
            if horizon <= 0:
                raise ValueError('Горизонт расчёта должен быть положительным')
            end = as_of + pd.Timedelta(days=horizon)
            dates = pd.date_range(as_of + pd.Timedelta(days=1), end, freq='D')
            quality = dataset.get('quality_report', pd.DataFrame())
            if not quality.empty:
                quality = quality.loc[
                    (quality['supplier'].isna() | quality['supplier'].eq(supplier))
                    & (quality['sku'].isna() | quality['sku'].eq(sku))
                ]
            if not quality.empty:
                for _, finding in quality.iterrows():
                    severity = finding.get('severity', 'warning')
                    note(supplier, sku, severity, str(finding.get('issue', 'Ошибка качества данных')))
                if quality['severity'].eq('error').any():
                    raise ValueError('Загрузчик сообщил об ошибках данных для SKU')

            history = _rows(dataset, 'monthly_sales', supplier, sku)
            required = {'month', 'quantity', 'is_complete'}
            if history.empty or not required.issubset(history.columns):
                raise ValueError('Нет истории завершённых месяцев')
            history['month'] = pd.to_datetime(history['month'], errors='coerce')
            history['quantity'] = pd.to_numeric(history['quantity'], errors='coerce')
            history = history.loc[
                history['is_complete'].eq(True)
                & (history['month'] + pd.offsets.MonthEnd(0) < as_of)
            ].sort_values('month')
            if history['month'].duplicated().any():
                raise ValueError('Повторный месяц в monthly_sales')
            history = history.tail(6)
            if history.empty or not np.isfinite(history['quantity']).all() or history['quantity'].lt(0).any():
                raise ValueError('Нет пригодной истории: пропуски или отрицательный месячный спрос')
            if len(history) < 3:
                note(supplier, sku, 'warning', 'Меньше трёх полных месяцев: короткая история')

            for flag, table, label in [
                ('exclude_anomalies', 'transactions', 'Исключение аномалий'),
                ('restore_stockouts', 'stockouts', 'Восстановление stockout'),
            ]:
                if settings.get(flag, False) and not _rows(dataset, table, supplier, sku).empty:
                    raise ValueError(f'{label} ещё не реализовано в первом этапе; требуется следующий этап engine')

            season = _rows(dataset, 'seasonality', supplier)
            factors = {month: 1.0 for month in range(1, 13)}
            if not season.empty:
                specific = season.loc[season['category'].eq(product.get('category'))]
                season = specific if not specific.empty else season.loc[season['category'].eq('__all__')]
                if season['month_number'].duplicated().any():
                    raise ValueError('Дубли месяцев в сезонном профиле')
                for _, factor in season.iterrows():
                    month = int(_number(factor['month_number'], 'month_number', positive=True, integer=True))
                    if month > 12:
                        raise ValueError('month_number должен быть от 1 до 12')
                    factors[month] = _number(factor['factor'], 'seasonality.factor', positive=True)
                if len(season) < 12:
                    note(supplier, sku, 'warning', 'Неполный сезонный профиль: для отсутствующих месяцев коэффициент 1')
            else:
                note(supplier, sku, 'warning', 'Сезонный профиль отсутствует: коэффициент 1')
            baseline = float((history['quantity'] / history['month'].dt.days_in_month
                              / history['month'].dt.month.map(factors)).mean())
            predicted = pd.Series([baseline * factors[day.month] for day in dates], index=dates)
            forecast_qty = float(predicted.sum())
            safety_days = _number(settings.get('safety_days_by_category', {}).get(
                product.get('category'), settings.get('default_safety_days')), 'safety_days')
            safety = float(predicted.mean() * safety_days)
            record.update(forecast_qty=forecast_qty, safety_stock=safety)
            forecasts.extend(dict(supplier=supplier, sku=sku, date=day, predicted_qty=float(qty))
                             for day, qty in predicted.items())

            stock = _rows(dataset, 'stock', supplier, sku)
            if stock.empty:
                raise ValueError('Нет текущего свободного остатка')
            stock['as_of'] = pd.to_datetime(stock['as_of'], errors='coerce')
            stock = stock.loc[stock['as_of'].le(as_of)]
            if stock.empty or stock['warehouse'].isna().any() or stock['warehouse'].nunique() != 1:
                raise ValueError('Не определён единый контур склада для остатка')
            latest = stock.loc[stock['as_of'].eq(stock['as_of'].max())]
            if len(latest) != 1:
                raise ValueError('Неоднозначный снимок остатка')
            snapshot = latest.iloc[0]
            record['stock_as_of'] = snapshot['as_of']
            if pd.isna(snapshot['is_current']) or snapshot['is_current'] != True:
                raise ValueError('Остаток исторический: нужен актуальный снимок')
            free = _number(snapshot['free_stock'], 'free_stock')
            record['free_stock'] = free

            incoming = _rows(dataset, 'transit', supplier, sku)
            receipts = pd.Series(0.0, index=dates)
            today_receipts = 0.0
            if not incoming.empty:
                incoming['quantity'] = pd.to_numeric(incoming['quantity'], errors='coerce')
                if not np.isfinite(incoming['quantity']).all() or incoming['quantity'].lt(0).any():
                    raise ValueError('Некорректное количество транзита')
                incoming = incoming.loc[incoming['quantity'].gt(0)]
                incoming['expected_date'] = pd.to_datetime(incoming['expected_date'], errors='coerce').dt.normalize()
                if incoming['expected_date'].isna().any():
                    raise ValueError('Неизвестна дата поступления')
                if incoming['warehouse'].isna().any() or not incoming['warehouse'].eq(snapshot['warehouse']).all():
                    raise ValueError('Контур склада транзита не совпадает с остатком')
                if incoming['expected_date'].lt(as_of).any():
                    raise ValueError('Просроченный транзит: уточните фактическое поступление')
                today_receipts = float(incoming.loc[incoming.expected_date.eq(as_of), 'quantity'].sum())
                arrivals = incoming.groupby('expected_date')['quantity'].sum()
                receipts = arrivals.reindex(dates, fill_value=0.0)
            total_incoming = float(receipts.sum() + today_receipts)
            raw = max(0.0, forecast_qty + safety - free - total_incoming)
            recommended = round_order(raw, product.get('min_order_qty'), product.get('pack_multiple'))
            balance = free + today_receipts + receipts.cumsum() - predicted.cumsum()
            shortage = balance[balance < -1e-9]
            if not shortage.empty:
                urgency = 'urgent'
                shortage_day = shortage.index[0]
                note(supplier, sku, 'warning', f'Риск дефицита {shortage_day.date()} без нового заказа')
                if shortage_day < as_of + pd.Timedelta(days=lead):
                    note(supplier, sku, 'warning', 'Обычная новая поставка не успевает до первого дефицита')
            else:
                urgency = 'normal' if recommended > 0 else 'none'
            note(supplier, sku, 'warning', 'Первый этап: базовый прогноз с сезонностью; устойчивый тренд ещё не реализован')
            record.update(incoming_in_horizon=total_incoming, raw_order_qty=raw,
                          recommended_qty=recommended, urgency=urgency,
                          reason=(f'max(0, {forecast_qty:.4f} + {safety:.4f} - {free:.4f} - '
                                  f'{total_incoming:.4f}) = {raw:.4f}; '
                                  f'с учётом минимума и кратности: {recommended:.4f} {product["unit"]}'),
                          data_quality='warning' if len(diagnostics) > start_notes else 'ok')
        except (ValueError, KeyError, TypeError) as error:
            record['reason'] = str(error)
            note(supplier, sku, 'error', str(error))
        orders.append(record)

    return {
        'orders': pd.DataFrame(orders, columns=ORDER_COLUMNS),
        'forecast': pd.DataFrame(forecasts, columns=FORECAST_COLUMNS),
        'anomalies': pd.DataFrame(columns=ANOMALY_COLUMNS),
        'diagnostics': pd.DataFrame(diagnostics, columns=DIAGNOSTIC_COLUMNS),
    }
