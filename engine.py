"""Deterministic replenishment calculation; no I/O, network or LLM dependency."""
from __future__ import annotations

import math
import re
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
ENGINE_CAPABILITIES = {
    'version': '2.0-local', 'exclude_anomalies': True,
    'restore_stockouts': True, 'trend': True, 'strict_snapshot_date': True,
}


def _check_units(frame, unit, label):
    if 'unit' in frame and frame['unit'].dropna().ne(unit).any():
        raise ValueError(f'{label}: единица не совпадает с products.unit')


def _regular_history(history, tx, supplier, sku, unit, scope, anomalies, emit):
    """Remove isolated document spikes only from reconciled monthly quantities."""
    if tx.empty:
        emit('warning', 'Нет transactions: аномалии не проверены; месячная история сохранена')
        return history
    required = {'date', 'document_id', 'warehouse', 'quantity', 'transaction_type'}
    if not required.issubset(tx.columns):
        raise ValueError('transactions: недостаточно полей для анализа сделок')
    tx = tx.copy()
    tx['date'] = pd.to_datetime(tx['date'], errors='coerce').dt.normalize()
    if tx['date'].isna().any():
        raise ValueError('transactions: неизвестная дата сделки')
    tx['month'] = tx['date'].dt.to_period('M').dt.to_timestamp()
    tx = tx.loc[tx['month'].isin(history['month'])].copy()
    if tx.empty:
        emit('warning', 'Нет сделок в выбранных полных месяцах; аномалии не проверены')
        return history
    _check_units(tx, unit, 'transactions')
    tx['quantity'] = pd.to_numeric(tx['quantity'], errors='coerce')
    if not np.isfinite(tx['quantity']).all():
        raise ValueError('transactions: неизвестное количество')
    sale = tx['transaction_type'].eq('sale').fillna(False)
    returned = tx['transaction_type'].eq('return').fillna(False)
    if (~(sale | returned)).any() or (sale & tx.quantity.lt(0)).any() or (returned & tx.quantity.gt(0)).any():
        raise ValueError('transactions: неподтверждённый тип или знак операции')
    tx['warehouse'] = tx['warehouse'].map(_scope)
    if tx.warehouse.isna().any() or (scope != '__all__' and not tx.warehouse.eq(scope).all()):
        raise ValueError('transactions: неизвестный или несовместимый склад сделок')
    if tx.document_id.isna().any() or tx.document_id.astype(str).str.strip().eq('').any():
        raise ValueError('transactions: нет document_id для объединения строк сделки')
    if 'customer_id' not in tx or tx.customer_id.isna().any():
        emit('warning', 'Нет customer_id у части/всех сделок: документы не объединяются по клиенту')
    # Returns never enter the positive-order distribution; their signed amounts remain in history.
    deals = tx.loc[sale & tx.quantity.gt(0)].groupby(
        ['document_id', 'date', 'warehouse'], as_index=False, dropna=False).quantity.sum()
    if len(deals) < 8 or deals.date.nunique() < 4:
        emit('warning', 'Мало сделок для устойчивого порога аномалий: требуется 8 сделок и 4 даты')
        return history
    median = float(deals.quantity.median())
    mad = float((deals.quantity - median).abs().median())
    threshold = max(5 * median, median + 6 * 1.4826 * mad)
    candidates = deals.loc[deals.quantity.gt(threshold)].copy()
    pending = []
    entries = {}
    for row in candidates.itertuples(index=False):
        peers = candidates.loc[candidates.quantity.between(row.quantity / 2, row.quantity * 2)]
        repeated = peers.date.nunique() >= 3 and (peers.date.max() - peers.date.min()).days >= 14
        entry = dict(supplier=supplier, sku=sku, date=row.date, document_id=row.document_id,
            quantity=float(row.quantity), excluded=False,
            reason=f'Порог {threshold:.4f}; склад {row.warehouse}; ' +
                ('повторяющийся крупный спрос сохранён' if repeated else 'разовая крупная сделка; проверка сверки'))
        anomalies.append(entry)
        entries[(row.document_id, row.date, row.warehouse)] = entry
        if not repeated:
            pending.append(entry)
    # A known customer may split a single-day purchase into several documents.
    # Missing IDs are never invented, and purchases on different days are not merged.
    if 'customer_id' in tx:
        keys = ['document_id', 'date', 'warehouse']
        sales = tx.loc[sale & tx.quantity.gt(0)]
        groups = sales.groupby(keys, dropna=False).customer_id
        if groups.nunique().gt(1).any():
            raise ValueError('transactions: у одного документа несколько customer_id')
        identities = groups.agg(lambda s: s.iloc[0] if s.notna().all() else None).reset_index()
        known = deals.merge(identities, on=keys).dropna(subset=['customer_id'])
        daily = known.groupby(['customer_id','date','warehouse'], as_index=False).quantity.sum()
        if len(daily) >= 8 and daily.date.nunique() >= 4:
            mid = float(daily.quantity.median())
            customer_threshold = max(mid*5, mid+6*1.4826*float((daily.quantity-mid).abs().median()))
            large = daily.loc[daily.quantity.gt(customer_threshold)]
            for row in large.itertuples(index=False):
                peers = large.loc[large.quantity.between(row.quantity/2, row.quantity*2)]
                if peers.date.nunique() >= 3 and (peers.date.max()-peers.date.min()).days >= 14:
                    continue
                documents = known.loc[known.customer_id.eq(row.customer_id) & known.date.eq(row.date) & known.warehouse.eq(row.warehouse)]
                for doc in documents.itertuples(index=False):
                    key = (doc.document_id, doc.date, doc.warehouse)
                    entry = entries.get(key)
                    if entry is None:
                        entry = dict(supplier=supplier,sku=sku,date=doc.date,document_id=doc.document_id,
                            quantity=float(doc.quantity),excluded=False,reason='')
                        entries[key] = entry
                        anomalies.append(entry)
                    if not any(item is entry for item in pending):
                        pending.append(entry)
                    entry['reason'] = f'Разовая дневная покупка одного подтверждённого клиента выше порога {customer_threshold:.4f}'
    if not pending:
        emit('info', f'Аномалии: порог {threshold:.4f}; разовых выбросов нет')
        return history
    # Correction between alternative views is safe only after exact-period reconciliation.
    sums = tx.groupby('month').quantity.sum().reindex(history['month'])
    if sums.isna().any() or not np.allclose(sums.to_numpy(), history.quantity.to_numpy(), rtol=0, atol=1e-6):
        for entry in pending:
            entry['reason'] = 'Не исключено: месячная история и сделки не сверены за выбранный период'
        raise ValueError('Нельзя исключить сделку: monthly_sales и transactions не совпадают за выбранные месяцы')
    corrected = history.copy()
    removed = pd.Series(0.0, index=corrected.index)
    for entry in pending:
        month = entry['date'].to_period('M').to_timestamp()
        removed.loc[corrected.month.eq(month)] += entry['quantity']
    if (corrected.quantity - removed).lt(-1e-6).any():
        for entry in pending:
            entry['reason'] = 'Не исключено: после удаления сделки месячный нетто-спрос отрицателен'
        raise ValueError('Исключение аномалий требует сверки возвратов: отрицательный остаточный спрос')
    corrected['quantity'] = (corrected.quantity - removed).clip(lower=0)
    for entry in pending:
        entry['excluded'] = True
        entry['reason'] += '; исключена из сверенной месячной истории'
    emit('warning', f'Из регулярного спроса исключено {removed.sum():.4f}; сделок: {len(pending)}')
    return corrected


def _demand_rates(history, intervals, as_of, unit, scope, factors, emit):
    """Estimate intensity per observed in-stock day; merge inclusive intervals."""
    days = history.month.dt.days_in_month.astype(float)
    exposure = days.copy()
    if not intervals.empty:
        _check_units(intervals, unit, 'stockouts')
        if not {'start_date', 'end_date', 'warehouse'}.issubset(intervals.columns):
            raise ValueError('stockouts: отсутствуют даты или склад')
        if 'confirmed' in intervals and not intervals.confirmed.eq(True).fillna(False).all():
            raise ValueError('stockouts: переданы неподтверждённые интервалы')
        lost_days = set()
        first = history.month.min()
        last = min(as_of - pd.Timedelta(days=1), (history.month + pd.offsets.MonthEnd(0)).max())
        for row in intervals.itertuples(index=False):
            start, end = _date(row.start_date, 'stockouts.start_date'), _date(row.end_date, 'stockouts.end_date')
            if end < start:
                raise ValueError('stockouts: конец раньше начала')
            start, end = max(start, first), min(end, last)
            if end < start:
                continue
            if _scope(row.warehouse) != scope:
                raise ValueError('stockouts: интервал должен описывать весь выбранный контур склада')
            lost_days.update(pd.date_range(start, end, freq='D'))
        for idx, month in history.month.items():
            count = sum(day in lost_days for day in pd.date_range(month, month + pd.offsets.MonthEnd(0)))
            exposure.loc[idx] -= count
        if ((exposure == 0) & history.quantity.gt(0)).any():
            raise ValueError('stockouts: продажи в месяце с нулевым числом доступных дней требуют сверки')
        emit('info', f'Stockout: учтено {int((days-exposure).sum())} уникальных дней в выбранных месяцах')
    seasonal = history.month.dt.month.map(factors)
    rates = history.quantity / exposure.replace(0, np.nan) / seasonal
    if rates.isna().any():
        if rates.notna().sum() == 0:
            raise ValueError('Нет дней доступности для оценки спроса при stockout')
        rates = rates.fillna(float(rates.median()))
        emit('warning', 'Полностью отсутствовавшие месяцы оценены по медиане интенсивности доступных месяцев')
    raw = float((history.quantity / days / seasonal).mean())
    emit('info', f'Интенсивность после stockout {rates.mean():.4f}; без компенсации {raw:.4f}')
    return rates, exposure


def _forecast_rates(history, rates, exposure, dates, enabled, emit):
    baseline = float(rates.mean())
    result = pd.Series(baseline, index=dates)
    if not enabled:
        emit('info', 'Тренд выключен настройкой enable_trend')
        return result
    x = history.month.dt.year.to_numpy() * 12 + history.month.dt.month.to_numpy()
    y = rates.to_numpy(dtype=float)
    if len(y) < 4 or not np.all(np.diff(x) == 1) or (exposure == 0).any() or baseline <= 0:
        emit('info', 'Тренд: коэффициент 1; недостаточно последовательных наблюдаемых месяцев')
        return result
    changes = np.diff(y)
    sustained = np.mean(changes > 0) >= 0.6 and np.mean(y[-2:]) > 1.1 * np.mean(y[:2])
    if not sustained:
        emit('info', 'Тренд: коэффициент 1; устойчивый рост не подтверждён')
        return result
    slopes = [(y[j]-y[i])/(x[j]-x[i]) for i in range(len(x)) for j in range(i+1,len(x))]
    slope = float(np.clip(np.median(slopes), 0, baseline * 0.2))
    intercept = float(np.median(y - slope * (x-x[-1])))
    future = np.array([d.year*12+d.month + (d.day-0.5)/d.days_in_month-0.5-x[-1] for d in dates])
    result[:] = np.clip(intercept + slope * future, baseline, baseline * 1.5)
    emit('info', f'Тренд: средний коэффициент {result.mean()/baseline:.4f}; наклон {slope:.4f}/месяц; предел 1.5')
    return result


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


def _scope(value):
    """Accept the legacy loader spelling without guessing an unknown scope."""
    if pd.isna(value):
        return None
    return '__all__' if value in ('__all__', '*all*') else value


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
    orders, forecasts, anomalies, diagnostics = [], [], [], []
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
            confirmed_scope = _scope(settings.get('confirmed_warehouse_scope', {}).get(supplier))
            if confirmed_scope is not None and (not isinstance(confirmed_scope, str) or not confirmed_scope.strip()):
                raise ValueError('confirmed_warehouse_scope: нужен непустой склад или __all__')
            if not quality.empty:
                quality = quality.loc[
                    (quality['supplier'].isna() | quality['supplier'].eq(supplier))
                    & (quality['sku'].isna() | quality['sku'].eq(sku))
                ]
            if not quality.empty:
                for _, finding in quality.iterrows():
                    severity = finding.get('severity', 'warning')
                    note(supplier, sku, severity, str(finding.get('issue', 'Ошибка качества данных')))

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
            # All enabled demand algorithms below use this same explicit month set.
            used_months = set(history.month.dt.strftime('%Y-%m'))
            if not quality.empty:
                blocking = []
                for finding in quality.loc[quality.severity.eq('error')].itertuples(index=False):
                    issue = str(finding.issue)
                    match = re.fullmatch(r'monthly_transaction_mismatch:((?:\d{4}-(?:0[1-9]|1[0-2]))(?:,\d{4}-(?:0[1-9]|1[0-2]))*)', issue)
                    if match and used_months and used_months.isdisjoint(match[1].split(',')):
                        note(supplier, sku, 'info', 'Ошибка сохранена, но неприменима к выбранному периоду '
                            + ','.join(sorted(used_months)) + ': ' + issue)
                    else:
                        blocking.append(issue)
                if blocking:
                    raise ValueError('Ошибки данных: ' + '; '.join(dict.fromkeys(blocking))[:2000])
                if quality['issue'].eq('warehouse_scope_unconfirmed').any() and confirmed_scope is None:
                    raise ValueError('Подтвердите единый контур продаж, остатков и транзита: confirmed_warehouse_scope')
            if history.empty or not np.isfinite(history['quantity']).all() or history['quantity'].lt(0).any():
                raise ValueError('Нет пригодной истории: пропуски или отрицательный месячный спрос')
            if len(history) < 3:
                note(supplier, sku, 'warning', 'Меньше трёх полных месяцев: короткая история')

            season = _rows(dataset, 'seasonality', supplier)
            factors = {month: 1.0 for month in range(1, 13)}
            if not season.empty:
                specific = season.loc[season['category'].eq(product.get('category'))]
                season = specific if not specific.empty else season.loc[season['category'].isin(['__all__', '*all*'])]
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
            stock = _rows(dataset, 'stock', supplier, sku)
            if stock.empty:
                raise ValueError('Нет текущего свободного остатка')
            stock['as_of'] = pd.to_datetime(stock['as_of'], errors='coerce')
            stock = stock.loc[stock['as_of'].le(as_of) & stock['is_current'].eq(True)]
            if stock.empty:
                raise ValueError('Остаток исторический или отсутствует: нужен актуальный снимок')
            stock['warehouse'] = stock['warehouse'].map(_scope)
            if confirmed_scope is not None:
                stock['warehouse'] = stock['warehouse'].fillna(confirmed_scope)
                if not stock['warehouse'].eq(confirmed_scope).all():
                    raise ValueError('Подтверждённый контур не совпадает с текущим остатком')
                note(supplier, sku, 'warning', f'Контур продаж, остатков и транзита подтверждён настройкой: {confirmed_scope}')
            if stock.empty or stock['warehouse'].isna().any() or stock['warehouse'].nunique() != 1:
                raise ValueError('Не определён единый контур склада для остатка')
            latest = stock.loc[stock['as_of'].eq(stock['as_of'].max())]
            if len(latest) != 1:
                raise ValueError('Неоднозначный снимок остатка')
            snapshot = latest.iloc[0]
            record['stock_as_of'] = snapshot['as_of']
            if snapshot['as_of'].normalize() != as_of:
                raise ValueError('Дата актуального снимка не совпадает с as_of: требуется снимок на дату расчёта')
            if pd.isna(snapshot['is_current']) or snapshot['is_current'] != True:
                raise ValueError('Остаток исторический: нужен актуальный снимок')
            free = _number(snapshot['free_stock'], 'free_stock')
            record['free_stock'] = free

            emit = lambda severity, issue: note(supplier, sku, severity, issue)
            emit('info', 'Источник спроса: monthly_sales; transactions используется только для сверенной коррекции аномалий')
            _check_units(history, product['unit'], 'monthly_sales')
            if settings.get('exclude_anomalies', False):
                history = _regular_history(history, _rows(dataset, 'transactions', supplier, sku),
                    supplier, sku, product['unit'], snapshot['warehouse'], anomalies, emit)
            else:
                emit('info', 'Исключение аномалий выключено')
            intervals = pd.DataFrame()
            if settings.get('restore_stockouts', False):
                intervals = _rows(dataset, 'stockouts', supplier, sku)
                if intervals.empty:
                    emit('warning', 'Нет подтверждённых stockout: расчёт без компенсации')
            rates, exposure = _demand_rates(history, intervals, as_of, product['unit'], snapshot['warehouse'], factors, emit)
            predicted = _forecast_rates(history, rates, exposure, dates, settings.get('enable_trend', True), emit)
            predicted *= [factors[day.month] for day in dates]
            forecast_qty = float(predicted.sum())
            safety_days = _number(settings.get('safety_days_by_category', {}).get(
                product.get('category'), settings.get('default_safety_days')), 'safety_days')
            safety = float(predicted.mean() * safety_days)
            record.update(forecast_qty=forecast_qty, safety_stock=safety)
            forecasts.extend(dict(supplier=supplier, sku=sku, date=day, predicted_qty=float(qty))
                             for day, qty in predicted.items())

            incoming = _rows(dataset, 'transit', supplier, sku)
            receipts = pd.Series(0.0, index=dates)
            today_receipts = 0.0
            if not incoming.empty:
                incoming['quantity'] = pd.to_numeric(incoming['quantity'], errors='coerce')
                if not np.isfinite(incoming['quantity']).all() or incoming['quantity'].lt(0).any():
                    raise ValueError('Некорректное количество транзита')
                incoming = incoming.loc[incoming['quantity'].gt(0)]
                incoming['warehouse'] = incoming['warehouse'].map(_scope)
                if confirmed_scope is not None:
                    incoming['warehouse'] = incoming['warehouse'].fillna(confirmed_scope)
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
            constraints = {}
            for field in ['min_order_qty', 'pack_multiple']:
                value = product.get(field)
                supplied = settings.get('order_constraints_by_supplier', {}).get(supplier, {})
                if raw > 0 and pd.isna(value) and field in supplied:
                    value = supplied[field]
                    note(supplier, sku, 'warning', f'{field} отсутствует в файле; применена явная настройка {value}')
                constraints[field] = value
            recommended = round_order(raw, **constraints)
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
            note(supplier, sku, 'warning', 'Прогноз по ограниченной истории: проверьте допущения и условия поставки')
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
        'anomalies': pd.DataFrame(anomalies, columns=ANOMALY_COLUMNS),
        'diagnostics': pd.DataFrame(diagnostics, columns=DIAGNOSTIC_COLUMNS),
    }
