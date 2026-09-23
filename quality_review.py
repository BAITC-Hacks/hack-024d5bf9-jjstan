"""Read-only, actionable view of loader findings and engine decisions."""
from collections import defaultdict
import re

import pandas as pd

from export import safe_cell

BLOCKING = 'Блокирует расчёт'
REVIEW = 'Требует проверки'
OUTSIDE = 'Вне выбранного периода'
INFO = 'Информация'
MONTHS = r'\d{4}-(?:0[1-9]|1[0-2])(?:,\d{4}-(?:0[1-9]|1[0-2]))*'
EXCEPTION = re.compile(r'Ошибка сохранена, но неприменима к выбранному периоду (' + MONTHS
                       + r'): (monthly_transaction_mismatch:' + MONTHS + r')')
COLUMNS = ['supplier', 'sku', 'name', 'linked', 'severity', 'category', 'summary',
           'applicability', 'months', 'source', 'stage', 'action', 'evidence', 'issue', 'order_status']
LABELS = dict(zip(COLUMNS, ['Поставщик', 'Код 1С', 'Наименование', 'Есть в справочнике',
    'Исходный уровень', 'Тип проблемы', 'Проблема', 'Применимость', 'Период / месяцы',
    'Источник', 'Этап', 'Что исправить или уточнить', 'Основание применимости',
    'Исходное сообщение', 'Итог расчёта товара']))


def text(value):
    return None if value is None or pd.isna(value) else str(value)


def describe(issue, severity):
    """Advice identifies required evidence, never supplies missing business values."""
    value = issue.lower()
    if value.startswith('monthly_transaction_mismatch:'):
        return ('Расхождения продаж', 'Месячные продажи не совпадают с суммой операций',
            'Сверить указанные месяцы, возвраты, типы документов и охват отчётов. Исправить подтверждённую ошибку источника; не складывать два представления продаж.')
    rules = [
        (('snapshot_sku_missing_monthly_history',), 'История продаж', 'Товар есть в остатках, но не связан с месячной историей',
         'Проверить код 1С и поставщика в месячном отчёте. Предоставить историю этого товара; отсутствие строки не означает нулевые продажи.'),
        (('stockout', 'периодов отсутствия'), 'Периоды отсутствия товара', 'Нет или требуют проверки интервалы отсутствия товара',
         'Указать подтверждённые даты начала и окончания отсутствия товара и склад. Месяц без продаж сам по себе не доказывает stockout.'),
        (('customer', 'клиент'), 'Данные клиентов', 'Ограничена проверка покупок одного клиента',
         'Нужен обезличенный ID клиента в операциях. Номер накладной не заменяет ID клиента.'),
        (('warehouse', 'контур', 'склад сделок'), 'Складской охват', 'Требует проверки совместимость складов',
         'Сопоставить охват продаж, свободного остатка и транзита. Указать подтверждённый общий контур только при наличии основания.'),
        (('unit', 'coil', 'единиц', 'бухт'), 'Единицы учёта', 'Единицы отсутствуют или не согласованы',
         'Сверить единицы продаж, остатка, транзита и ограничений заказа. Перевод бухт в метры допустим только с известной длиной бухты.'),
        (('current_stock', 'free_stock', 'missing_table_stock', 'остат', 'снимк', 'snapshot'), 'Актуальный остаток', 'Требуется проверка свободного остатка и даты',
         'Нужен свободный остаток на дату расчёта в том же складском контуре. Начальный остаток месяца не является текущим; резерв из свободного остатка повторно не вычитают.'),
        (('order_constraint', 'min_order_qty', 'pack_multiple', 'кратност', 'минимум'), 'Минимум и кратность', 'Не подтверждены ограничения заказа',
         'Указать отдельно минимальную партию и кратность с их единицами. Пропуск не означает ноль или единицу.'),
        (('transit', 'транзит', 'поступлен'), 'Товары в пути', 'Требуют проверки ожидаемые поставки',
         'Уточнить количество, единицу, склад и ожидаемую дату. Проверить фактическое получение просроченной поставки и отсутствие двойного учёта в остатке.'),
        (('lead_time', 'срок'), 'Сроки поставки', 'Срок поставки не подтверждён',
         'Указать проверенный срок поставки для поставщика. Значение по умолчанию не подтверждает условия поставки.'),
        (('season', 'сезон'), 'Сезонность', 'Требует проверки сезонный профиль',
         'Проверить месяцы, категорию и применимость коэффициентов к единицам спроса; исключить влияние неполного месяца.'),
        (('histor', 'истори', 'monthly', 'месяц', 'reconciliation'), 'История продаж', 'История продаж отсутствует или требует сверки',
         'Проверить завершённые месяцы, пропуски и охват источников. Не заменять неизвестное количество нулём и не сокращать историю ради снятия ошибки.'),
        (('bom',), 'Материальная ведомость', 'BOM не предоставлена',
         'Указать это ограничение: расчёт потребностей по составу изделий в текущей версии не реализован.'),
        (('document', 'transaction', 'операци', 'сделк', 'возврат'), 'Операции продаж', 'Требуют проверки документы продаж',
         'Проверить дату, вид документа, знак и количество операции. Продажи положительные, возвраты отрицательные; нельзя применять abs ко всем операциям.'),
        (('sku', 'header', 'workbook', 'input_read', 'file', 'duplicate', 'conflicting'), 'Файлы и связи товаров', 'Требует проверки файл или связь товара',
         'Проверить файл, заголовки, поставщика и строковый код 1С во всех источниках. Сохранить ведущие нули; не связывать товары только по названию.'),
    ]
    if severity == 'info':
        return 'Информация расчёта', issue, 'Ознакомиться с пояснением; исправление из этой записи не следует.'
    for terms, category, summary, action in rules:
        if any(term in value for term in terms):
            return category, summary, action
    return ('Прочие замечания', issue,
            'Проверить исходное сообщение и источник. Устранить подтверждённую причину и повторить расчёт.')


def build_issue_report(dataset, result):
    """Merge repeated findings without changing severity or recalculating eligibility.

    Supplier-wide findings are attached to their products. A period exception is
    recognized only through the engine's exact supplier/SKU/issue evidence.
    Errors not reached by the engine remain pending, not automatically blocking.
    """
    products = dataset.get('products', pd.DataFrame()).to_dict('records')
    orders = result.get('orders', pd.DataFrame()).to_dict('records')
    quality = dataset.get('quality_report', pd.DataFrame()).to_dict('records')
    diagnostics = result.get('diagnostics', pd.DataFrame()).to_dict('records')
    names = {(text(p.get('supplier')), text(p.get('sku'))): text(p.get('name')) for p in products}
    order_map = {(text(o.get('supplier')), text(o.get('sku'))): o for o in orders}
    keys = list(dict.fromkeys([*names, *order_map]))
    by_supplier, by_sku = defaultdict(list), defaultdict(list)
    for key in keys:
        by_supplier[key[0]].append(key)
        by_sku[key[1]].append(key)
    exceptions = {}
    for item in diagnostics:
        match = EXCEPTION.fullmatch(text(item.get('issue')) or '')
        if match and item.get('severity') == 'info':
            exceptions[(text(item.get('supplier')), text(item.get('sku')), match[2])] = item['issue']

    records = {}
    def add(item, stage):
        supplier, sku = text(item.get('supplier')), text(item.get('sku'))
        issue = text(item.get('issue')) or 'Сообщение отсутствует'
        if EXCEPTION.fullmatch(issue) and stage == 'Расчёт':
            return  # Evidence is retained on its original finding, never discarded.
        if issue.startswith('Ошибки данных: ') and stage == 'Расчёт':
            return  # The exact individual findings remain visible below.
        severity = text(item.get('severity')) or 'warning'
        if supplier is not None and sku is not None:
            targets = [(supplier, sku)]
        elif supplier is not None:
            targets = by_supplier[supplier] or [(supplier, None)]
        elif sku is not None:
            targets = by_sku[sku] or [(None, sku)]
        else:
            targets = keys or [(None, None)]
        for key in targets:
            identity = (*key, severity, issue)
            record = records.setdefault(identity, dict(supplier=key[0], sku=key[1], name=names.get(key),
                linked=key in names and key[0] is not None and key[1] is not None,
                severity=severity, issue=issue, sources=set(), stages=set(), periods=set()))
            record['stages'].add(stage)
            source = text(item.get('source'))
            if source:
                record['sources'].add(source)
            elif stage == 'Расчёт':
                record['sources'].add('engine.py')
            months = text(item.get('affected_months'))
            if not months and issue.startswith('monthly_transaction_mismatch:'):
                months = issue.split(':', 1)[1]
            if months:
                record['periods'].add(months)
            elif pd.notna(item.get('period_start')) and pd.notna(item.get('period_end')):
                record['periods'].add(f"{pd.Timestamp(item['period_start']).date()} — {pd.Timestamp(item['period_end']).date()}")

    for item in quality:
        add(item, 'Источник')
    for item in diagnostics:
        add(item, 'Расчёт')
    for key, order in order_map.items():
        reason = text(order.get('reason')) or ''
        if order.get('data_quality') == 'insufficient' and reason:
            if reason.startswith('Ошибки данных: '):
                for issue in reason[len('Ошибки данных: '):].split('; '):
                    if (*key, 'error', issue) not in records:
                        add(dict(supplier=key[0], sku=key[1], severity='error', issue=issue), 'Расчёт')
            else:
                add(dict(supplier=key[0], sku=key[1], severity='error', issue=reason), 'Расчёт')

    rows = []
    for record in records.values():
        key, issue = (record['supplier'], record['sku']), record['issue']
        order = order_map.get(key, {})
        reason = text(order.get('reason')) or ''
        blockers = reason[len('Ошибки данных: '):].split('; ') if reason.startswith('Ошибки данных: ') else [reason]
        exception = exceptions.get((*key, issue)) if record['severity'] == 'error' else None
        if exception:
            applicability, evidence = OUTSIDE, exception
        elif order.get('data_quality') == 'insufficient' and issue in blockers:
            applicability, evidence = BLOCKING, reason
        elif record['severity'] == 'info':
            applicability, evidence = INFO, 'Информационная запись; сама по себе не определяет допуск заказа.'
        else:
            applicability, evidence = REVIEW, 'Есть замечание источника или расчёта; отдельная блокирующая роль этой записи не подтверждена.'
        category, summary, action = describe(issue, record['severity'])
        if exception:
            action = 'Сохранить расхождение для сверки источников. Расчёт явно исключил его только для указанного периода; данные не признаны исправленными.'
        rows.append({k: v for k, v in record.items() if k not in {'sources', 'stages', 'periods'}} | dict(
            category=category, summary=summary, action=action, applicability=applicability,
            evidence=evidence, months='; '.join(sorted(record['periods'])) or None,
            source='; '.join(sorted(record['sources'])) or None, stage='; '.join(sorted(record['stages'])),
            order_status=order.get('data_quality', 'Нет результата расчёта')))
    report = pd.DataFrame(rows, columns=COLUMNS)
    if not report.empty:
        rank = report.applicability.map({BLOCKING: 0, REVIEW: 1, OUTSIDE: 2, INFO: 3})
        report = report.assign(_rank=rank).sort_values(['_rank', 'supplier', 'sku', 'category'], kind='stable').drop(columns='_rank').reset_index(drop=True)
    return report


def summarize_issues(report):
    columns = ['Тип проблемы', 'Применимость', 'Товаров', 'Замечаний']
    if report.empty:
        return pd.DataFrame(columns=columns)
    grouping = ['category', 'applicability']
    counts = report.groupby(grouping, sort=False).size().rename('Замечаний')
    products = report.loc[report.linked].drop_duplicates(grouping + ['supplier', 'sku']).groupby(grouping).size().rename('Товаров')
    result = counts.to_frame().join(products).reset_index().rename(columns={'category': 'Тип проблемы', 'applicability': 'Применимость'})
    result['Товаров'] = result['Товаров'].fillna(0).astype(int)
    return result[columns]


def reconciliation_details(monthly, finding):
    columns = ['month', 'quantity', 'reconciled_transaction_quantity', 'reconciliation_difference', 'reconciliation_status', 'is_complete']
    if monthly.empty or not finding['issue'].startswith('monthly_transaction_mismatch:'):
        return pd.DataFrame(columns=columns)
    months = finding['issue'].split(':', 1)[1].split(',')
    selected = monthly.loc[monthly.supplier.eq(finding['supplier']) & monthly.sku.eq(finding['sku'])].copy()
    selected = selected.loc[pd.to_datetime(selected.month, errors='coerce').dt.strftime('%Y-%m').isin(months)]
    return selected.reindex(columns=columns).sort_values('month')


def build_issue_csv(report, *, calculation_date, source_mode):
    output = report.reindex(columns=COLUMNS).rename(columns=LABELS).copy()
    output['Дата расчёта'] = str(calculation_date)
    output['Режим данных'] = source_mode
    output['Назначение'] = 'Список замечаний для проверки; не заказ поставщику'
    for column in output:
        output[column] = output[column].map(safe_cell)
    return output.to_csv(index=False, sep=';', lineterminator='\n').encode('utf-8-sig')
