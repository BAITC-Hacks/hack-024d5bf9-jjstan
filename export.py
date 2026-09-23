"""CSV export of a validated approval snapshot; no sending or recalculation."""
from datetime import datetime
import pandas as pd
from order_review import validate_selection

EXPORT_COLUMNS = {
    'supplier': 'Поставщик', 'sku': 'Код 1С', 'supplier_article': 'Артикул поставщика',
    'name': 'Наименование', 'unit': 'Единица', 'approved_qty': 'Утверждено',
    'recommended_qty': 'Рекомендовано', 'reason': 'Обоснование',
}


def safe_cell(value):
    if pd.isna(value):
        return ''
    text = str(value)
    if text.lstrip().startswith(('=', '+', '-', '@')) or text.startswith(('\t', '\r', '\n')):
        return "'" + text
    return text


def build_approved_csv(snapshot, *, calculation_date, approved_at, source_mode):
    selected, errors = validate_selection(snapshot)
    if errors:
        raise ValueError('\n'.join(errors))
    selected = selected.loc[selected.approved_qty.gt(0)].copy()
    selected = selected.sort_values(['supplier', 'sku'])
    output = selected[list(EXPORT_COLUMNS)].rename(columns=EXPORT_COLUMNS)
    for source, name in EXPORT_COLUMNS.items():
        if source not in {'approved_qty', 'recommended_qty'}:
            output[name] = output[name].map(safe_cell)
    output['Дата расчёта'] = str(calculation_date)
    output['Утверждено в'] = str(approved_at)
    output['Источник'] = source_mode
    return output.to_csv(index=False, sep=';', lineterminator='\n').encode('utf-8-sig')
