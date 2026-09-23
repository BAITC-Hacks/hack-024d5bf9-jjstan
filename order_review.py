"""Review state and validation belong to the frontend, not the demand engine."""
from hashlib import sha256
import json
import math
import numpy as np
import pandas as pd


def row_key(supplier, sku):
    return json.dumps([str(supplier), str(sku)], ensure_ascii=False)


def make_review(orders, products, settings=None):
    if orders.duplicated(['supplier', 'sku']).any() or products.duplicated(['supplier', 'sku']).any():
        raise ValueError('Повторные коды товаров: сначала устраните неоднозначность данных.')
    draft = orders.copy(deep=True)
    draft['review_key'] = [row_key(s, k) for s, k in zip(draft.supplier, draft.sku)]
    draft = draft.set_index('review_key', drop=True)
    product_lookup = products.set_index(['supplier', 'sku'])
    for field in ['min_order_qty', 'pack_multiple', 'supplier_article']:
        draft[field] = [product_lookup.loc[(s, k)].get(field, np.nan) for s, k in zip(draft.supplier, draft.sku)]
    # The engine accepts explicit supplier defaults only for missing conditions.
    # Validate manual quantities against those same approved input settings.
    supplied = (settings or {}).get('order_constraints_by_supplier', {})
    for field in ['min_order_qty', 'pack_multiple']:
        draft[field] = [supplied.get(supplier, {}).get(field, value) if pd.isna(value) else value
                        for supplier, value in zip(draft.supplier, draft[field])]
    draft['approved_qty'] = pd.to_numeric(draft.recommended_qty, errors='coerce')
    draft['selected'] = False
    return draft


def merge_visible_edits(draft, edited):
    """Index-based join prevents sorting/filtering from moving changes to a SKU."""
    if not edited.index.is_unique or not edited.index.isin(draft.index).all():
        raise ValueError('Не удалось сопоставить изменения с кодами товаров.')
    result = draft.copy(deep=True)
    result.loc[edited.index, 'approved_qty'] = pd.to_numeric(edited['approved_qty'], errors='coerce')
    result.loc[edited.index, 'selected'] = edited['selected'].fillna(False).astype(bool)
    return result


def review_signature(draft, calculation_id):
    values = draft[['supplier', 'sku', 'selected', 'approved_qty']].sort_index().to_json(orient='split')
    return sha256((calculation_id + values).encode('utf-8')).hexdigest()


def validate_selection(draft):
    selected = draft.loc[draft.selected.fillna(False)].copy()
    errors = []
    if selected.empty:
        return selected, ['Выберите хотя бы одну позицию в таблице.']
    for _, row in selected.iterrows():
        label = f'{row.supplier} / {row.sku}'
        if row.data_quality == 'insufficient' or not _finite(row.recommended_qty):
            errors.append(f'{label}: недостаточно данных для утверждения.')
            continue
        if not _finite(row.approved_qty) or float(row.approved_qty) < 0:
            errors.append(f'{label}: укажите конечное неотрицательное количество.')
            continue
        qty = float(row.approved_qty)
        if qty == 0:
            continue  # A deliberate refusal is not replaced with the recommendation.
        if pd.isna(row.unit) or not str(row.unit).strip():
            errors.append(f'{label}: неизвестна единица учёта.')
        if not _finite(row.min_order_qty) or not _finite(row.pack_multiple) or row.pack_multiple <= 0:
            errors.append(f'{label}: не подтверждены минимум и кратность.')
            continue
        if qty + 1e-9 < float(row.min_order_qty):
            errors.append(f'{label}: минимум заказа {row.min_order_qty:g}.')
        packs = qty / float(row.pack_multiple)
        if not math.isclose(packs, round(packs), rel_tol=0, abs_tol=1e-7):
            errors.append(f'{label}: количество должно быть кратно {row.pack_multiple:g}.')
    if not errors and selected.approved_qty.le(0).all():
        errors.append('Все выбранные количества равны нулю. Заказ для отправки пуст.')
    return selected, errors


def _finite(value):
    try:
        return not isinstance(value, (bool, np.bool_)) and math.isfinite(float(value))
    except (ValueError, TypeError):
        return False
