from io import BytesIO
import numpy as np
import pandas as pd
import pytest
from demo_data import make_demo_dataset, default_settings
from ui_integration import run_calculation
from order_review import make_review
from export import build_approved_csv


def snapshot():
    data = make_demo_dataset()
    result, _ = run_calculation(data, default_settings())
    frame = make_review(result['orders'], data['products'])
    frame.iloc[0, frame.columns.get_loc('selected')] = True
    return frame


def export(frame):
    return build_approved_csv(frame, calculation_date='2026-09-22', approved_at='2026-09-23T10:00:00', source_mode='Демонстрация')


def test_csv_uses_approved_quantity_and_preserves_source():
    frame = snapshot()
    frame.iloc[0, frame.columns.get_loc('approved_qty')] = 84
    payload = export(frame)
    assert payload.startswith(b'\xef\xbb\xbf')
    result = pd.read_csv(BytesIO(payload), sep=';')
    assert result['Утверждено'].iloc[0] == 84
    assert result['Рекомендовано'].iloc[0] == 72
    assert len(result) == 1
    assert result['Источник'].iloc[0] == 'Демонстрация'
    assert frame.iloc[0].recommended_qty == 72


def test_unselected_and_zero_rows_not_exported():
    frame = snapshot()
    frame.iloc[1, frame.columns.get_loc('selected')] = True
    frame.iloc[1, frame.columns.get_loc('approved_qty')] = 0
    result = pd.read_csv(BytesIO(export(frame)), sep=';')
    assert result['Код 1С'].tolist() == ['DEMO-001']


@pytest.mark.parametrize('field', ['name', 'sku', 'supplier_article', 'reason'])
def test_csv_formula_injection_is_neutralized(field):
    frame = snapshot()
    frame.iloc[0, frame.columns.get_loc(field)] = '=1+1'
    text = export(frame).decode('utf-8-sig')
    assert "'=1+1" in text


def test_export_rejects_insufficient_even_after_manual_override():
    frame = snapshot()
    frame.iloc[0, frame.columns.get_loc('recommended_qty')] = np.nan
    frame.iloc[0, frame.columns.get_loc('approved_qty')] = 12
    with pytest.raises(ValueError, match='недостаточно'):
        export(frame)


def test_export_supports_fractional_unit_with_confirmed_multiple():
    frame = snapshot()
    frame['selected'] = False
    index = frame.index[frame.sku.eq('DEMO-006')][0]
    frame.loc[index, ['selected', 'approved_qty']] = [True, 5.5]
    result = pd.read_csv(BytesIO(export(frame)), sep=';')
    assert result['Утверждено'].iloc[0] == 5.5
