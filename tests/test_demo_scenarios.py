from io import BytesIO
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from demo_data import DEMO_DATE, DEMO_SCENARIOS


def open_app():
    return AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'), default_timeout=30).run()


@pytest.mark.parametrize('scenario,flag,before,after', [
    ('anomaly', 'exclude_anomalies', 5232, 72),
    ('stockout', 'restore_stockouts', 0, 72),
    ('trend', 'enable_trend', 192, 336),
])
def test_demo_algorithm_comparison_and_selected_order(scenario, flag, before, after):
    at = open_app()
    at.selectbox(key='demo_scenario').set_value(scenario).run()
    assert not at.exception
    assert at.date_input(key='as_of').value == pd.Timestamp('2026-04-01').date()
    assert at.button(key='approve').disabled
    at.button(key='calculate').click().run()
    assert not at.exception
    assert at.session_state['draft'].iloc[0].recommended_qty == before
    comparison = at.session_state['demo_comparison']
    assert comparison['before'] == before
    assert comparison['after'] == after
    assert comparison['enabled'] is False
    spec = DEMO_SCENARIOS[scenario]
    metrics = {item.label: item.value for item in at.metric}
    assert metrics[spec['before']] == f'{before:,} шт'.replace(',', ' ')
    assert metrics[spec['after']] == f'{after:,} шт'.replace(',', ' ')
    assert at.checkbox(key='ai_enabled').value is False

    at.checkbox(key=flag).check().run()
    assert not at.exception
    assert at.button(key='approve').disabled
    assert spec['before'] not in {item.label for item in at.metric}
    at.button(key='calculate').click().run()
    assert not at.exception
    assert at.session_state['draft'].iloc[0].recommended_qty == after
    assert at.session_state['demo_comparison']['enabled'] is True
    if scenario == 'anomaly':
        assert at.session_state['result']['anomalies'].excluded.sum() == 1
    at.button(key='select_visible').click().run()
    at.checkbox(key='confirm_review').check().run()
    at.button(key='approve').click().run()
    assert not at.exception
    exported = pd.read_csv(BytesIO(at.session_state['approval']['payload']), sep=';')
    assert exported['Утверждено'].tolist() == [after]
    assert exported['Источник'].tolist() == ['Демонстрация']


def test_scenario_change_invalidates_approval_and_cache_even_with_same_parameters():
    at = open_app()
    at.selectbox(key='demo_scenario').set_value('anomaly').run()
    at.button(key='calculate').click().run()
    original_id = at.session_state['calculation_id']
    original_settings = at.session_state['calculated_settings']
    at.button(key='select_visible').click().run()
    at.checkbox(key='confirm_review').check().run()
    at.button(key='approve').click().run()
    assert not at.exception
    assert at.session_state['approval']['count'] == 1
    at.session_state['ai_cache'] = {'old': {'status': 'ok'}}
    at.text_input(key='search').set_value('SYNTH').run()

    at.selectbox(key='demo_scenario').set_value('stockout').run()
    assert not at.exception
    assert 'approval' not in at.session_state
    assert at.session_state['ai_cache'] == {}
    assert at.text_input(key='search').value == ''
    assert at.button(key='approve').disabled
    at.button(key='calculate').click().run()
    assert not at.exception
    assert at.session_state['calculation_id'] != original_id
    assert at.session_state['calculated_settings'] == original_settings
    assert at.session_state['dataset']['monthly_sales'].iloc[0].quantity == 160
    assert at.session_state['dataset']['transactions'].empty
    assert not at.session_state['dataset']['stockouts'].empty
    assert not at.session_state['draft'].selected.any()


def test_comparison_uses_edited_parameters_and_overview_restores_original_demo():
    at = open_app()
    at.selectbox(key='demo_scenario').set_value('trend').run()
    at.button(key='calculate').click().run()
    at.number_input(key='lead_iek').set_value(14).run()
    at.button(key='calculate').click().run()
    assert not at.exception
    comparison = at.session_state['demo_comparison']
    assert comparison['before'] != 192
    assert comparison['after'] != 336
    assert at.session_state['draft'].iloc[0].recommended_qty == comparison['before']

    at.selectbox(key='demo_scenario').set_value('overview').run()
    at.button(key='calculate').click().run()
    assert not at.exception
    assert at.date_input(key='as_of').value == DEMO_DATE.date()
    assert at.number_input(key='lead_iek').value == 7
    assert len(at.session_state['dataset']['products']) == 8
    assert at.session_state['draft'].iloc[0].recommended_qty == 72
    assert at.session_state['demo_comparison'] is None


def test_real_mode_has_no_demo_comparison_and_cannot_export_previous_scenario():
    at = open_app()
    at.selectbox(key='demo_scenario').set_value('anomaly').run()
    at.button(key='calculate').click().run()
    at.radio(key='source_mode').set_value('Мои файлы').run()
    assert not at.exception
    assert at.button(key='approve').disabled
    assert DEMO_SCENARIOS['anomaly']['before'] not in {item.label for item in at.metric}
    assert not any(widget.key == 'demo_scenario' for widget in at.selectbox)
    at.button(key='calculate').click().run()
    assert any('Добавьте' in item.value for item in at.error)
    at.radio(key='source_mode').set_value('Демонстрация').run()
    at.button(key='calculate').click().run()
    assert not at.exception
    assert at.date_input(key='as_of').value == DEMO_DATE.date()
    assert len(at.session_state['draft']) == 8
