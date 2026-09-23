"""Ramazan's Streamlit frontend. Demand and Excel logic stay in teammate modules."""
from datetime import datetime
from html import escape
from pathlib import Path
import hashlib
import traceback

import numpy as np
import pandas as pd
import streamlit as st

from demo_data import DEMO_DATE, make_demo_dataset
from export import build_approved_csv
from order_review import make_review, merge_visible_edits, review_signature, validate_selection
from ui_integration import input_signature, load_uploads, run_calculation

st.set_page_config(page_title='StockPilot · Заказы поставщикам', page_icon='↗', layout='wide')

LABELS = {
    'supplier': 'Поставщик', 'sku': 'Код 1С', 'name': 'Наименование', 'category': 'Категория',
    'unit': 'Ед.', 'free_stock': 'Свободно', 'incoming_in_horizon': 'В пути',
    'recommended_qty': 'Рекомендовано', 'approved_qty': 'К заказу', 'reason': 'Обоснование',
    'data_quality': 'Данные', 'urgency': 'Срочность', 'selected': 'Выбрать',
    'stock_as_of': 'Дата остатка', 'forecast_qty': 'Прогноз', 'safety_stock': 'Страховой запас',
    'supplier_article': 'Артикул', 'issue': 'Описание', 'source': 'Источник', 'severity': 'Уровень',
}
QUALITY = {'ok': 'Проверены', 'warning': 'Есть предупреждения', 'insufficient': 'Недостаточно данных'}
URGENCY = {'urgent': 'Срочно', 'normal': 'Планово', 'none': 'Не требуется', 'unknown': 'Нет оценки'}
ISSUES = {
    'current_stock_missing': 'Нет актуального свободного остатка по поставщику.',
    'sku_current_free_stock_missing': 'Нет актуального свободного остатка товара.',
    'warehouse_scope_unconfirmed': 'Не подтверждён общий охват складов в отчётах.',
    'missing_customer_id': 'В источниках отсутствует анонимный ID клиента.',
    'missing_daily_stockouts': 'Нет точных периодов отсутствия товара.',
    'missing_supplier_lead_times': 'Сроки поставки задаются менеджером как допущение.',
    'missing_bom': 'Спецификация состава изделий не предоставлена.',
    'coil_meter_conversion_unconfirmed': 'Нужен подтверждённый перевод бухт в метры.',
    'order_constraint_units_not_explicit_in_moq': 'Единицы ограничений MOQ требуют подтверждения.',
    'transit_blank_cells_do_not_confirm_zero_incoming': 'Пустые ячейки транзита не доказывают отсутствие поставок.',
    'historical_stock_not_confirmed_free_stock': 'Исторический остаток не подтверждает свободное количество.',
    'supplier_seasonality_includes_partial_2026_and_unconfirmed_measure': 'Применимость сезонного профиля к количествам требует проверки.',
    'transit_header_date_requires_confirmation': 'Дата в заголовке транзита требует подтверждения.',
    'blank_monthly_cells_preserved_as_nan': 'Пустые продажи сохранены как неизвестные значения.',
    'unit_missing': 'Не найдена единица учёта.',
    'unknown_warehouse': 'Склад не указан.',
}


def readable_issue(value):
    value = str(value)
    if value.startswith('monthly_transaction_mismatch:'):
        return 'Месячный отчёт и динамика расходятся: ' + value.split(':', 1)[1]
    return ISSUES.get(value, value)


def fmt(value):
    return 'Нет данных' if pd.isna(value) else f'{float(value):,.2f}'.replace(',', ' ').rstrip('0').rstrip('.')


def style():
    st.html('''<style>
    .stMainBlockContainer {padding:2rem 2rem 3rem; max-width:1550px;}
    [data-testid="stSidebar"] {border-right:1px solid #e2e8ef;}
    [data-testid="stSidebar"] .stMarkdown h3 {font-size:1.2rem;}
    h1 {font-size:2.2rem!important; letter-spacing:-.055rem; font-weight:700!important;}
    h2,h3 {letter-spacing:-.025rem;}
    .brand {display:flex;gap:10px;align-items:center;font-size:23px;font-weight:750;color:#153e35;margin-bottom:6px;}
    .brand-icon {background:#147d64;color:white;padding:2px 9px;border-radius:9px;font-size:24px;}
    .eyebrow {font-size:11px;letter-spacing:.15em;font-weight:700;color:#718493;text-transform:uppercase;}
    .context-pill {display:inline-block;background:#e5f2ec;border:1px solid #c5e1d5;border-radius:20px;padding:5px 12px;font-size:12px;color:#16664d;}
    .intro {font-size:15px;color:#697c8a;margin-top:-12px;margin-bottom:24px;}
    [data-testid="stMetric"] {background:white;border:1px solid #e1e8ef;border-radius:12px;padding:18px 22px;}
    [data-testid="stMetricLabel"] {color:#637787;font-size:13px;}
    [data-testid="stMetricValue"] {font-size:2rem;color:#173b32;}
    [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] [data-testid="stMetric"]) {flex-wrap:wrap;}
    [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] [data-testid="stMetric"]) > [data-testid="stColumn"] {min-width:145px;}
    @media(max-width:1100px) {
      [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] [data-testid="stMetric"]) > [data-testid="stColumn"] {flex:1 1 calc(50% - 1rem)!important;min-width:0!important;}
    }
    button[aria-label="Download as CSV"], button[title="Download as CSV"] {display:none!important;}
    .stTabs [data-baseweb="tab-list"] {gap:28px;border-bottom:1px solid #dfe7ec;margin-top:12px;margin-bottom:16px;}
    .stTabs [data-baseweb="tab"] {padding:12px 0;font-size:14px;}
    .stButton button,.stDownloadButton button {border-radius:8px;min-height:40px;}
    .stButton button p,.stDownloadButton button p {white-space:normal!important;}
    .stDataFrame {border:1px solid #e1e8ef;border-radius:10px;overflow:hidden;}
    .formula {background:#edf5f1;border-left:3px solid #168167;padding:20px;border-radius:8px;color:#244d42;}
    .foot {border-top:1px solid #e1e8ef;margin-top:28px;padding-top:14px;color:#81919b;font-size:12px;}
    </style>''')


def clear_approval():
    st.session_state.pop('approval', None)
    if 'confirm_review' in st.session_state:
        st.session_state.confirm_review = False


def calculate(mode, uploads, settings, signature):
    clear_approval()
    source_id = input_signature(mode, uploads, {})
    if st.session_state.get('loaded_source_id') != source_id:
        data = make_demo_dataset() if mode == 'Демонстрация' else load_uploads(uploads)
    else:
        data = st.session_state.dataset
    result, prepared = run_calculation(data, settings)
    draft = make_review(result['orders'], data['products'], settings)
    st.session_state.update(result=result, prepared=prepared, draft=draft, dataset=data, loaded_source_id=source_id,
        calculation_id=signature, calculated_settings=settings,
        calculated_mode=mode, calculated_files=[name for name, _ in uploads],
        calculated_at=datetime.now().strftime('%H:%M:%S'), revision=st.session_state.get('revision', 0) + 1)
    st.session_state.pop('calculation_error', None)


def sidebar():
    with st.sidebar:
        st.markdown('<div class="brand"><span class="brand-icon">↗</span> StockPilot</div>', unsafe_allow_html=True)
        st.caption('Рабочее место закупщика')
        st.divider()
        st.markdown('### Источник данных')
        mode = st.radio('Режим работы', ['Демонстрация', 'Мои файлы'], key='source_mode', horizontal=True)
        uploads = []
        if mode == 'Мои файлы':
            files = st.file_uploader('Выгрузки поставщиков', type=['zip', 'xlsx'], accept_multiple_files=True, key='files',
                help='Архивы Systeme Electric и IEK или исходные XLSX. Названия файлов сохраняют поставщика.')
            uploads = [(f.name, f.getvalue()) for f in files]
            st.caption('Файлы читаются локально. Исходники не изменяются.')
        else:
            st.caption('8 вымышленных товаров · 2 поставщика\n\nРекомендации меняются вместе с параметрами заказа.')
        st.divider()
        st.markdown('### Параметры заказа')
        as_of = st.date_input('Дата расчёта', value=DEMO_DATE.date(), key='as_of')
        st.caption('Дата должна соответствовать актуальному снимку остатков.')
        a, b = st.columns(2)
        lead_se = a.number_input('Systeme, дней', min_value=0, max_value=365, value=7, key='lead_se')
        lead_iek = b.number_input('IEK, дней', min_value=0, max_value=365, value=7, key='lead_iek')
        review = st.number_input('Пересчёт каждые, дней', min_value=1, max_value=90, value=7, key='review')
        safety = st.number_input('Страховой запас, дней', min_value=0, max_value=90, value=2, key='safety')
        st.caption('Сроки и страховой запас — настройки менеджера, не подтверждённые условия поставщика.')
        category_map = {}
        with st.expander('Страховой запас по категориям'):
            if st.session_state.get('calculated_mode') == mode and 'dataset' in st.session_state:
                categories = sorted(st.session_state.dataset['products'].category.dropna().astype(str).unique())
            else:
                categories = ['A', 'B', 'C'] if mode == 'Демонстрация' else []
            if categories:
                policy = pd.DataFrame({'category': categories, 'days': [int(safety)] * len(categories)})
                key = 'category_' + hashlib.sha256(str((mode, categories, safety)).encode()).hexdigest()[:12]
                edited = st.data_editor(policy, hide_index=True, key=key, disabled=['category'],
                    column_config={'category': 'Категория', 'days': st.column_config.NumberColumn('Дней', min_value=0, max_value=90, step=1)},
                    width='stretch')
                st.caption('Пустое значение использует общий страховой запас.')
                category_map = {cat: int(days) for cat, days in zip(edited.category, edited.days) if pd.notna(days) and days != safety}
            else:
                st.caption('Категории появятся после первой загрузки файлов.')
        confirmed_scopes, constraints = {}, {}
        with st.expander('Подтверждённые условия поставщиков'):
            st.caption('Заполняйте только после проверки с владельцем данных. Эти настройки не исправляют ошибки сверки продаж или отсутствие остатков.')
            for supplier_name, prefix in [('Systeme Electric', 'se'), ('IEK', 'iek')]:
                st.markdown(f'**{supplier_name}**')
                if st.checkbox('Подтверждаю единый контур продаж, остатков и транзита', key=f'scope_confirm_{prefix}'):
                    scope = st.text_input('Подтверждённый склад (__all__ — все склады)', value='__all__', key=f'scope_{prefix}')
                    confirmed_scopes[supplier_name] = scope.strip()
                supplied = {}
                for field, label, initial in [('min_order_qty', 'Минимум заказа', 0.0), ('pack_multiple', 'Кратность заказа', 1.0)]:
                    if st.checkbox(f'{label}: подтверждаю общее условие для пропусков', key=f'{field}_confirm_{prefix}'):
                        supplied[field] = st.number_input(label, min_value=0.0 if field == 'min_order_qty' else 0.001,
                            value=initial, step=1.0, key=f'{field}_{prefix}')
                if supplied:
                    constraints[supplier_name] = supplied
            st.caption('Условия из файлов имеют приоритет. Общая настройка применяется только к отсутствующим значениям и должна подходить каждому такому товару.')
        with st.expander('Возможности прогноза'):
            st.checkbox('Исключать разовые аномалии', value=False, disabled=True)
            st.checkbox('Восстанавливать спрос при stockout', value=False, disabled=True)
            st.caption('В текущей версии доступны базовый прогноз и сезонность. Аномалии, устойчивый тренд и stockout ещё не реализованы.')
        settings = dict(as_of=as_of, lead_time_days={'Systeme Electric': lead_se, 'IEK': lead_iek},
            review_period_days=review, default_safety_days=safety, safety_days_by_category=category_map,
            exclude_anomalies=False, restore_stockouts=False,
            confirmed_warehouse_scope=confirmed_scopes, order_constraints_by_supplier=constraints)
        pressed = st.button('Рассчитать заказ', type='primary', width='stretch', key='calculate')
        st.caption('Изменения параметров требуют пересчёта и нового утверждения.')
    return mode, uploads, settings, pressed


def orders_tab(stale):
    draft = st.session_state.draft
    st.subheader('Рекомендации к закупке')
    st.caption('Проверьте количества, выберите позиции и утвердите состав заказа.')
    a, b, c = st.columns([2, 3, 2])
    supplier = a.selectbox('Поставщик', ['Все поставщики'] + sorted(draft.supplier.unique().tolist()), key='supplier_filter')
    search = b.text_input('Поиск по товару или коду', placeholder='Например, розетка или DEMO-001', key='search')
    status = c.selectbox('Показать', ['Все позиции', 'К заказу', 'Срочные', 'Недостаточно данных'], key='status_filter')
    mask = pd.Series(True, index=draft.index)
    if supplier != 'Все поставщики':
        mask &= draft.supplier.eq(supplier)
    if search:
        mask &= draft.name.fillna('').str.contains(search, case=False, regex=False) | draft.sku.str.contains(search, case=False, regex=False)
    if status == 'К заказу':
        mask &= draft.recommended_qty.gt(0)
    elif status == 'Срочные':
        mask &= draft.urgency.eq('urgent')
    elif status == 'Недостаточно данных':
        mask &= draft.data_quality.eq('insufficient')
    visible = draft.loc[mask].copy()
    eligible = visible.recommended_qty.notna() & visible.data_quality.ne('insufficient') & visible.recommended_qty.gt(0)
    left, right = st.columns([3, 2])
    with left:
        st.caption(f'Показано {len(visible)} из {len(draft)} позиций. Выбор сохраняется при смене фильтра.')
    with right:
        x, y = st.columns(2)
        if x.button('Выбрать', help='Выбрать доступные положительные рекомендации в текущем фильтре.', disabled=stale or not eligible.any(), key='select_visible', width='stretch'):
            draft.loc[visible.index[eligible], 'selected'] = True
            st.session_state.draft = draft
            st.session_state.revision += 1
            clear_approval()
            st.rerun()
        if y.button('Сбросить', help='Снять выбор со всех позиций, включая скрытые фильтром.', disabled=stale, key='clear_selection', width='stretch'):
            draft['selected'] = False
            st.session_state.revision += 1
            clear_approval()
            st.rerun()
    cols = ['selected', 'name', 'recommended_qty', 'approved_qty', 'unit', 'urgency', 'data_quality',
            'supplier', 'sku', 'free_stock', 'incoming_in_horizon', 'reason']
    display = visible[cols].copy()
    display['urgency'] = display.urgency.map(URGENCY)
    display['data_quality'] = display.data_quality.map(QUALITY)
    if display.empty:
        st.info('По выбранным фильтрам ничего не найдено. Измените поиск или статус.')
    else:
        # A new editor identity for every visible key set prevents positional edits
        # from leaking across filters. Canonical edits are stored by supplier + SKU.
        view_id = hashlib.sha256('|'.join(display.index).encode()).hexdigest()[:12]
        editor_key = f'orders_{st.session_state.revision}_{view_id}'
        disabled = True if stale else [col for col in cols if col not in {'selected', 'approved_qty'}]
        edited = st.data_editor(display, hide_index=True, width='stretch', height=min(480, 42 + 35 * len(display)),
            key=editor_key, disabled=disabled, num_rows='fixed',
            column_config={**LABELS,
                'selected': st.column_config.CheckboxColumn('✓', width=45),
                'name': st.column_config.TextColumn('Наименование', width=200),
                'recommended_qty': st.column_config.NumberColumn('Расчёт', format='%.2f', width=85),
                'approved_qty': st.column_config.NumberColumn('К заказу', min_value=0, format='%.2f', width=85,
                    help='Ручное количество. Проверяются минимум и кратность; ноль означает отказ от закупки.'),
                'data_quality': st.column_config.TextColumn('Данные', width='medium')})
        updated = merge_visible_edits(draft, edited)
        if review_signature(updated, st.session_state.calculation_id) != review_signature(draft, st.session_state.calculation_id):
            clear_approval()
        st.session_state.draft = updated
        draft = updated
    selected, errors = validate_selection(draft)
    st.divider()
    l, r = st.columns([3, 2], gap='large')
    with l:
        st.markdown('#### Проверка и утверждение')
        positive = selected.loc[selected.approved_qty.gt(0)]
        st.write(f'Выбрано **{len(selected)}** позиций · к закупке **{len(positive)}** · поставщиков **{positive.supplier.nunique()}**')
        st.caption('Утверждается весь выбранный состав, включая скрытые фильтром строки. Нулевые количества в CSV не попадают.')
        if not selected.empty:
            with st.expander('Проверить полный состав', expanded=False):
                st.dataframe(selected[['supplier', 'sku', 'name', 'unit', 'approved_qty']].rename(columns=LABELS), hide_index=True, width='stretch')
        if errors and not selected.empty:
            for error in errors[:5]:
                st.error(error)
            if len(errors) > 5:
                st.caption(f'И ещё {len(errors)-5} ошибок. Проверьте выбранные строки.')
        confirmed = st.checkbox('Я проверил состав, количества и предупреждения', key='confirm_review', disabled=stale)
        if st.button('Утвердить выбранные позиции', key='approve', type='primary', disabled=stale or bool(errors) or not confirmed):
            stamp = datetime.now().isoformat(timespec='seconds')
            snapshot = draft.copy(deep=True)
            signature = review_signature(draft, st.session_state.calculation_id)
            payload = build_approved_csv(snapshot, calculation_date=st.session_state.calculated_settings['as_of'],
                approved_at=stamp, source_mode=st.session_state.calculated_mode)
            st.session_state.approval = dict(signature=signature, payload=payload, at=stamp,
                count=len(positive), snapshot=snapshot)
    with r:
        st.markdown('#### Готовый заказ')
        approval = st.session_state.get('approval')
        current_signature = review_signature(draft, st.session_state.calculation_id)
        valid = approval is not None and not stale and approval['signature'] == current_signature
        if valid:
            st.success(f'Утверждено: {approval["count"]} позиций. CSV готов.')
            label = 'DEMO_' if st.session_state.calculated_mode == 'Демонстрация' else ''
            st.download_button('Скачать утверждённый CSV', data=approval['payload'], mime='text/csv',
                file_name=f'{label}StockPilot_{st.session_state.calculated_settings["as_of"]}.csv',
                type='primary', width='stretch', key='download_order', on_click='ignore')
            st.caption(f'Снимок утверждения: {approval["at"]}. Строки сгруппированы по поставщику.')
        else:
            st.info('Сначала выберите позиции и подтвердите заказ. После правок нужно утвердить его заново.')
        st.caption('CSV в UTF-8 для Excel. Автоматической отправки поставщикам нет. Состояние текущей сессии не заменяет сохранённый файл.')


def details_tab():
    orders = st.session_state.result['orders']
    st.subheader('Почему предлагается такое количество')
    keys = [f'{r.supplier} / {r.sku} — {r["name"]}' for _, r in orders.iterrows()]
    if not keys:
        st.info('Нет товаров для объяснения.')
        return
    choice = st.selectbox('Товар', range(len(keys)), format_func=lambda i: keys[i], key='details_sku')
    row = orders.iloc[choice]
    st.caption(f'Категория {row.category} · единица {row.unit} · {QUALITY.get(row.data_quality, row.data_quality)}')
    numbers = st.columns(4)
    for col, label, value in zip(numbers, ['Спрос на горизонт', 'Страховой запас', 'Свободный остаток', 'В пути на горизонт'],
                               [row.forecast_qty, row.safety_stock, row.free_stock, row.incoming_in_horizon]):
        col.metric(label, fmt(value))
    st.markdown(f'<div class="formula"><strong>Рекомендация: {escape(fmt(row.recommended_qty))} {escape(str(row.unit))}</strong><br>{escape(str(row.reason))}</div>', unsafe_allow_html=True)
    st.caption('Дата остатка: ' + (str(pd.Timestamp(row.stock_as_of).date()) if pd.notna(row.stock_as_of) else 'нет подтверждённого снимка'))
    history = st.session_state.dataset['monthly_sales']
    history = history.loc[history.supplier.eq(row.supplier) & history.sku.eq(row.sku)].copy()
    forecast = st.session_state.result['forecast']
    forecast = forecast.loc[forecast.supplier.eq(row.supplier) & forecast.sku.eq(row.sku)].copy()
    left, right = st.columns([3, 2], gap='large')
    with left:
        st.markdown('#### История спроса')
        st.caption('Количество за завершённый месяц. Пустые значения не считаются нулём.')
        history = history.loc[history.is_complete.fillna(False)].tail(12)
        if not history.empty:
            plot = history[['month', 'quantity']].rename(columns={'month': 'Месяц', 'quantity': 'Продажи'})
            st.bar_chart(plot, x='Месяц', y='Продажи', color='#147D64', height=250)
        else:
            st.info('Нет завершённых месяцев.')
    with right:
        st.markdown('#### Прогноз по дням')
        st.caption('Отдельная шкала: количество за один день.')
        if not forecast.empty:
            st.line_chart(forecast.rename(columns={'date': 'Дата', 'predicted_qty': 'Прогноз'}), x='Дата', y='Прогноз', color='#147D64', height=250)
        else:
            st.info('Прогноз недоступен для этой позиции.')
    with st.expander('Ожидаемые поступления и ограничения'):
        transit = st.session_state.dataset.get('transit', pd.DataFrame())
        if not transit.empty:
            transit = transit.loc[transit.supplier.eq(row.supplier) & transit.sku.eq(row.sku)]
            st.dataframe(transit, hide_index=True, width='stretch')
        diag = st.session_state.result['diagnostics']
        subset = diag.loc[diag.supplier.eq(row.supplier) & diag.sku.eq(row.sku)].copy()
        if not subset.empty:
            subset['issue'] = subset.issue.map(readable_issue)
            st.dataframe(subset.rename(columns=LABELS), hide_index=True, width='stretch')
    st.info('Исключение аномалий, устойчивый тренд и восстановление stockout ещё не реализованы. Пустая таблица аномалий не означает отсутствие аномалий в продажах.')


def data_tab():
    st.subheader('Источники и качество данных')
    dataset = st.session_state.dataset
    files = st.session_state.get('calculated_files', [])
    st.caption('Источники: ' + (', '.join(files) if files else 'синтетический набор StockPilot, 8 товаров'))
    counts = pd.DataFrame({'Таблица': list(dataset), 'Строк': [len(frame) for frame in dataset.values()]})
    a, b = st.columns([2, 3], gap='large')
    a.dataframe(counts, hide_index=True, width='stretch')
    with b:
        st.markdown('#### Перед расчётом')
        st.write('Проверьте дату свободного остатка, единицы учёта, минимум и кратность. Для IEK исторические остатки не являются актуальным складским снимком.')
        st.caption('Исторические остатки доступны для просмотра; расчёт использует текущий снимок. Значения и ошибки исходных данных сохраняются.')
        with st.expander('Параметры выполненного расчёта'):
            st.json(st.session_state.calculated_settings)
    q = st.session_state.prepared.get('quality_report', pd.DataFrame())
    diag = st.session_state.result['diagnostics']
    report = pd.concat([q.assign(stage='Источник'), diag.assign(stage='Расчёт')], ignore_index=True)
    if report.empty:
        st.success('Замечаний к данным нет.')
    else:
        level = st.selectbox('Уровень сообщения', ['Все', 'Ошибки', 'Предупреждения', 'Информация'], key='diagnostic_level')
        mapping = {'Ошибки': 'error', 'Предупреждения': 'warning', 'Информация': 'info'}
        if level != 'Все':
            report = report.loc[report.severity.eq(mapping[level])]
        report['issue'] = report.issue.map(readable_issue)
        report['severity'] = report.severity.replace({'error': 'Ошибка', 'warning': 'Предупреждение', 'info': 'Информация'})
        st.caption(f'Сообщений: {len(report)}. Строки с недостаточными данными нельзя включить в заказ.')
        st.dataframe(report.rename(columns={**LABELS, 'stage': 'Этап'}), hide_index=True, width='stretch', height=350)
    with st.expander('Посмотреть нормализованные таблицы'):
        table = st.selectbox('Таблица источника', list(dataset), key='source_table')
        st.caption('Показаны первые 200 строк; исходные таблицы не изменяются.')
        st.dataframe(dataset[table].head(200), hide_index=True, width='stretch')


def main():
    style()
    mode, uploads, settings, pressed = sidebar()
    signature = input_signature(mode, uploads, settings)
    if pressed or ('result' not in st.session_state and mode == 'Демонстрация' and 'calculation_error' not in st.session_state):
        try:
            with st.spinner('Читаем данные и рассчитываем рекомендации…'):
                calculate(mode, uploads, settings, signature)
        except Exception as exc:
            # No silent switch to demo: keep the failure explicit and block exports.
            clear_approval()
            st.session_state.calculation_error = f'{type(exc).__name__}: {exc}'
            st.session_state.error_details = traceback.format_exc()
    stale = signature != st.session_state.get('calculation_id') or bool(st.session_state.get('calculation_error'))
    if stale:
        clear_approval()
    st.markdown('<div class="eyebrow">Закупки / Пополнение склада</div>', unsafe_allow_html=True)
    h, badge = st.columns([4, 2])
    h.title('Заказы поставщикам')
    badge.markdown(f'<div style="text-align:right;padding-top:20px"><span class="context-pill">{escape(mode)}<br>{pd.Timestamp(settings["as_of"]).strftime("%d.%m.%Y")}</span></div>', unsafe_allow_html=True)
    st.markdown('<p class="intro">От истории продаж до утверждённого заказа поставщику — в одном рабочем окне.</p>', unsafe_allow_html=True)
    if mode == 'Демонстрация':
        st.info('Демонстрационные данные: товары, остатки и условия вымышлены. Вы можете проверить расчёт, правки и утверждение заказа.')
    if st.session_state.get('calculation_error'):
        st.error('Расчёт не выполнен. ' + st.session_state.calculation_error)
        with st.expander('Технические подробности для команды'):
            st.code(st.session_state.error_details)
    if 'result' not in st.session_state:
        st.info('Загрузите архивы и нажмите «Рассчитать заказ». Можно начать с демонстрационного набора.')
        return
    if stale:
        st.warning('Параметры или источники изменены. Ниже — предыдущий результат; утверждение и экспорт заблокированы до пересчёта.')
    orders = st.session_state.result['orders']
    metrics = st.columns(4)
    metrics[0].metric('Позиций к заказу', int(orders.recommended_qty.gt(0).sum()))
    metrics[1].metric('Риск дефицита', int(orders.urgency.eq('urgent').sum()))
    metrics[2].metric('Требуют данных', int(orders.data_quality.eq('insufficient').sum()))
    metrics[3].metric('Поставщиков', int(orders.supplier.nunique()))
    st.caption(f'Результат: {st.session_state.calculated_mode} · {st.session_state.calculated_settings["as_of"]} · рассчитан в {st.session_state.calculated_at}. Количества разных единиц не суммируются.')
    tabs = st.tabs(['Заказы', 'Объяснение SKU', 'Данные'])
    if orders.empty:
        with tabs[0]:
            st.info('Товары не найдены. Проверьте файлы и вкладку «Данные».')
    else:
        with tabs[0]:
            orders_tab(stale)
        with tabs[1]:
            details_tab()
    with tabs[2]:
        data_tab()
    st.markdown('<div class="foot">StockPilot · HackAlem AI &nbsp; / &nbsp; Нурасыл — данные · Гапар — расчёты · Рамазан — интерфейс</div>', unsafe_allow_html=True)


if __name__ == '__main__':
    main()
