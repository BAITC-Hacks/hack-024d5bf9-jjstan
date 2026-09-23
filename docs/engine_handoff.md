# Передача Рамазану: engine cc15fe0

Версии сверены в GitHub: feat/engine@cc15fe00d16f0e6a1491e0fc627b7ae338c3ac93,
feat/ui@72a4f864b341e1c5de7bbb85abc6536946edc61c, feat/data@c1f657fc071e78f12e80900706f4ee59aacf31d9.
Расчётный модуль сохраняет версию cc15fe0. Эта поставка добавляет примеры и контракт.
Фронтенд и loader не изменялись. Непубликованные изменения Рамазана здесь отсутствуют.

## Контракт

`from engine import ENGINE_CAPABILITIES, calculate_orders`

```python
ENGINE_CAPABILITIES == {
    'version': '2.0-local',
    'exclude_anomalies': True,
    'restore_stockouts': True,
    'trend': True,
    'strict_snapshot_date': True,
}
```

Строка version — фактическое значение опубликованного кода. Готовность UI проверяет
по булевым ключам, а не по суффиксу версии. Готовность не означает, что коррекция
применена к каждому SKU: это зависит от настроек, данных и диагностики.

Настройки `exclude_anomalies=False`, `restore_stockouts=False`, `enable_trend=True`
по умолчанию. Обратите внимание: ключ возможности `trend`, а настройка `enable_trend`.
У as_of, lead_time_days, review_period_days и default_safety_days нет бизнес-значений
по умолчанию: их задаёт интерфейс. safety_days_by_category, confirmed_warehouse_scope,
order_constraints_by_supplier по умолчанию пустые словари. Не создавайте подтверждения автоматически.

Таблица anomalies: `supplier, sku, date, document_id, quantity, excluded, reason`.
Показывайте найденное количество, фактический excluded и reason. При выключенном анализе
или ранней блокировке пустая таблица не доказывает отсутствие аномалий.
Таблица diagnostics: `supplier, sku, severity, issue`.
severity принимает info/warning/error. Отдельного машинного поля blocking/applicability нет.

Разрешение на переход к ручному утверждению проверяется по orders:
`recommended_qty` известно, конечно, неотрицательно, `data_quality` равно ok или warning.
`insufficient` или NaN всегда запрещают утверждение. Это не само утверждение: UI отдельно
проверяет выбор, approved_qty, MOQ/кратность и явное подтверждение менеджера.

Исторический error сохраняется. Для неприменимой ошибки в том же расчёте и у того же
supplier/SKU существует отдельный info с точным началом
`Ошибка сохранена, но неприменима к выбранному периоду ` и окончанием `: ` + исходный issue.
Связывайте эту пару по точному тексту issue и ключу SKU. Именно это сообщение engine,
а не самостоятельное сокращение истории в UI, определяет неприменимость.
Другие error нельзя массово считать историческими. Если есть неприменимая ошибка,
заказ всё равно может быть заблокирован другой причиной — смотрите orders.reason.

Для AI передавайте исходное замечание вместе с объяснением применимости и состоянием
orders. evidence_id, calculation_id и подготовка контекста принадлежат фронтенду.
Engine не вызывает explain_context и не принимает количества от AI.

## Реальные данные

Подробный отчёт real-validation.json передан локально и не включён в Git. Расчёт на 22.09.2026,
сроки обоих поставщиков 7 дней, пересмотр 7, страховка 2, категории {}, все три алгоритма
включены. Эти сроки/страховка — тестовые параметры, не подтверждённые условия поставщиков.
Подтверждения склада, условий MOQ и подмены остатков не вводились.

Результат: 3909 insufficient, 0 допустимых рекомендаций; IEK 3185, Systeme 724.
Первая причина блокировки: ошибки данных 2765, отсутствие завершённой истории 544,
единицы 355, складской охват 245. После первой блокировки могут обнаружиться другие.

Manifest/CSV проверены: SHA256 архива совпадает, 36 SKU × 3 месяца = 108 совпадений
monthly_sales / transactions / CSV / manifest. Дополнительно 216 ссылок на ячейки
месячных продаж и встроенной истории проверены непосредственно в архиве.
Это численная сверка, а не независимое подтверждение смысла источников.
Все 36 остались insufficient. У 26 расхождения попадают в исходное шестимесячное окно.
Окно не сокращено до июня–августа, whitelist SKU не создан.

Менеджер/Нурасыл: подтвердить общий складской охват, единицы и условия MOQ/кратности,
разобрать расхождения в используемом периоде и предоставить текущие остатки IEK.
Рамазан: подключить новый engine, переключатели, диагностику, утверждение/CSV и AI-контекст.

## Синтетические примеры для защиты

Все данные вымышлены. engine_examples.py находится рядом с engine.py в корне репозитория.

```python
from engine_examples import make_examples
from engine import calculate_orders
case = make_examples()['anomaly_on']
result = calculate_orders(case['dataset'], case['settings'])
```

`python engine_examples.py` проверяет все восемь запусков, ожидаемые результаты и неизменность входов.

| Сценарий | Ключ | Прогноз | Заказ |
|---|---|---:|---:|
| Контроль | control_72 | 140 | 72 |
| Выброс, выключено | anomaly_off | 4656.129032258064 | 5232 |
| Выброс, включено | anomaly_on | 140 | 72 |
| Stockout, выключено | stockout_off | 72.25806451612904 | 0 |
| Stockout, включено | stockout_on | 140 | 72 |
| Тренд, выключено | trend_off | 245 | 192 |
| Тренд, включено | trend_on | 367.5 | 336 |
| Неизвестный остаток | unknown_stock | NaN | NaN / insufficient |

Связанные тесты tests/test_engine.py:
- test_known_answer_and_contract
- test_split_document_spike_does_not_inflate_regular_demand
- test_stockout_increases_intensity_and_overlaps_count_once
- test_growth_survives_anomaly_detection_and_is_bounded
- test_missing_required_inputs_are_not_zero_orders
- test_mismatch_applies_only_to_used_months

Проверенный полный проход моей сборки: 114 passed без пропусков, включая реальные
архивы и Streamlit. Дополнительно сейчас пройдены 8 запусков engine_examples.py.
147 passed, 1 skipped — результат, сообщённый Рамазаном; его неопубликованные файлы
в эту локальную проверку не входили.
