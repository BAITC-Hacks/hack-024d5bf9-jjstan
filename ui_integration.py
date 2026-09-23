"""UI-owned boundary adapter. Teammate modules and source tables stay unchanged."""
from copy import deepcopy
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import json
import re
from tempfile import TemporaryDirectory
from zipfile import BadZipFile, ZipFile

import pandas as pd

MAX_TOTAL_BYTES = 80 * 1024 * 1024


def input_signature(mode, uploads, settings):
    parts = [(name, sha256(content).hexdigest()) for name, content in uploads]
    payload = json.dumps([mode, parts, settings], sort_keys=True, default=str, ensure_ascii=False)
    return sha256(payload.encode('utf-8')).hexdigest()


def safe_upload_name(name):
    leaf = re.split(r'[/\\]', name)[-1]
    leaf = re.sub(r'[<>:"|?*\x00-\x1f]', '_', leaf).strip(' .')
    if not leaf or Path(leaf).suffix.lower() not in {'.zip', '.xlsx'}:
        raise ValueError('Выберите ZIP или XLSX с исходным названием поставщика.')
    return leaf


def load_uploads(uploads):
    """Persist only for the synchronous loader call, keeping informative names."""
    if not uploads:
        raise ValueError('Добавьте хотя бы один ZIP или XLSX.')
    if sum(len(content) for _, content in uploads) > MAX_TOTAL_BYTES:
        raise ValueError('Общий объём загрузки превышает 80 МБ.')
    from data_loader import load_data
    with TemporaryDirectory(prefix='stockpilot-') as folder:
        paths = []
        for number, (name, content) in enumerate(uploads):
            safe_name = safe_upload_name(name)
            # Both ZIP and XLSX are ZIP containers. Inspect without extracting.
            try:
                with ZipFile(BytesIO(content)) as archive:
                    if sum(item.file_size for item in archive.infolist()) > 256 * 1024 * 1024:
                        raise ValueError('Распакованный файл превышает лимит 256 МБ.')
                    if len(archive.infolist()) > 5000:
                        raise ValueError('Слишком много элементов в архиве.')
            except BadZipFile as exc:
                raise ValueError(f'Файл «{safe_name}» не является исправным ZIP/XLSX.') from exc
            subfolder = Path(folder) / str(number)
            subfolder.mkdir()
            target = subfolder / safe_name
            target.write_bytes(content)
            paths.append(str(target))
        return load_data(paths)


def prepare_for_engine(dataset, settings):
    """Normalize explicit aliases and pass only current snapshots where present.

    Historical tables remain in the original dataset for display. Null scopes,
    missing MOQ, missing quantities and loader errors are never invented/erased.
    """
    result = {name: frame.copy(deep=True) for name, frame in dataset.items()}
    notes = []
    for table, column in [('stock', 'warehouse'), ('transit', 'warehouse'), ('seasonality', 'category')]:
        frame = result.get(table, pd.DataFrame())
        if column in frame and frame[column].eq('*all*').any():
            frame.loc[frame[column].eq('*all*'), column] = '__all__'
            notes.append(dict(supplier=None, sku=None, severity='info',
                issue=f'Маркер общего контура {table} приведён к контракту __all__.', source='ui_integration'))
    stock = result.get('stock', pd.DataFrame())
    if not stock.empty:
        as_of = pd.Timestamp(settings['as_of']).normalize()
        eligible = stock['is_current'].fillna(False) & pd.to_datetime(stock.as_of).le(as_of)
        current_keys = set(stock.loc[eligible, ['supplier', 'sku']].itertuples(index=False, name=None))
        has_current = pd.Series([(s, k) in current_keys for s, k in zip(stock.supplier, stock.sku)], index=stock.index)
        result['stock'] = stock.loc[eligible | ~has_current].copy()
        for row in stock.loc[eligible].itertuples():
            if pd.Timestamp(row.as_of).normalize() != as_of:
                notes.append(dict(supplier=row.supplier, sku=row.sku, severity='error',
                    issue='Дата актуального снимка не совпадает с датой расчёта. Нужен подтверждённый снимок на выбранную дату.',
                    source='ui_integration'))
    quality = result.get('quality_report', pd.DataFrame(columns=['supplier', 'sku', 'severity', 'issue', 'source']))
    if notes:
        result['quality_report'] = pd.concat([quality, pd.DataFrame(notes)], ignore_index=True)
    return result


def run_calculation(dataset, settings):
    from engine import calculate_orders
    prepared = prepare_for_engine(dataset, settings)
    return calculate_orders(prepared, deepcopy(settings)), prepared
