"""Synthetic fixtures only: no commercial workbooks in Git."""
from zipfile import ZipFile

import pandas as pd
import pytest
from openpyxl import Workbook

from data_loader import ALL, SCHEMAS, load_data


def book(tmp_path, name, rows):
    path = tmp_path / name
    wb = Workbook()
    for row in rows:
        wb.active.append(row)
    wb.save(path)
    wb.close()
    return str(path)


def issues(data):
    return set(data['quality_report'].issue)


def test_empty_schema_and_types():
    data = load_data([])
    assert set(data) == set(SCHEMAS)
    for name, columns in SCHEMAS.items():
        assert list(data[name].columns) == columns
    assert str(data['stock'].as_of.dtype) == 'datetime64[ns]'
    assert str(data['transit'].quantity.dtype) == 'float64'


def test_systeme_end_to_end(tmp_path):
    sales = book(tmp_path, 'Ежемесячные продажи Systeme.xlsx', [
        ['Номенклатура', 'Номенклатура.Код', 'янв. 2026', 'сент. 2026', 'Итого'],
        [None, None, 'Количество', 'Количество', 'Количество'],
        ['Тест', '001_', 8, None, 8], ['Итого', None, 8, None, 8]])
    tx = book(tmp_path, 'Динамика Systeme.xlsx', [
        ['Дата', 'Номер', 'Документ', 'Код', 'Номенклатура', 'Ед.', 'Склад', 'Количество'],
        ['10.01.2026 12:30:00', '123', 'Расходная накладная 123', '001_', 'Тест', 'шт', 'Алматы', 10],
        ['12.01.2026', '124', 'Возврат от покупателя 124', '001_', 'Тест', 'шт', 'Алматы', 2]])
    stock = book(tmp_path, 'Товар в пути Systeme на 22.09.2026.xlsx', [
        ['Код 1с', 'Наименование', 'Категория 2026', 'Свободный остаток', 'Зарезервировано', 'Остаток', 'Витрина', 'СЭ в пути 24.09'],
        ['001_', 'Тест', 2, 30, 7, 37, 5, 12]])
    moq = book(tmp_path, 'MOQ Systeme.xlsx', [
        ['Номенклатура.Код', 'Номенклатура', 'Кратность'], ['001_', 'Тест', 6]])
    data = load_data([sales, tx, stock, moq])
    assert data['products'].sku.tolist() == ['001_']
    assert data['products'].pack_multiple.iloc[0] == 6
    assert pd.isna(data['products'].min_order_qty.iloc[0])
    assert data['monthly_sales'].quantity.iloc[0] == 8
    assert pd.isna(data['monthly_sales'].quantity.iloc[1])
    assert data['monthly_sales'].is_complete.tolist() == [True, False]
    assert data['transactions'].quantity.tolist() == [10, -2]
    assert data['transactions'].transaction_type.tolist() == ['sale', 'return']
    assert data['transactions'].customer_id.isna().all()
    assert data['stock'].free_stock.tolist() == [30]
    assert data['stock'].warehouse.tolist() == [ALL]
    assert data['stock'].is_current.all()
    assert data['stock'].as_of.iloc[0] == pd.Timestamp('2026-09-22')
    assert data['transit'].expected_date.iloc[0] == pd.Timestamp('2026-09-24')
    assert not any(i.startswith('monthly_transaction_mismatch') for i in issues(data))


def test_iek_opening_stock_transit_and_moq(tmp_path):
    stock = book(tmp_path, 'Ежемесячные остатки ИЭК.xlsx', [
        ['Номенклатура', 'Ед.', 'Номенклатура.Код', 'сент. 2026', 'Итого'],
        [None, None, None, 'Количество', 'Количество'],
        [None, None, None, 'нач. остаток', 'нач. остаток'],
        ['Тест', 'м', '0002', 100, 100]])
    transit = book(tmp_path, 'Путь ИЭК 22.09.2026.xlsx', [
        ['Код 1с', 'Артикул ИЭК', 'Наименование', 'от 01.09.2026 (поступление до 10.10.2026)'],
        ['0002', 'TEST', 'Кабель ЗАКУПАЮТСЯ БУХТАМИ, САДЯТСЯ МЕТРАЖОМ', 2]])
    moq = book(tmp_path, 'MOQ ИЭК.xlsx', [
        ['Код 1с', 'Наименование', 'Мин. разр. к отгр.'], ['0002', 'Тест', 1], ['0003', 'Тест 2', 5]])
    data = load_data([stock, transit, moq])
    assert len(data['stock']) == 1
    assert not data['stock'].is_current.any()
    assert data['stock'].as_of.iloc[0] == pd.Timestamp('2026-09-01')
    assert data['stock'].stock_quantity.iloc[0] == 100
    assert data['stock'].free_stock.isna().all()
    assert data['transit'].quantity.isna().all()
    assert data['transit'].raw_quantity.iloc[0] == 2
    assert data['transit'].expected_date.iloc[0] == pd.Timestamp('2026-10-10')
    assert data['products'].set_index('sku').loc['0003','min_order_qty'] == 5
    assert data['products'].pack_multiple.isna().all()
    assert 'coil_meter_conversion_unconfirmed' in issues(data)
    assert 'current_stock_missing' in issues(data)


def test_zip_duplicate_and_numeric_format(tmp_path):
    path = book(tmp_path, 'MOQ Systeme.xlsx', [['Код 1с','Кратность'], [12, 4]])
    from openpyxl import load_workbook
    wb=load_workbook(path)
    wb.active['A2'].number_format='00000'
    wb.save(path)
    wb.close()
    archive = tmp_path/'Systeme.zip'
    with ZipFile(archive, 'w') as z:
        z.write(path, 'Systeme/MOQ.xlsx')
    data = load_data([str(archive),path])
    assert data['products'].sku.tolist() == ['00012']
    assert 'duplicate_file_skipped' in issues(data)


def test_unknown_document_never_becomes_sale(tmp_path):
    path = book(tmp_path,'Динамика IEK.xlsx',[
        ['Дата','Номер','Документ','Код','Номенклатура','Ед.','Склад','Количество'],
        ['bad',1,'Перемещение', '01_', 'Тест', None,None,-15],
        ['22.09.2026',2,'Расходная накладная 2','01_', 'Тест','шт',None,-3]])
    data=load_data([path])
    assert pd.isna(data['transactions'].quantity.iloc[0])
    assert data['transactions'].quantity.iloc[1] == -3
    assert data['transactions'].transaction_type.iloc[1] == 'return'
    assert {'unknown_document_type','unknown_warehouse','invalid_transaction_date'} <= issues(data)


def test_missing_file_is_reported(tmp_path):
    data = load_data([str(tmp_path/'missing.zip')])
    assert any(i.startswith('input_read_error') for i in issues(data))
    assert data['products'].empty


def test_monthly_mismatch_reported(tmp_path):
    sales=book(tmp_path,'Ежемесячные продажи IEK.xlsx',[
        ['Номенклатура.Код','янв. 2026'], ['01',99]])
    tx=book(tmp_path,'Динамика IEK.xlsx',[
        ['Дата','Номер','Документ','Код','Количество'],
        ['01.01.2026',1,'Расходная накладная 1','01',3]])
    data=load_data([sales,tx])
    assert 'monthly_transaction_mismatch:2026-01' in issues(data)
