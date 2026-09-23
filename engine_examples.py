"""Synthetic defense cases. Copy beside engine.py; python engine_examples.py validates them."""
from copy import deepcopy
import json
import math
import pandas as pd
from engine import calculate_orders


def baseline():
    dataset = {
        'products': pd.DataFrame([dict(supplier='IEK',sku='SYNTH-001',supplier_article='SYNTH',name='SYNTHETIC TEST',category='A',unit='шт',min_order_qty=24,pack_multiple=12)]),
        'monthly_sales': pd.DataFrame([dict(supplier='IEK',sku='SYNTH-001',month=pd.Timestamp('2026-03-01'),quantity=310.0,is_complete=True)]),
        'stock': pd.DataFrame([dict(supplier='IEK',sku='SYNTH-001',warehouse='__all__',as_of=pd.Timestamp('2026-04-01'),free_stock=50.0,is_current=True)]),
        'transit': pd.DataFrame([dict(supplier='IEK',sku='SYNTH-001',warehouse='__all__',expected_date=pd.Timestamp('2026-04-02'),quantity=40.0)]),
        'seasonality': pd.DataFrame([dict(supplier='IEK',category='__all__',month_number=m,factor=1.0) for m in range(1,13)]),
        'transactions': pd.DataFrame(columns='supplier sku date document_id customer_id warehouse quantity transaction_type'.split()),
        'stockouts': pd.DataFrame(columns='supplier sku warehouse start_date end_date'.split()),
        'quality_report': pd.DataFrame(columns='supplier sku severity issue source'.split()),
    }
    settings=dict(as_of='2026-04-01',lead_time_days={'IEK':7},review_period_days=7,
        default_safety_days=2,safety_days_by_category={},exclude_anomalies=False,restore_stockouts=False,enable_trend=True)
    return dataset,settings


def make_examples():
    """Each value has dataset/settings/expected; every call returns fresh frames."""
    examples={}
    def add(name,data,settings,forecast,qty):
        examples[name]=dict(dataset=deepcopy(data),settings=deepcopy(settings),expected=dict(forecast_qty=forecast,recommended_qty=qty))
    data,settings=baseline()
    add('control_72',data,settings,140,72)
    data['transactions']=pd.DataFrame([dict(supplier='IEK',sku='SYNTH-001',date=day,document_id=f'day-{day.day}',customer_id=None,warehouse='__all__',quantity=10.0,transaction_type='sale') for day in pd.date_range('2026-03-01','2026-03-31')]+[
        dict(supplier='IEK',sku='SYNTH-001',date=pd.Timestamp('2026-03-15'),document_id='one-off',customer_id=None,warehouse='__all__',quantity=10000.0,transaction_type='sale')])
    data['monthly_sales']['quantity']=10310.0
    add('anomaly_off',data,settings,10310/31*14,5232)
    settings['exclude_anomalies']=True
    add('anomaly_on',data,settings,140,72)
    data,settings=baseline()
    data['monthly_sales']['quantity']=160.0
    data['stockouts']=pd.DataFrame([dict(supplier='IEK',sku='SYNTH-001',warehouse='__all__',start_date=pd.Timestamp('2026-03-01'),end_date=pd.Timestamp('2026-03-15'),confirmed=True,unit='шт')])
    add('stockout_off',data,settings,160/31*14,0)
    settings['restore_stockouts']=True
    add('stockout_on',data,settings,140,72)
    data,settings=baseline()
    data['monthly_sales']=pd.DataFrame([dict(supplier='IEK',sku='SYNTH-001',month=m,quantity=(10+3*i)*m.days_in_month,is_complete=True) for i,m in enumerate(pd.date_range('2025-10-01',periods=6,freq='MS'))])
    settings['enable_trend']=False
    add('trend_off',data,settings,245,192)
    settings['enable_trend']=True
    add('trend_on',data,settings,367.5,336)
    data,settings=baseline()
    data['stock']['free_stock']=float('nan')
    add('unknown_stock',data,settings,None,None)
    return examples


def verify_examples():
    rows=[]
    for name,case in make_examples().items():
        before=deepcopy(case['dataset'])
        result=calculate_orders(case['dataset'],case['settings'])
        row=result['orders'].iloc[0]
        for field,expected in case['expected'].items():
            assert pd.isna(row[field]) if expected is None else math.isclose(row[field],expected,abs_tol=1e-7), (name,field,row[field],expected)
        if name=='unknown_stock': assert row.data_quality=='insufficient'
        else: assert row.data_quality in ['ok','warning']
        if name=='anomaly_on':
            assert len(result['anomalies'])==1 and result['anomalies'].excluded.all()
        for table in before: pd.testing.assert_frame_equal(case['dataset'][table],before[table])
        rows.append(dict(example=name,forecast_qty=None if pd.isna(row.forecast_qty) else row.forecast_qty,
            recommended_qty=None if pd.isna(row.recommended_qty) else row.recommended_qty,data_quality=row.data_quality))
    return rows


if __name__=='__main__':
    print(json.dumps(verify_examples(),ensure_ascii=False,indent=2))
