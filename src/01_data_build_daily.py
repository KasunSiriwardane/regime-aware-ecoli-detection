"""Build the raw daily backbone and pull labeled samples.

The outputs from this script mirror the logic in the research notebook:

* Flow is cubic feet per second (cfs).
* Rain is inches/day.
* Turbidity is in FNU; we capture both mean/median and max where available.
* Conductivity is microsiemens (uS/cm).

Values are kept timezone-naive and aligned to midnight to avoid accidental
timezone drift. Any negative physical measurements are treated as sensor
errors and set to ``NaN`` in the downstream cleaning stage.
"""

import io
import os
import re
from pathlib import Path

import dataretrieval.nwis as nwis
import numpy as np
import pandas as pd
import requests

# --- PATH SETUP ---
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

# ==========================================
# CONFIGURATION
# ==========================================
SITE_ID = '02335000'
SITE_ID_WQP = 'USGS-02335000'
START_DATE = '2000-01-01'
END_DATE = '2025-01-01'
EPA_THRESHOLD_LOG = np.log10(235 + 1)

print(f"--- PHASE 1: DATASET REBUILD (FINAL V6) ---")

# ==========================================
# STEP 1: THE DAILY BACKBONE
# ==========================================
print("\n[1/5] Building Continuous Daily Backbone...")
daily_dates = pd.date_range(start=START_DATE, end=END_DATE, freq='D')
df_daily = pd.DataFrame({'Date': daily_dates})
# Force timezone-naive
df_daily['Date'] = pd.to_datetime(df_daily['Date']).dt.tz_localize(None).dt.normalize()
print(f" > Backbone created. {len(df_daily)} continuous days.")

# ==========================================
# STEP 2: ACQUIRE DAILY PROXIES (NWIS)
# ==========================================
print("\n[2/5] Fetching Daily Proxy Data (NWIS)...")
# 00060: Flow, 63680: Turbidity, 00045: Rain, 00010: Temp, 00095: Cond
params = ['00060', '63680', '00045', '00010', '00095']

try:
    # Fetch Data
    dv = nwis.get_dv(sites=SITE_ID, parameterCd=params, start=START_DATE, end=END_DATE)
    # dataretrieval can return a tuple (df, metadata) or a list; handle both
    if isinstance(dv, tuple):
        df_nwis_raw = dv[0]
    elif isinstance(dv, list) and len(dv) > 0:
        df_nwis_raw = dv[0]
    else:
        df_nwis_raw = dv
    df_nwis = df_nwis_raw.reset_index()
    # 'datetime' column name varies by version; use the first column if needed
    dt_col = 'datetime' if 'datetime' in df_nwis.columns else df_nwis.columns[0]
    df_nwis['Date'] = pd.to_datetime(df_nwis[dt_col], utc=True, errors='coerce').dt.tz_localize(None).dt.normalize()

    # --- LOGIC UPDATE: Robust Text-Based Column Selection ---
    selection_rules = {
        'Flow_cfs': ('00060', ['Mean', 'Max']),
        'Rain_inches': ('00045', ['Sum', 'Max']),
        'Temp_C': ('00010', ['Mean', 'Max']),
        'Cond_uS': ('00095', ['Mean', 'Max']),
        'Turbidity_FNU': ('63680', ['Mean', 'Median']),
    }

    final_cols = ['Date']

    def get_best_col(df_cols, p_code, suffixes):
        # 1. Narrow down to this parameter code (ignore _cd flags)
        candidates = [c for c in df_cols if p_code in c and '_cd' not in c]
        if not candidates: return None, "Missing"
        
        # 2. Look for preferred suffixes in order (Case Insensitive)
        for suffix in suffixes:
            matches = [c for c in candidates if suffix.lower() in c.lower()]
            if matches: return matches[0], suffix
            
        # 3. Fallback: Take the first available column
        return candidates[0], "Fallback"

    # 1. Process Standard Parameters
    for target_name, (p_code, suffixes) in selection_rules.items():
        best_col, found_type = get_best_col(df_nwis.columns, p_code, suffixes)
        if best_col:
            df_nwis[target_name] = df_nwis[best_col]
            df_nwis[f'{target_name}_Source'] = f"{best_col} ({found_type})"
            final_cols.extend([target_name, f'{target_name}_Source'])
            print(f" Mapped {target_name} <- {best_col}")
        else:
            df_nwis[target_name] = np.nan
            df_nwis[f'{target_name}_Source'] = "Missing"
            final_cols.extend([target_name, f'{target_name}_Source'])
            print(f" Warning: {target_name} ({p_code}) not found.")

    # 2. Process Turbidity MAX Specially
    turb_max_col, _ = get_best_col(df_nwis.columns, '63680', ['Maximum', 'Max'])
    if turb_max_col:
        df_nwis['Turbidity_FNU_Max'] = df_nwis[turb_max_col]
        final_cols.append('Turbidity_FNU_Max')
        print(f" Mapped Turbidity_FNU_Max <- {turb_max_col}")
    else:
        df_nwis['Turbidity_FNU_Max'] = np.nan
        final_cols.append('Turbidity_FNU_Max')
        print(f" Warning: Turbidity Max not found.")

    # Clean and Merge
    df_nwis_clean = df_nwis[final_cols].copy()
    # Check for duplicates
    df_nwis_clean = df_nwis_clean.loc[:, ~df_nwis_clean.columns.duplicated()]
    df_daily = df_daily.merge(df_nwis_clean, on='Date', how='left')

    # Create Missing Flags
    proxies = list(selection_rules.keys()) + ['Turbidity_FNU_Max']
    for col in proxies:
        if col in df_daily.columns:
            df_daily[f'{col}_Missing'] = df_daily[col].isna().astype(int)

    # CRITICAL ASSERTION
    flow_coverage = df_daily['Flow_cfs'].notna().mean()
    print(f" > Flow Data Coverage: {flow_coverage:.1%}")
    if flow_coverage < 0.10:
        raise ValueError("CRITICAL: Flow data is missing! Check NWIS column selection")

except Exception as e:
    print(f"CRITICAL ERROR in NWIS: {e}")
    raise e

# ==========================================
# STEP 3: ACQUIRE TARGETS (E. COLI)
# ==========================================
print("\n[3/5] Fetching E. coli Samples...")
wqp_url = (
    f"https://www.waterqualitydata.us/data/Result/search?"
    f"siteid={SITE_ID_WQP}"
    f"&characteristicName=Escherichia%20coli"
    f"&startDateLo=01-01-2000"
    f"&startDateHi=01-01-2025"
    f"&mimeType=csv"
    f"&dataProfile=resultPhysChem"
)

try:
    response = requests.get(wqp_url)
    df_ecoli_raw = pd.read_csv(io.StringIO(response.content.decode('utf-8')))
    
    possible_cols = {
        'ActivityStartDate': 'Date',
        'ActivityStartTime/Time': 'Time',
        'ResultMeasureValue': 'Ecoli_CFU',
        'ResultMeasure/MeasureUnitCode': 'Units',
        'ResultDetectionConditionText': 'Detect_Cond',
        'ResultMeasureQualifierCode': 'Qualifier',
        'ResultAnalyticalMethod/MethodIdentifier': 'Method'
    }
    
    available = {k:v for k,v in possible_cols.items() if k in df_ecoli_raw.columns}
    df_ecoli = df_ecoli_raw[list(available.keys())].copy()
    df_ecoli.rename(columns=available, inplace=True)
    
    # Standardize Date
    df_ecoli['Date'] = pd.to_datetime(df_ecoli['Date'], utc=True).dt.tz_localize(None).dt.normalize()
    
    # FIX: STRICT PANDAS DATE FILTER
    mask = (df_ecoli['Date'] >= pd.to_datetime(START_DATE)) & (df_ecoli['Date'] <= pd.to_datetime(END_DATE))
    df_ecoli = df_ecoli[mask]
    
    # Clean Numeric
    df_ecoli['Ecoli_CFU'] = pd.to_numeric(df_ecoli['Ecoli_CFU'], errors='coerce')
    df_ecoli = df_ecoli.dropna(subset=['Ecoli_CFU'])
    
    # Log Transform & Target
    df_ecoli['Log_Ecoli'] = np.log10(df_ecoli['Ecoli_CFU'] + 1)
    df_ecoli['Target_Unsafe'] = (df_ecoli['Log_Ecoli'] > EPA_THRESHOLD_LOG).astype(int)
    print(f" > Raw Samples (Cleaned & Filtered): {len(df_ecoli)}")
    
    # Aggregation Policy
    agg_funcs = {
        'Ecoli_CFU': ['max', 'median'],
        'Log_Ecoli': 'max',
        'Target_Unsafe': 'max',
    }
    if 'Units' in df_ecoli.columns: agg_funcs['Units'] = 'first'
    if 'Time' in df_ecoli.columns: agg_funcs['Time'] = 'first'
    
    df_ecoli_daily = df_ecoli.groupby('Date').agg(agg_funcs)
    # Flatten MultiIndex
    df_ecoli_daily.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df_ecoli_daily.columns.values]
    
    # Rename
    rename_map = {
        'Ecoli_CFU_max': 'Ecoli_CFU_Max',
        'Ecoli_CFU_median': 'Ecoli_CFU_Median',
        'Log_Ecoli_max': 'Log_Ecoli',
        'Target_Unsafe_max': 'Target_Unsafe',
        'Units_first': 'Units',
        'Time_first': 'Sample_Time'
    }
    rename_final = {k:v for k,v in rename_map.items() if k in df_ecoli_daily.columns}
    df_ecoli_daily = df_ecoli_daily.rename(columns=rename_final).reset_index()
    
    print(f" > Unique Daily Labels: {len(df_ecoli_daily)}")

except Exception as e:
    print(f"CRITICAL ERROR in E. coli Fetch: {e}")
    raise e

# Save
daily_path = DATA / '01_daily_proxies_v6.csv'
samples_path = DATA / '01_ecoli_samples_v6.csv'

df_daily.to_csv(daily_path, index=False)
df_ecoli.to_csv(samples_path, index=False)
print(f"\nSUCCESS. Files saved to {DATA}")