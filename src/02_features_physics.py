"""Causal feature engineering for hydrology and water quality signals.

Derived features and units (aligned with the notebook):
* Flow in cfs; rises are day-over-day increases only (no negative drops).
* Rain in inches; ``Rain_3Day_Sum``/``Rain_7Day_Sum`` use zero-filled totals
  but include accompanying missing-count windows to expose uncertainty.
* Turbidity in FNU; ``Log_Turbidity`` uses a composite of mean then max and
  applies ``log10(x + 1)``.
* Conductivity in uS/cm; ``Cond_Ratio`` compares daily value to a 30-day mean
  (not median).

Negative physical readings are treated as sensor errors and coerced to NaN.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

# --- PATH SETUP ---
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

# Load Phase 1 Output
df_daily = pd.read_csv(DATA / '01_daily_proxies_v6.csv')
df_samples_raw = pd.read_csv(DATA / '01_ecoli_samples_v6.csv')

# Ensure Dates are Datetime
df_daily['Date'] = pd.to_datetime(df_daily['Date'])
df_samples_raw['Date'] = pd.to_datetime(df_samples_raw['Date'])

required_cols = ['Flow_cfs', 'Turbidity_FNU', 'Turbidity_FNU_Max', 'Rain_inches', 'Cond_uS']
missing = [c for c in required_cols if c not in df_daily.columns]
if missing:
    raise ValueError(f"Missing required proxy columns: {missing}")

print("--- PHASE 1.5: PROCESSING & CLEANING ---")

# 1. SANITY CLEANING
cols_to_clean = ['Flow_cfs', 'Turbidity_FNU', 'Turbidity_FNU_Max', 'Rain_inches', 'Cond_uS']
for col in cols_to_clean:
    if col in df_daily.columns:
        neg_mask = df_daily[col] < 0
        if neg_mask.sum() > 0:
            print(f" > Setting {neg_mask.sum()} negative values in {col} to NaN")
            df_daily.loc[neg_mask, col] = np.nan

# 2. TURBIDITY COMPOSITE & LOG
print(f" > Creating Turbidity_Composite...")
df_daily['Turbidity_Composite'] = df_daily['Turbidity_FNU'].fillna(df_daily['Turbidity_FNU_Max'])
df_daily['Turbidity_UsedMaxFlag'] = np.where(df_daily['Turbidity_FNU'].isna() & df_daily['Turbidity_FNU_Max'].notna(), 1, 0)
df_daily['Log_Turbidity'] = np.log10(df_daily['Turbidity_Composite'] + 1)
print(f" Composite Coverage: {df_daily['Turbidity_Composite'].notna().mean():.1%}")

# 3. RAIN GAP HANDLING
df_daily['Rain_Filled'] = df_daily['Rain_inches'].fillna(0)
df_daily['Rain_Missing_Flag'] = df_daily['Rain_inches'].isna().astype(int)

print("\n--- PHASE 2: CAUSAL FEATURE ENGINEERING ---")

# 1. HYDROLOGY (FLOW)
df_daily['Flow_Change'] = df_daily['Flow_cfs'].diff()
df_daily['Flow_Rise'] = np.maximum(df_daily['Flow_Change'], 0)
df_daily['Flow_Rise_Pct'] = df_daily['Flow_Rise'] / (df_daily['Flow_cfs'].shift(1) + 1)

# 2. RAIN HISTORY
df_daily['Rain_3Day_Sum'] = df_daily['Rain_Filled'].rolling(window=3).sum()
df_daily['Rain_7Day_Sum'] = df_daily['Rain_Filled'].rolling(window=7).sum()
df_daily['Rain_3Day_Missing_Count'] = df_daily['Rain_Missing_Flag'].rolling(window=3).sum()
df_daily['Rain_7Day_Missing_Count'] = df_daily['Rain_Missing_Flag'].rolling(window=7).sum()

# 3. DAYS SINCE RAIN
rain_threshold = 0.1
is_rainy = df_daily['Rain_Filled'] > rain_threshold
days_since = []
counter = 0
for r in is_rainy:
    if r:
        counter = 0
    else:
        counter += 1
    days_since.append(counter)
df_daily['Days_Since_Rain'] = days_since

# 4. CONDUCTIVITY & SEASONALITY
df_daily['Cond_Roll30'] = df_daily['Cond_uS'].rolling(window=30, min_periods=15).mean()
df_daily['Cond_Ratio'] = df_daily['Cond_uS'] / df_daily['Cond_Roll30']

day_of_year = df_daily['Date'].dt.dayofyear
df_daily['Season_Sin'] = np.sin(2 * np.pi * day_of_year / 365.25)
df_daily['Season_Cos'] = np.cos(2 * np.pi * day_of_year / 365.25)

# 5. AGGREGATION & MERGE
print("\n[MERGE FIX] Aggregating Samples to Daily Level...")
agg_rules = {
    'Ecoli_CFU': ['max', 'median'],
    'Log_Ecoli': 'max',
    'Target_Unsafe': 'max',
}
if 'Units' in df_samples_raw.columns: agg_rules['Units'] = 'first'
if 'Method' in df_samples_raw.columns: agg_rules['Method'] = 'first'

df_samples_daily = df_samples_raw.groupby('Date').agg(agg_rules)
df_samples_daily.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df_samples_daily.columns.values]

rename_map = {
    'Ecoli_CFU_max': 'Ecoli_CFU_Max',
    'Ecoli_CFU_median': 'Ecoli_CFU_Median',
    'Log_Ecoli_max': 'Log_Ecoli',
    'Target_Unsafe_max': 'Target_Unsafe',
    'Units_first': 'Units',
    'Time_first': 'Sample_Time',
}
rename_final = {k:v for k,v in rename_map.items() if k in df_samples_daily.columns}
df_samples_daily = df_samples_daily.rename(columns=rename_final).reset_index()

# Final Merge
df_modeled = df_daily.merge(df_samples_daily, on='Date', how='left')
df_modeled['Has_Label'] = df_modeled['Log_Ecoli'].notna().astype(int)

# QA
assert len(df_modeled) == len(df_daily), f"Row Mismatch! Daily: {len(df_daily)}, Modeled: {len(df_modeled)}"
print(f" > Assertion Passed: Row count preserved ({len(df_modeled)}).")

out_path = DATA / '02_features_modeled_v6.csv'
df_modeled.to_csv(out_path, index=False)
print(f"\nSUCCESS. Saved to {out_path}")