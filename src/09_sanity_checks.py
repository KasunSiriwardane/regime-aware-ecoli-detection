# src/09_sanity_checks.py
import json, os
import pandas as pd

DATA_DIR = "../data"
final_csv = os.path.join(DATA_DIR, "07_regime_final_v6c.csv")
metrics_csv = os.path.join(DATA_DIR, "08_metrics_summary.csv")

df = pd.read_csv(final_csv, parse_dates=["Date"])
assert len(df) == 9133, f"Expected 9133 rows, got {len(df)}"
assert df["Date"].is_monotonic_increasing, "Date not monotonic"
assert df["Date"].diff().dropna().dt.days.value_counts().index[0] == 1, "Not daily frequency"

# Storm share sanity
for split in ["TRAIN","CALIB","VAULT"]:
    share = (df.loc[df["Split"]==split, "Regime_ID"]==1).mean()
    assert share < 0.40, f"{split} storm share too high: {share:.1%}"

# Review share sanity (if you expect review to exist)
for split in ["TRAIN","CALIB","VAULT"]:
    rshare = (df.loc[df["Split"]==split, "Regime_ID_Final"]==3).mean() if "Regime_ID_Final" in df else 0.0
    # loosen if you *really* want 0, but generally this should be >0 if review gate is active
    # assert rshare > 0.001, f"{split} review share is zero; review gate likely not applied"

print("SANITY CHECKS PASSED")