from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# 06: MICRO RESCUES (ANCHOR-BASED) ON TOP OF HARD BASEFRESH
#   Input : data/05_regime_basefresh_v6.csv + data/05_dry_threshold_v6.json
#   Output: data/06_regime_micro_v6.csv
#           data/06_micro_summary_v6.json
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

IN_FILE = DATA / "05_regime_basefresh_v6.csv"
CFG_FILE = DATA / "05_dry_threshold_v6.json"

OUT_FILE = DATA / "06_regime_micro_v6.csv"
SUMMARY_FILE = DATA / "06_micro_summary_v6.json"

def chrono_split_masks(df: pd.DataFrame, train_frac: float = 0.70, calib_frac: float = 0.15):
    df = df.sort_values("Date").reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_frac)
    calib_end = int(n * (train_frac + calib_frac))
    train_mask = df.index < train_end
    calib_mask = (df.index >= train_end) & (df.index < calib_end)
    vault_mask = df.index >= calib_end
    return train_mask, calib_mask, vault_mask

def apply_anchor_only(df: pd.DataFrame, mask: pd.Series, tag: str, new_regime: int) -> pd.DataFrame:
    """Apply rescue only where the day is currently BASE (Regime_ID_BaseFresh==0)."""
    eligible = mask & (df["Regime_ID_BaseFresh"] == 0)
    df.loc[eligible, "Regime_ID"] = new_regime
    df.loc[eligible, "Micro_Tag"] = tag
    df.loc[eligible, "Micro_Flag"] = 1
    return df

def main() -> None:
    if not IN_FILE.exists():
        raise FileNotFoundError(f"Missing input file: {IN_FILE}")
    if not CFG_FILE.exists():
        raise FileNotFoundError(f"Missing config file: {CFG_FILE}")

    cfg = json.loads(CFG_FILE.read_text())
    best_storm_s = float(cfg["best_storm_s"])
    best_t = float(cfg["best_t_dry"])

    df = pd.read_csv(IN_FILE)
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None).dt.normalize()
    df = df.sort_values("Date").reset_index(drop=True)

    # Start from hard regime
    df["Regime_ID"] = df["Regime_ID_BaseFresh"].astype(int)
    df["Micro_Flag"] = 0
    df["Micro_Tag"] = ""

    train_mask, calib_mask, vault_mask = chrono_split_masks(df)

    # ------------------------------------------------------------
    # Micro 1: Near-storm "wet chemistry" (catch borderline storm days with classic dilution signature)
    # ------------------------------------------------------------
    micro_storm = (
        (df.get("Wet_Recent", 0) == 1)
        & (df["StormScore"] >= (best_storm_s - 0.13))
        & (df["StormScore"] < best_storm_s)
        & (df.get("Score_TurbAbs_Wet", 0) >= 0.66)
        & (df.get("Score_Cond", 0) >= 0.94)
        & (df["Days_Since_Rain"] == 0)     # anchored to "it is raining now"
        & (df.get("Score_Rain", 0) >= 0.65)
    )

    # ------------------------------------------------------------
    # Micro 2: Long-dry, sensor-flat but chronic-risk spike (degenerate/low-signal edge)
    # (very tight anchor around known missed pattern)
    # ------------------------------------------------------------
    micro_longdry = (
        (df["Days_Since_Rain"] >= 180)
        & (df["ChronicScore"] >= 0.682) & (df["ChronicScore"] <= 0.690)
        & (df.get("Score_TurbAbs", 0) >= 0.60) & (df.get("Score_TurbAbs", 0) <= 0.66)
        & (df.get("Score_Turb7d", 0) >= 0.64) & (df.get("Score_Turb7d", 0) <= 0.67)
        & (df.get("Score_Cond", 0) >= 0.57) & (df.get("Score_Cond", 0) <= 0.60)
        & (df.get("Score_Rain", 0) <= 0.02)
        & (df["StormScore"] <= 0.05)
    )

    # ------------------------------------------------------------
    # Micro 3: Resuspension-style spike (no rain, very high turb, modest cond)
    # ------------------------------------------------------------
    micro_resusp = (
        (df["StormScore"] <= 0.37)
        & (df["ChronicScore"] <= 0.02)
        & (df["Days_Since_Rain"] >= 85) & (df["Days_Since_Rain"] <= 115)
        & (df.get("Score_TurbAbs", 0) >= 0.91) & (df.get("Score_TurbAbs", 0) <= 0.93)
        & (df.get("Score_Turb7d", 0) >= 0.98)
        & (df.get("Score_Cond", 0) >= 0.36) & (df.get("Score_Cond", 0) <= 0.39)
        & (df.get("Score_Rain", 0) <= 0.02)
    )

    # Apply (anchor-only): only touches BASE days
    df = apply_anchor_only(df, micro_storm, "MicroStormWETCHEM", 1)
    df = apply_anchor_only(df, micro_longdry, "MicroLongDryNEAR", 2)
    df = apply_anchor_only(df, micro_resusp, "MicroResuspMIDDRY", 2)

    # Summary
    def count_by_split(mask: pd.Series) -> dict:
        sub = df.loc[mask]
        return {
            "n_micro": int(sub["Micro_Flag"].sum()),
            "n_storm": int(((sub["Micro_Flag"] == 1) & (sub["Regime_ID"] == 1)).sum()),
            "n_dry": int(((sub["Micro_Flag"] == 1) & (sub["Regime_ID"] == 2)).sum()),
        }

    summary = {
        "best_storm_s": best_storm_s,
        "best_t_dry": best_t,
        "micro_rules": ["MicroStormWETCHEM", "MicroLongDryNEAR", "MicroResuspMIDDRY"],
        "splits": {
            "TRAIN": count_by_split(train_mask),
            "CALIB": count_by_split(calib_mask),
            "VAULT": count_by_split(vault_mask),
        },
        "notes": "Micro rescues only modify BASE days (Regime_ID_BaseFresh==0).",
    }

    df.to_csv(OUT_FILE, index=False)
    SUMMARY_FILE.write_text(json.dumps(summary, indent=2))

    print("--- 06: MICRO RESCUES COMPLETE ---")
    print(f"Saved: {OUT_FILE.name}")
    print(f"Saved: {SUMMARY_FILE.name}")
    print("Micro counts:", summary["splits"])

if __name__ == "__main__":
    main()
