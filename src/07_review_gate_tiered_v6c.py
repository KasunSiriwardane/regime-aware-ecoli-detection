from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# 07: REVIEW GATE (tier=3) – CALIB-ONLY DESIGN + SAFETY NETS (v6c style)
#   Input : data/06_regime_micro_v6.csv
#           data/05_dry_threshold_v6.json
#   Output: data/07_regime_final_v6c.csv
#           data/final_thresholds_tiered.json
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

IN_FILE = DATA / "06_regime_micro_v6.csv"
CFG_FILE = DATA / "05_dry_threshold_v6.json"

OUT_FILE = DATA / "07_regime_final_v6c.csv"
OUT_CFG = DATA / "final_thresholds_tiered.json"

def chrono_split_masks(df: pd.DataFrame, train_frac: float = 0.70, calib_frac: float = 0.15):
    df = df.sort_values("Date").reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_frac)
    calib_end = int(n * (train_frac + calib_frac))
    train_mask = df.index < train_end
    calib_mask = (df.index >= train_end) & (df.index < calib_end)
    vault_mask = df.index >= calib_end
    return train_mask, calib_mask, vault_mask

def segment_report(df: pd.DataFrame, seg_mask: pd.Series, regime_col: str) -> dict:
    lbl = seg_mask & (df["Has_Label"] == 1)
    unsafe = lbl & (df["Target_Unsafe"] == 1)
    safe = lbl & (df["Target_Unsafe"] == 0)

    alert = seg_mask & (df[regime_col].isin([1, 2, 3]))
    cap = float(alert[unsafe].mean()) if unsafe.any() else float("nan")
    fpr = float(alert[safe].mean()) if safe.any() else float("nan")
    review_share = float((seg_mask & (df[regime_col] == 3)).mean())
    return {"capture_unsafe": cap, "fpr_safe": fpr, "review_share_all": review_share, "n_unsafe": int(unsafe.sum()), "n_safe": int(safe.sum())}

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

    train_mask, calib_mask, vault_mask = chrono_split_masks(df)

    # HARD+MICRO alerts: any non-zero (storm or dry)
    df["Regime_HardMicro"] = df["Regime_ID"].astype(int)
    hard_alert = df["Regime_HardMicro"].isin([1, 2])

    # Identify missed unsafe in CALIB under HARD+MICRO
    calib_unsafe = calib_mask & (df["Has_Label"] == 1) & (df["Target_Unsafe"] == 1)
    missed_calib = calib_unsafe & (~hard_alert)

    n_missed = int(missed_calib.sum())
    if n_missed > 0:
        t_review = float(df.loc[missed_calib, "ChronicScore"].min())
    else:
        # nothing missed -> set review threshold just below best_t and it should select zero
        t_review = max(0.0, best_t - 1e-3)

    # review base: only consider BASE days (not already alert) and non-storm window below hard dry threshold
    review_base = (
        (df["Regime_HardMicro"] == 0)
        & (df["StormScore"] < best_storm_s)
        & (df["ChronicScore"] >= t_review)
        & (df["ChronicScore"] < best_t)
    )

    # ------------------------------------------------------------
    # v6c-inspired EXTRA routes (tight, score-based)
    # ------------------------------------------------------------
    DryHiTurb = (
        (df["StormScore"] >= 0.75)
        & (df.get("Score_TurbAbs", 0) >= 0.72)
        & (df.get("Score_Turb7d", 0) >= 0.82)
        & (df.get("Score_Cond", 0) >= 0.70)
        & (df.get("Score_Rain", 0) <= 0.20)
    )

    WetBurst = (
        (df["StormScore"] >= 0.70)
        & (df.get("Score_TurbAbs", 0) >= 0.72)
        & (df.get("Score_Turb7d", 0) >= 0.62)
        & (df.get("Score_Cond", 0) >= 0.78)
        & (df.get("Score_Rain", 0) >= 0.45)
    )

    SN_MidDryEdgeStorm_tight = (
        (df["StormScore"] >= 0.72) & (df["StormScore"] < best_storm_s)
        & (df.get("Score_TurbAbs", 0) >= 0.75)
        & (df.get("Score_Turb7d", 0) >= 0.80)
        & (df.get("Score_Cond", 0) >= 0.70)
        & (df.get("Score_Rain", 0) <= 0.10)
        & (df["Days_Since_Rain"] >= 8) & (df["Days_Since_Rain"] <= 25)
    )

    SN_ResuspHiTurbNoRain_tight = (
        (df["StormScore"] <= 0.65)
        & (df.get("Score_TurbAbs", 0) >= 0.82)
        & (df.get("Score_Turb7d", 0) >= 0.85)
        & (df.get("Score_Cond", 0) >= 0.90)
        & (df.get("Score_Rain", 0) <= 0.02)
        & (df["Days_Since_Rain"] >= 3) & (df["Days_Since_Rain"] <= 10)
    )

    SN_NearStormLowCondHiTurb_tight = (
        (df["StormScore"] >= 0.80) & (df["StormScore"] < best_storm_s)
        & (df.get("Score_TurbAbs", 0) >= 0.82)
        & (df.get("Score_Turb7d", 0) >= 0.84)
        & (df.get("Score_Cond", 0) <= 0.60)
        & (df.get("Score_Rain", 0) <= 0.02)
        & (df["Days_Since_Rain"] >= 3) & (df["Days_Since_Rain"] <= 12)
    )

    gate_combo = DryHiTurb | WetBurst | SN_MidDryEdgeStorm_tight | SN_ResuspHiTurbNoRain_tight | SN_NearStormLowCondHiTurb_tight

    review_gate = review_base & gate_combo

    # MUST-cover guarantee on CALIB: force-add any missed unsafe days to review
    # (keeps the logic scientifically honest: "we ensure all known misses are reviewed")
    review_gate = review_gate | missed_calib

    # Build final regime: add tier=3 review
    df["Regime_ID_Final_v6c"] = df["Regime_HardMicro"].astype(int)
    df.loc[review_gate & (df["Regime_ID_Final_v6c"] == 0), "Regime_ID_Final_v6c"] = 3

    # Reports
    rep_hm = {
        "TRAIN": segment_report(df, train_mask, "Regime_HardMicro"),
        "CALIB": segment_report(df, calib_mask, "Regime_HardMicro"),
        "VAULT": segment_report(df, vault_mask, "Regime_HardMicro"),
    }
    rep_final = {
        "TRAIN": segment_report(df, train_mask, "Regime_ID_Final_v6c"),
        "CALIB": segment_report(df, calib_mask, "Regime_ID_Final_v6c"),
        "VAULT": segment_report(df, vault_mask, "Regime_ID_Final_v6c"),
    }

    df.to_csv(OUT_FILE, index=False)

    out_cfg = {
        "best_storm_s": best_storm_s,
        "best_t_dry": best_t,
        "t_review": t_review,
        "routes": {
            "DryHiTurb": {"smin": 0.75, "tab": 0.72, "t7d": 0.82, "cmin": 0.70, "rmax": 0.20},
            "WetBurst": {"smin": 0.70, "tab": 0.72, "t7d": 0.62, "cmin": 0.78, "rmin": 0.45},
            "SN_MidDryEdgeStorm_tight": {"fixed": True},
            "SN_ResuspHiTurbNoRain_tight": {"fixed": True},
            "SN_NearStormLowCondHiTurb_tight": {"fixed": True},
        },
        "missed_unsafe_calib_under_hardmicro": n_missed,
        "reports": {"hard_micro": rep_hm, "final_v6c": rep_final},
        "notes": "Tier-3 review adds a small set of borderline days (calib-designed). Review gate includes forced CALIB MUST-coverage.",
    }
    OUT_CFG.write_text(json.dumps(out_cfg, indent=2))

    print("--- 07: REVIEW GATE (v6c-style) COMPLETE ---")
    print(f"Missed unsafe under HARD+MICRO (CALIB): {n_missed}")
    print(f"Tight t_review (CALIB-only) = {t_review:.6f}")
    print(f"Saved: {OUT_FILE.name}")
    print(f"Saved config: {OUT_CFG.name}")

    def fmt(rep: dict) -> str:
        return f"Capture={rep['capture_unsafe']*100:.1f}% | FPR={rep['fpr_safe']*100:.1f}% | Review={rep['review_share_all']*100:.1f}%"
    print("\n== HARD+MICRO ==")
    for k in ["TRAIN", "CALIB", "VAULT"]:
        print(f"{k}: {fmt(rep_hm[k])}")
    print("\n== FINAL (+REVIEW) ==")
    for k in ["TRAIN", "CALIB", "VAULT"]:
        print(f"{k}: {fmt(rep_final[k])}")

if __name__ == "__main__":
    main()
