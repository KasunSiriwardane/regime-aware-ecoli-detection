from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# 05: TUNE DRY THRESHOLD (ChronicScore) + BUILD BASEFRESH REGIME
#   Input : data/04_scored_storm_chronic_v6.csv
#           data/03_storm_threshold_v6.json
#   Output: data/05_regime_basefresh_v6.csv
#           data/05_dry_threshold_v6.json
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

IN_FILE = DATA / "04_scored_storm_chronic_v6.csv"
STORM_THRESH_FILE = DATA / "03_storm_threshold_v6.json"

OUT_FILE = DATA / "05_regime_basefresh_v6.csv"
THRESH_FILE = DATA / "05_dry_threshold_v6.json"

def chrono_split_masks(df: pd.DataFrame, train_frac: float = 0.70, calib_frac: float = 0.15):
    df = df.sort_values("Date").reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_frac)
    calib_end = int(n * (train_frac + calib_frac))
    train_mask = df.index < train_end
    calib_mask = (df.index >= train_end) & (df.index < calib_end)
    vault_mask = df.index >= calib_end
    return train_mask, calib_mask, vault_mask, train_end, calib_end

def metrics_on_segment(df: pd.DataFrame, seg_mask: pd.Series, alert_mask: pd.Series) -> dict:
    lbl = seg_mask & (df["Has_Label"] == 1)
    unsafe = lbl & (df["Target_Unsafe"] == 1)
    safe = lbl & (df["Target_Unsafe"] == 0)
    cap = float(alert_mask[unsafe].mean()) if unsafe.any() else float("nan")
    fpr = float(alert_mask[safe].mean()) if safe.any() else float("nan")
    return {"capture_unsafe": cap, "fpr_safe": fpr, "n_unsafe": int(unsafe.sum()), "n_safe": int(safe.sum())}

def main() -> None:
    if not IN_FILE.exists():
        raise FileNotFoundError(f"Missing input file: {IN_FILE}")
    if not STORM_THRESH_FILE.exists():
        raise FileNotFoundError(f"Missing storm threshold file: {STORM_THRESH_FILE}")

    best_storm_s = float(json.loads(STORM_THRESH_FILE.read_text())["best_storm_s"])

    df = pd.read_csv(IN_FILE)
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None).dt.normalize()
    df = df.sort_values("Date").reset_index(drop=True)

    train_mask, calib_mask, vault_mask, train_end, calib_end = chrono_split_masks(df)
    train_mid = int(train_end * 0.60)  # late-train diagnostic window
    train_late_mask = (df.index >= train_mid) & (df.index < train_end)

    base_nonstorm = df["StormScore"] <= best_storm_s

    # Grid search t over ChronicScore to balance safety vs false positives.
    # NOTE: This is *policy selection* -> tune ONLY on TRAIN/LATE + CALIB, never VAULT.
    candidates = np.round(np.linspace(0.70, 0.99, 147), 3)  # step ~0.002
    best_t = None
    best_cost = np.inf
    best_row = None

    for t in candidates:
        dry_flag = base_nonstorm & (df["ChronicScore"] >= t)
        alert = (df["StormScore"] > best_storm_s) | dry_flag

        m_trainlate = metrics_on_segment(df, train_late_mask, alert)
        m_calib = metrics_on_segment(df, calib_mask, alert)

        # extra diagnostics: dry-only FPR on CALIB safe non-storm days
        lbl_cal = calib_mask & (df["Has_Label"] == 1) & base_nonstorm
        safe_cal = lbl_cal & (df["Target_Unsafe"] == 0)
        fpr_dry_cal = float(dry_flag[safe_cal].mean()) if safe_cal.any() else float("nan")

        # non-storm dry volume on CALIB (all days)
        vol_dry_cal = float(dry_flag[calib_mask & base_nonstorm].mean()) if (calib_mask & base_nonstorm).any() else float("nan")

        mincap = np.nanmin([m_trainlate["capture_unsafe"], m_calib["capture_unsafe"]])

        # Safety-first cost: penalize misses heavily; keep CALIB FPR bounded
        # (tune these weights only once; don't keep adjusting after looking at VAULT).
        cost = (1 - mincap) * 8.0 + (m_calib["fpr_safe"] * 2.0) + (vol_dry_cal * 0.3)

        # hard guardrails (keeps thresholds from becoming silly)
        if np.isfinite(fpr_dry_cal) and fpr_dry_cal > 0.08:
            continue
        if np.isfinite(vol_dry_cal) and vol_dry_cal > 0.25:
            continue

        if cost < best_cost:
            best_cost = cost
            best_t = float(t)
            best_row = {
                "t": best_t,
                "mincap": float(mincap),
                "trainlate_capture": m_trainlate["capture_unsafe"],
                "calib_capture": m_calib["capture_unsafe"],
                "calib_fpr_any": m_calib["fpr_safe"],
                "calib_fpr_dry_nonstorm": fpr_dry_cal,
                "calib_dryvol_nonstorm": vol_dry_cal,
            }

    if best_t is None:
        # fail-safe (matches your stable notebook choice)
        best_t = 0.85
        best_row = {"t": best_t, "note": "fallback"}

    # Build BaseFresh regimes
    df["Regime_ID_BaseFresh"] = 0
    df.loc[df["StormScore"] > best_storm_s, "Regime_ID_BaseFresh"] = 1
    df.loc[(df["StormScore"] <= best_storm_s) & (df["ChronicScore"] >= best_t), "Regime_ID_BaseFresh"] = 2

    # Save dataset and threshold config
    df.to_csv(OUT_FILE, index=False)
    with open(THRESH_FILE, "w") as f:
        json.dump(
            {
                "best_storm_s": best_storm_s,
                "best_t_dry": best_t,
                "selection_summary": best_row,
                "train_end_index": int(train_end),
                "calib_end_index": int(calib_end),
                "notes": "Regime_ID_BaseFresh: 0=base, 1=storm, 2=dry (hard chronic). best_t tuned on TRAIN-late + CALIB only.",
            },
            f,
            indent=2,
        )

    # Quick report
    print("--- 05: DRY THRESHOLD + BASEFRESH REGIME COMPLETE ---")
    print(f"Saved: {OUT_FILE.name}")
    print(f"Saved: {THRESH_FILE.name}")
    print(f"best_storm_s={best_storm_s:.3f} | best_t_dry={best_t:.3f}")
    if best_row:
        print("Selection summary:", best_row)

if __name__ == "__main__":
    main()
