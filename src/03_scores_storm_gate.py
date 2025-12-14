from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# 03: STORM PHYSICS SCORE + STORM GATE THRESHOLD
#   Input : data/02_features_modeled_v6.csv
#   Output: data/03_scored_storm_v6.csv
#           data/03_storm_threshold_v6.json
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
IN_FILE = DATA / "02_features_modeled_v6.csv"
OUT_FILE = DATA / "03_scored_storm_v6.csv"
THRESH_FILE = DATA / "03_storm_threshold_v6.json"

np.random.seed(42)

def vectorize_percentile(values: np.ndarray, ref: np.ndarray, clip_low: float = 0.0, clip_high: float = 1.0) -> np.ndarray:
    """
    Fast percentile score in [0,1] using an empirical CDF based on a reference sample.
    NaNs in `values` -> 0.0 (conservative).
    """
    ref = np.asarray(ref, dtype=float)
    ref = ref[np.isfinite(ref)]
    if ref.size == 0:
        return np.zeros_like(values, dtype=float)
    ref_sorted = np.sort(ref)

    v = np.asarray(values, dtype=float)
    out = np.zeros_like(v, dtype=float)
    good = np.isfinite(v)
    # fraction <= v
    out[good] = np.searchsorted(ref_sorted, v[good], side="right") / ref_sorted.size
    out = np.clip(out, clip_low, clip_high)
    return out

def chrono_split_masks(df: pd.DataFrame, train_frac: float = 0.70, calib_frac: float = 0.15) -> tuple[pd.Series, pd.Series, pd.Series, int, int]:
    df = df.sort_values("Date").reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_frac)
    calib_end = int(n * (train_frac + calib_frac))
    train_mask = df.index < train_end
    calib_mask = (df.index >= train_end) & (df.index < calib_end)
    vault_mask = df.index >= calib_end
    return train_mask, calib_mask, vault_mask, train_end, calib_end

def main() -> None:
    if not IN_FILE.exists():
        raise FileNotFoundError(f"Missing input file: {IN_FILE}")

    df = pd.read_csv(IN_FILE)
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None).dt.normalize()
    df = df.sort_values("Date").reset_index(drop=True)

    # --- Rolling storm physics signals (all trailing / causal)
    # 7d median of log turb (captures "persistently elevated")
    if "Log_Turbidity" not in df.columns:
        raise ValueError("Expected Log_Turbidity from script 02.")
    df["LogTurb_7dMed"] = df["Log_Turbidity"].rolling(window=7, min_periods=3).median()

    # 30d median baseline + anomaly (captures "sudden spike relative to recent baseline")
    df["LogTurb_30dMed"] = df["Log_Turbidity"].rolling(window=30, min_periods=15).median()
    df["LogTurb_Anom"] = df["Log_Turbidity"] - df["LogTurb_30dMed"]

    # Wet_Recent flag (conservative): only trust rain windows if window is complete
    # Also allow very small rain sums / sensor behavior by adding a Days_Since_Rain backstop
    if "Rain_3Day_Sum" not in df.columns or "Rain_3Day_Missing_Count" not in df.columns:
        raise ValueError("Expected Rain_3Day_Sum and Rain_3Day_Missing_Count from script 02.")
    if "Days_Since_Rain" not in df.columns:
        raise ValueError("Expected Days_Since_Rain from script 02.")
    df["Wet_Recent"] = (
        ((df["Rain_3Day_Missing_Count"].fillna(3) == 0) & (df["Rain_3Day_Sum"].fillna(0) > 0.01))
        | (df["Days_Since_Rain"].fillna(999) <= 2)
    ).astype(int)

    # --- Split masks (time-ordered)
    train_mask, calib_mask, vault_mask, train_end, calib_end = chrono_split_masks(df)

    # Reference distributions (TRAIN only) to avoid leakage
    # Use all TRAIN days (not only labeled) for hydrologic intensity distributions.
    ref_flow = df.loc[train_mask, "Flow_Rise"].dropna().values if "Flow_Rise" in df.columns else np.array([])
    ref_turb_anom = df.loc[train_mask, "LogTurb_Anom"].dropna().values
    ref_rain = df.loc[train_mask, "Rain_3Day_Sum"].fillna(0).values
    ref_turb_abs = df.loc[train_mask, "Log_Turbidity"].dropna().values
    ref_turb7 = df.loc[train_mask, "LogTurb_7dMed"].dropna().values
    ref_cond = (1 - df.loc[train_mask, "Cond_Ratio"].fillna(0).values) if "Cond_Ratio" in df.columns else np.array([])

    # Storm physics scores (each is a percentile rank)
    df["Score_Flow"] = vectorize_percentile(df["Flow_Rise"].fillna(0).values, ref_flow)
    df["Score_TurbAnom"] = vectorize_percentile(df["LogTurb_Anom"].fillna(0).values, ref_turb_anom)
    df["Score_Rain"] = vectorize_percentile(df["Rain_3Day_Sum"].fillna(0).values, ref_rain)
    df["Score_TurbAbs"] = vectorize_percentile(df["Log_Turbidity"].fillna(0).values, ref_turb_abs)
    df["Score_Turb7d"] = vectorize_percentile(df["LogTurb_7dMed"].fillna(0).values, ref_turb7)
    df["Score_Cond"] = vectorize_percentile((1 - df["Cond_Ratio"].fillna(0)).values if "Cond_Ratio" in df.columns else np.zeros(len(df)),
                                            ref_cond)

    # Key trick: absolute turbidity counts as "storm signal" only when Wet_Recent is true
    df["Score_TurbAbs_Wet"] = df["Score_TurbAbs"] * df["Wet_Recent"]

    # StormScore = max of storm-mechanism channels (keeps dry high-turb events from auto-counting as storm)
    df["StormScore"] = df[["Score_Flow", "Score_TurbAnom", "Score_Rain", "Score_TurbAbs_Wet"]].max(axis=1)

    # ------------------------------------------------------------
    # Threshold selection (TRAIN labeled only): pick a threshold that
    #   - favors capturing unsafe points
    #   - penalizes false storm flags on safe points
    #   - slightly penalizes very large storm volume
    # If anything goes weird, fall back to 0.85 (your stable V6 value).
    # ------------------------------------------------------------
    best_storm_s = 0.85
    if "Has_Label" in df.columns and "Target_Unsafe" in df.columns:
        train_lbl = train_mask & (df["Has_Label"] == 1)
        y = df.loc[train_lbl, "Target_Unsafe"].astype(int).values
        s = df.loc[train_lbl, "StormScore"].values

        if y.sum() > 0 and (1 - y).sum() > 0:
            candidates = np.round(np.linspace(0.65, 0.95, 61), 3)
            best_cost = np.inf
            for t in candidates:
                pred_storm = (s > t).astype(int)
                # capture unsafe (recall)
                cap = (pred_storm[y == 1].mean()) if (y == 1).any() else 0.0
                # "FPR" of storm-flag on safe points
                fpr = (pred_storm[y == 0].mean()) if (y == 0).any() else 1.0
                vol = pred_storm.mean()

                # weighted cost (safety-first)
                cost = (1 - cap) * 6.0 + fpr * 1.5 + vol * 0.25
                if cost < best_cost:
                    best_cost = cost
                    best_storm_s = float(t)

    # Apply storm regime
    df["StormFlag"] = (df["StormScore"] > best_storm_s).astype(int)

    # Save
    df.to_csv(OUT_FILE, index=False)
    with open(THRESH_FILE, "w") as f:
        json.dump(
            {
                "best_storm_s": best_storm_s,
                "train_end_index": int(train_end),
                "calib_end_index": int(calib_end),
                "notes": "StormScore= max(Score_Flow, Score_TurbAnom, Score_Rain, Score_TurbAbs_Wet). Scores are TRAIN-referenced percentiles.",
            },
            f,
            indent=2,
        )

    print("--- 03: STORM SCORE COMPLETE ---")
    print(f"Saved: {OUT_FILE.name}")
    print(f"Saved: {THRESH_FILE.name}")
    print(f"Best storm threshold: {best_storm_s:.3f}")
    print("Storm share by split:")
    for name, mask in [("TRAIN", train_mask), ("CALIB", calib_mask), ("VAULT", vault_mask)]:
        print(f"  {name}: {df.loc[mask, 'StormFlag'].mean()*100:.1f}%")

if __name__ == "__main__":
    main()
