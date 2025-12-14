from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import xgboost as xgb

# ============================================================
# 04: CHRONIC RISK MODEL -> Prob_Chronic + ChronicScore
#   Input : data/03_scored_storm_v6.csv + data/03_storm_threshold_v6.json
#   Output: data/04_scored_storm_chronic_v6.csv
#           data/04_chronic_model_xgb.json
#           data/04_chronic_calibrator.pkl (optional)
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
IN_FILE = DATA / "03_scored_storm_v6.csv"
STORM_THRESH_FILE = DATA / "03_storm_threshold_v6.json"

OUT_FILE = DATA / "04_scored_storm_chronic_v6.csv"
MODEL_FILE = DATA / "04_chronic_model_xgb.json"
CAL_FILE = DATA / "04_chronic_calibrator.pkl"
META_FILE = DATA / "04_chronic_model_meta.json"

np.random.seed(42)

def vectorize_percentile(values: np.ndarray, ref: np.ndarray) -> np.ndarray:
    ref = np.asarray(ref, dtype=float)
    ref = ref[np.isfinite(ref)]
    if ref.size == 0:
        return np.zeros_like(values, dtype=float)
    ref_sorted = np.sort(ref)
    v = np.asarray(values, dtype=float)
    out = np.zeros_like(v, dtype=float)
    good = np.isfinite(v)
    out[good] = np.searchsorted(ref_sorted, v[good], side="right") / ref_sorted.size
    return np.clip(out, 0.0, 1.0)

def chrono_split_masks(df: pd.DataFrame, train_frac: float = 0.70, calib_frac: float = 0.15):
    df = df.sort_values("Date").reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_frac)
    calib_end = int(n * (train_frac + calib_frac))
    train_mask = df.index < train_end
    calib_mask = (df.index >= train_end) & (df.index < calib_end)
    vault_mask = df.index >= calib_end
    return train_mask, calib_mask, vault_mask, train_end, calib_end

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create any chronic-only features not already present; trailing / causal only."""
    out = df.copy()

    # Flow ratio to trailing 30d mean
    if "Flow_cfs" in out.columns:
        out["Flow_Roll30"] = out["Flow_cfs"].rolling(window=30, min_periods=15).mean()
        out["Flow_Ratio30"] = out["Flow_cfs"] / (out["Flow_Roll30"] + 1e-6)

    # Temp 7d mean
    if "Temp_C" in out.columns:
        out["Temp_7dMean"] = out["Temp_C"].rolling(window=7, min_periods=3).mean()

    # Turb rolling median + anomaly already computed in 03, but keep safe
    if "Log_Turbidity" in out.columns and "LogTurb_30dMed" not in out.columns:
        out["LogTurb_30dMed"] = out["Log_Turbidity"].rolling(window=30, min_periods=15).median()
    if "Log_Turbidity" in out.columns and "LogTurb_Anom" not in out.columns:
        out["LogTurb_Anom"] = out["Log_Turbidity"] - out["LogTurb_30dMed"]
    if "Log_Turbidity" in out.columns and "LogTurb_7dMed" not in out.columns:
        out["LogTurb_7dMed"] = out["Log_Turbidity"].rolling(window=7, min_periods=3).median()

    # Time index trend (slow drift) – useful for non-stationarity diagnostics
    out["Time_Index"] = np.arange(len(out), dtype=float) / max(len(out) - 1, 1)

    return out

def main() -> None:
    if not IN_FILE.exists():
        raise FileNotFoundError(f"Missing input file: {IN_FILE}")
    if not STORM_THRESH_FILE.exists():
        raise FileNotFoundError(f"Missing storm threshold file: {STORM_THRESH_FILE}")

    storm_cfg = json.loads(STORM_THRESH_FILE.read_text())
    best_storm_s = float(storm_cfg["best_storm_s"])

    df = pd.read_csv(IN_FILE)
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None).dt.normalize()
    df = df.sort_values("Date").reset_index(drop=True)
    df = build_features(df)

    train_mask, calib_mask, vault_mask, train_end, calib_end = chrono_split_masks(df)

    # We train chronic model on NON-STORM labeled days (to avoid letting storm dynamics dominate).
    if "Has_Label" not in df.columns or "Target_Unsafe" not in df.columns:
        raise ValueError("Expected Has_Label and Target_Unsafe in dataset.")

    base_nonstorm = (df["StormScore"] <= best_storm_s)
    train_idx = train_mask & (df["Has_Label"] == 1) & base_nonstorm
    calib_idx = calib_mask & (df["Has_Label"] == 1) & base_nonstorm

    # Feature set (physics + causal proxies). Keep small and explainable.
    feature_cols = [
        "Days_Since_Rain",
        "Cond_Ratio",
        "Log_Turbidity",
        "LogTurb_Anom",
        "LogTurb_7dMed",
        "Rain_3Day_Sum",
        "Flow_Rise",
        "Flow_Rise_Pct",
        "Flow_Ratio30",
        "Temp_7dMean",
        "Season_Sin",
        "Season_Cos",
        "Time_Index",
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    X_train = df.loc[train_idx, feature_cols]
    y_train = df.loc[train_idx, "Target_Unsafe"].astype(int)

    if y_train.sum() < 10:
        raise RuntimeError(f"Too few unsafe labels in TRAIN non-storm subset: {int(y_train.sum())}. Can't train chronic model.")

    # XGBoost can handle NaNs; we keep them.
    pos = float(y_train.sum())
    neg = float((y_train == 0).sum())
    scale_pos_weight = neg / max(pos, 1.0)

    clf = xgb.XGBClassifier(
        n_estimators=700,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        min_child_weight=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=4,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
    )

    clf.fit(X_train, y_train)

    # Raw probability for all days
    X_all = df[feature_cols]
    df["Prob_Chronic_Raw"] = clf.predict_proba(X_all)[:, 1]

    # Optional calibration (isotonic) on CALIB labeled non-storm
    calibrator = None
    if calib_idx.sum() >= 60 and df.loc[calib_idx, "Target_Unsafe"].sum() >= 20 and (df.loc[calib_idx, "Target_Unsafe"] == 0).sum() >= 20:
        x_cal = df.loc[calib_idx, "Prob_Chronic_Raw"].values
        y_cal = df.loc[calib_idx, "Target_Unsafe"].astype(int).values
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(x_cal, y_cal)
        df["Prob_Chronic_Cal"] = calibrator.transform(df["Prob_Chronic_Raw"].values)
        cal_used = True
    else:
        df["Prob_Chronic_Cal"] = df["Prob_Chronic_Raw"]
        cal_used = False

    # ChronicScore: percentile rank of RAW probability relative to CALIB distribution (stable, threshold-friendly)
    ref_prob_cal = df.loc[calib_idx, "Prob_Chronic_Raw"].dropna().values
    df["ChronicScore"] = vectorize_percentile(df["Prob_Chronic_Raw"].fillna(0).values, ref_prob_cal)

    # Quick diagnostics
    try:
        auc_train = roc_auc_score(y_train, df.loc[train_idx, "Prob_Chronic_Raw"])
    except Exception:
        auc_train = None

    # Save artifacts
    df.to_csv(OUT_FILE, index=False)
    clf.save_model(str(MODEL_FILE))
    if calibrator is not None:
        with open(CAL_FILE, "wb") as f:
            pickle.dump(calibrator, f)

    meta = {
        "best_storm_s": best_storm_s,
        "train_end_index": int(train_end),
        "calib_end_index": int(calib_end),
        "feature_cols": feature_cols,
        "scale_pos_weight": scale_pos_weight,
        "used_isotonic": cal_used,
        "auc_train_nonstorm": auc_train,
        "notes": "ChronicScore is a CALIB-referenced percentile rank of Prob_Chronic_Raw (not Prob_Chronic_Cal), matching notebook V6 convention.",
    }
    META_FILE.write_text(json.dumps(meta, indent=2))

    print("--- 04: CHRONIC MODEL COMPLETE ---")
    print(f"Saved: {OUT_FILE.name}")
    print(f"Saved model: {MODEL_FILE.name}")
    if calibrator is not None:
        print(f"Saved calibrator: {CAL_FILE.name}")
    print(f"Non-storm TRAIN labels: {int(train_idx.sum())} | unsafe={int(y_train.sum())} | scale_pos_weight={scale_pos_weight:.2f}")
    if auc_train is not None:
        print(f"TRAIN non-storm AUC (raw): {auc_train:.3f}")

if __name__ == "__main__":
    main()
