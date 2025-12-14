from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# 08: METRICS + EXPORTS (tables for paper/report)
#   Input : data/07_regime_final_v6c.csv
#           data/final_thresholds_tiered.json
#   Output: data/08_metrics_summary.csv
#           data/08_confusion_by_split.csv
#           data/08_advisory_days.csv
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

IN_FILE = DATA / "07_regime_final_v6c.csv"
CFG_FILE = DATA / "final_thresholds_tiered.json"

OUT_METRICS = DATA / "08_metrics_summary.csv"
OUT_CONF = DATA / "08_confusion_by_split.csv"
OUT_DAYS = DATA / "08_advisory_days.csv"

def chrono_split_masks(df: pd.DataFrame, train_frac: float = 0.70, calib_frac: float = 0.15):
    df = df.sort_values("Date").reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_frac)
    calib_end = int(n * (train_frac + calib_frac))
    train_mask = df.index < train_end
    calib_mask = (df.index >= train_end) & (df.index < calib_end)
    vault_mask = df.index >= calib_end
    return {"TRAIN": train_mask, "CALIB": calib_mask, "VAULT": vault_mask}

def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    rec = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = (2 * prec * rec / (prec + rec)) if np.isfinite(prec) and np.isfinite(rec) and (prec + rec) > 0 else float("nan")
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn, "precision": prec, "recall": rec, "f1": f1}

def main() -> None:
    if not IN_FILE.exists():
        raise FileNotFoundError(f"Missing input file: {IN_FILE}")
    if not CFG_FILE.exists():
        raise FileNotFoundError(f"Missing config file: {CFG_FILE}")

    cfg = json.loads(CFG_FILE.read_text())

    df = pd.read_csv(IN_FILE)
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None).dt.normalize()
    df = df.sort_values("Date").reset_index(drop=True)

    splits = chrono_split_masks(df)

    regime_col = "Regime_ID_Final_v6c"
    alert = df[regime_col].isin([1, 2, 3]).astype(int)

    metrics_rows = []
    conf_rows = []

    for split_name, mask in splits.items():
        seg = df.loc[mask].copy()
        # all-days volumes
        metrics_rows.append({
            "split": split_name,
            "days_total": int(len(seg)),
            "storm_share": float((seg[regime_col] == 1).mean()),
            "dry_share": float((seg[regime_col] == 2).mean()),
            "review_share": float((seg[regime_col] == 3).mean()),
            "alert_share_any": float(seg[regime_col].isin([1,2,3]).mean()),
        })

        # labeled-only classification metrics
        seg_lbl = seg.loc[seg["Has_Label"] == 1]
        if len(seg_lbl) == 0:
            continue

        y = seg_lbl["Target_Unsafe"].astype(int).values
        yhat = seg_lbl[regime_col].isin([1,2,3]).astype(int).values
        m = classification_metrics(y, yhat)

        # append with classification results
        metrics_rows[-1].update({
            "labels_n": int(len(seg_lbl)),
            "unsafe_n": int(y.sum()),
            "safe_n": int((y==0).sum()),
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
        })
        conf_rows.append({"split": split_name, **m})

    metrics_df = pd.DataFrame(metrics_rows)
    conf_df = pd.DataFrame(conf_rows)

    metrics_df.to_csv(OUT_METRICS, index=False)
    conf_df.to_csv(OUT_CONF, index=False)

    # Advisory-day export (for stakeholders / plots)
    out_days = df[["Date", "Has_Label", "Target_Unsafe", "StormScore", "ChronicScore", regime_col]].copy()
    out_days["Alert"] = out_days[regime_col].isin([1,2,3]).astype(int)
    out_days["Regime_Name"] = out_days[regime_col].map({0:"Base", 1:"Storm", 2:"Dry", 3:"Review"})
    out_days.to_csv(OUT_DAYS, index=False)

    print("--- 08: METRICS + EXPORTS COMPLETE ---")
    print(f"Saved: {OUT_METRICS.name}")
    print(f"Saved: {OUT_CONF.name}")
    print(f"Saved: {OUT_DAYS.name}")
    print("\nKey config:")
    print(f"  best_storm_s={cfg.get('best_storm_s')}")
    print(f"  best_t_dry={cfg.get('best_t_dry')}")
    print(f"  t_review={cfg.get('t_review')}")

if __name__ == "__main__":
    main()
