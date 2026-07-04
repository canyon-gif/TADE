#!/usr/bin/env python3
"""Multimodal late-fusion TADE variants for max-known curated hard negatives."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SEED = 20260702
BASE = Path("/root/autodl-tmp/尤毅复现/max_curated_hard_negative_outputs")
DATA = BASE / "max_known_positive_curated_negative_matrix.csv"


def make_logreg(c: float, penalty: str = "l2"):
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    solver="liblinear",
                    C=c,
                    penalty=penalty,
                    random_state=SEED,
                    max_iter=5000,
                ),
            ),
        ]
    )


def top10(y, scores):
    k = max(1, int(np.ceil(len(y) * 0.10)))
    idx = np.argsort(scores)[::-1][:k]
    p = float(np.mean(y[idx]))
    return p, p / float(np.mean(y))


def rank01(scores):
    order = np.argsort(np.argsort(scores))
    return order / max(1, len(scores) - 1)


def main():
    df = pd.read_csv(DATA)
    y = df["label"].to_numpy()
    text_cols = [c for c in df.columns if c.startswith("desc_vector_")]
    omics_cols = [c for c in df.columns if c not in {"gene_name", "label"} and not c.startswith("desc_vector_")]
    X_text = df[text_cols].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(float)
    X_omics = df[omics_cols].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(float)

    configs = [
        ("text_l2_C0.1__omics_l2_C0.1", make_logreg(0.1, "l2"), make_logreg(0.1, "l2")),
        ("text_l2_C0.03__omics_l2_C0.1", make_logreg(0.03, "l2"), make_logreg(0.1, "l2")),
        ("text_l2_C0.1__omics_l1_C0.1", make_logreg(0.1, "l2"), make_logreg(0.1, "l1")),
        ("text_l2_C0.03__omics_l1_C0.1", make_logreg(0.03, "l2"), make_logreg(0.1, "l1")),
    ]
    text_weights = [0.50, 0.70, 0.80, 0.90, 0.95, 0.98]
    fusion_modes = ["prob_average", "rank_average"]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = []
    oof_rows = []
    for config_name, text_model, omics_model in configs:
        for mode in fusion_modes:
            for wt in text_weights:
                fold_rows = []
                oof = np.full(len(y), np.nan)
                for fold, (tr, te) in enumerate(cv.split(X_text, y), start=1):
                    tm = make_logreg(0.1, "l2")
                    om = make_logreg(0.1, "l2")
                    # Clone by rebuilding from config names to avoid reusing fitted estimators.
                    if "text_l2_C0.03" in config_name:
                        tm = make_logreg(0.03, "l2")
                    if "omics_l1_C0.1" in config_name:
                        om = make_logreg(0.1, "l1")
                    tm.fit(X_text[tr], y[tr])
                    om.fit(X_omics[tr], y[tr])
                    text_scores = tm.predict_proba(X_text[te])[:, 1]
                    omics_scores = om.predict_proba(X_omics[te])[:, 1]
                    if mode == "rank_average":
                        text_scores = rank01(text_scores)
                        omics_scores = rank01(omics_scores)
                    scores = wt * text_scores + (1 - wt) * omics_scores
                    oof[te] = scores
                    p10, lift = top10(y[te], scores)
                    fold_rows.append(
                        {
                            "experiment_id": f"MAXKNOWN_MULTIMODAL_{config_name}_{mode}_textw{wt}",
                            "config": config_name,
                            "fusion_mode": mode,
                            "text_weight": wt,
                            "omics_weight": 1 - wt,
                            "fold": fold,
                            "AUROC": roc_auc_score(y[te], scores),
                            "AUPRC": average_precision_score(y[te], scores),
                            "top10_precision": p10,
                            "top10_lift": lift,
                        }
                    )
                fd = pd.DataFrame(fold_rows)
                rows.append(
                    {
                        "experiment_id": fold_rows[0]["experiment_id"],
                        "config": config_name,
                        "fusion_mode": mode,
                        "text_weight": wt,
                        "omics_weight": 1 - wt,
                        "AUROC_mean": fd["AUROC"].mean(),
                        "AUROC_sd": fd["AUROC"].std(),
                        "AUPRC_mean": fd["AUPRC"].mean(),
                        "AUPRC_sd": fd["AUPRC"].std(),
                        "top10_precision_mean": fd["top10_precision"].mean(),
                        "top10_lift_mean": fd["top10_lift"].mean(),
                        "OOF_AUROC": roc_auc_score(y, oof),
                        "OOF_AUPRC": average_precision_score(y, oof),
                        "risk_level": "sensitivity_multimodal_late_fusion",
                        "safe_wording": "Multimodal late-fusion TADE variant; text and omics models are trained within each fold and fused with pre-specified weights.",
                    }
                )
                oof_rows.append(
                    pd.DataFrame(
                        {
                            "gene_name": df["gene_name"],
                            "label": y,
                            "score": oof,
                            "experiment_id": fold_rows[0]["experiment_id"],
                        }
                    )
                )
    summary = pd.DataFrame(rows).sort_values("AUPRC_mean", ascending=False)
    oof_df = pd.concat(oof_rows, ignore_index=True)
    summary.to_csv(BASE / "max_known_multimodal_late_fusion_summary.csv", index=False)
    oof_df.to_csv(BASE / "max_known_multimodal_late_fusion_oof_scores.csv", index=False)
    print(summary.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
