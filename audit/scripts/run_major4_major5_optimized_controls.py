#!/usr/bin/env python3
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer, RobustScaler, StandardScaler
from sklearn.svm import SVC


BASE = Path("/root/autodl-tmp/尤毅复现")
OUT = BASE / "major4_major5_optimized_outputs"
OUT.mkdir(exist_ok=True)

POS = Path("/root/autodl-tmp/yyfuxian/zip_check_TADE/TADE/datasets/druggable_gene/drugbank/pos_omics_text.csv")
NEG_HARD = BASE / "remaining_experiment_outputs/hard_negative_omics_text_open_targets_strict_top270.csv"
DESC = Path("/root/autodl-tmp/yyfuxian/zip_check_TADE/TADE/data_source/all_gene_desc.csv")
TEMPORAL = Path("/root/autodl-tmp/TADE/code/train_val_test_draw/save/ablation/temporal_gene_correction.csv")

DIABETES_TERMS = [
    "diabetes", "type 2 diabetes", "t2d", "glucose", "insulin", "glycemic",
    "glycaemic", "beta-cell", "pancreatic", "obesity", "adipose", "metabolic",
    "lipid", "cardiometabolic",
]
UNCERTAIN_TERMS = ["uncertain", "unclear", "limited", "not well", "further research", "may", "potential", "suggest"]


def desc_metrics(desc: str) -> dict:
    text = "" if pd.isna(desc) else str(desc)
    lower = text.lower()
    words = re.findall(r"[a-zA-Z0-9]+", lower)
    diabetes_count = sum(lower.count(t) for t in DIABETES_TERMS)
    uncertainty_count = sum(lower.count(t) for t in UNCERTAIN_TERMS)
    return {
        "desc_char_len": len(text),
        "desc_word_len": len(words),
        "diabetes_keyword_count": diabetes_count,
        "diabetes_keyword_density": diabetes_count / max(len(words), 1),
        "uncertainty_count": uncertainty_count,
        "uncertainty_density": uncertainty_count / max(len(words), 1),
        "starts_with_yes": int(lower.strip().startswith("yes")),
    }


def desc_features():
    d = pd.read_csv(DESC)
    m = pd.DataFrame([desc_metrics(x) for x in d["gene_desc"]])
    return pd.concat([d[["gene_name"]], m], axis=1)


def load_strict():
    pos = pd.read_csv(POS).copy()
    neg = pd.read_csv(NEG_HARD).copy()
    pos["label"] = 1
    neg["label"] = 0
    df = pd.concat([pos, neg], ignore_index=True).drop_duplicates("gene_name")
    return df.merge(desc_features(), on="gene_name", how="left")


def top10(y, s):
    y = np.asarray(y)
    s = np.asarray(s)
    k = max(1, math.ceil(0.10 * len(y)))
    idx = np.argsort(s)[::-1][:k]
    precision = float(y[idx].mean())
    lift = precision / float(y.mean())
    return precision, lift


def metric_row(y, s):
    p, l = top10(y, s)
    return {
        "AUROC": roc_auc_score(y, s),
        "AUPRC": average_precision_score(y, s),
        "top10_precision": p,
        "top10_lift": l,
    }


def model_grid():
    return {
        "logreg_l2_standard": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced", C=1.0, random_state=42)),
        "logreg_l2_c03_standard": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced", C=0.3, random_state=42)),
        "logreg_l1_c03_standard": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced", C=0.3, penalty="l1", solver="liblinear", random_state=42)),
        "logreg_l2_robust": make_pipeline(RobustScaler(), LogisticRegression(max_iter=5000, class_weight="balanced", C=1.0, random_state=42)),
        "linear_svm_standard": make_pipeline(StandardScaler(), SVC(kernel="linear", class_weight="balanced", probability=True, C=0.3, random_state=42)),
        "rf_balanced": RandomForestClassifier(n_estimators=800, max_features="sqrt", min_samples_leaf=2, class_weight="balanced_subsample", random_state=42, n_jobs=-1),
        "extra_trees_balanced": ExtraTreesClassifier(n_estimators=1000, max_features="sqrt", min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=120, learning_rate=0.03, max_depth=2, random_state=42),
    }


def run_major4():
    df = load_strict()
    lit_cols = [
        "desc_char_len", "desc_word_len", "diabetes_keyword_count", "diabetes_keyword_density",
        "uncertainty_count", "uncertainty_density", "starts_with_yes",
    ]
    text_cols = [c for c in df.columns if c.startswith("desc_vector_")]
    omics_cols = [c for c in df.columns if c not in {"gene_name", "label"} and not c.startswith("desc_vector_") and c not in lit_cols]
    feature_sets = {
        "literature_density_only": lit_cols,
        "text_only": text_cols,
        "omics_only": omics_cols,
        "full_omics_text": omics_cols + text_cols,
        "full_plus_literature_covariates": omics_cols + text_cols + lit_cols,
    }
    y = df["label"].astype(int).values
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rows = []
    oof = []
    for feature_set, cols in feature_sets.items():
        for model_name, model in model_grid().items():
            # Skip tree models on literature-only only? No, keep all attempts recorded.
            for fold, (tr, te) in enumerate(skf.split(df, y), start=1):
                Xtr = df.iloc[tr][cols].fillna(0).values
                Xte = df.iloc[te][cols].fillna(0).values
                local_model = model
                local_model.fit(Xtr, y[tr])
                score = local_model.predict_proba(Xte)[:, 1]
                m = metric_row(y[te], score)
                rows.append({
                    "experiment_id": f"major4_opt_{feature_set}_{model_name}",
                    "dataset": "open_targets_strict_hard_negative",
                    "feature_set": feature_set,
                    "model": model_name,
                    "fold": fold,
                    **m,
                    "confirmatory_or_exploratory": "optimized_sensitivity",
                })
                oof.extend({
                    "experiment_id": f"major4_opt_{feature_set}_{model_name}",
                    "fold": fold,
                    "gene_name": g,
                    "label": int(lbl),
                    "score": float(sc),
                } for g, lbl, sc in zip(df.iloc[te]["gene_name"], y[te], score))

    per = pd.DataFrame(rows)
    summary = per.groupby(["experiment_id", "dataset", "feature_set", "model", "confirmatory_or_exploratory"]).agg(
        AUROC_mean=("AUROC", "mean"),
        AUROC_std=("AUROC", "std"),
        AUPRC_mean=("AUPRC", "mean"),
        AUPRC_std=("AUPRC", "std"),
        top10_precision_mean=("top10_precision", "mean"),
        top10_lift_mean=("top10_lift", "mean"),
        folds=("fold", "nunique"),
    ).reset_index()
    best_lit = summary[summary.feature_set.eq("literature_density_only")]["AUPRC_mean"].max()
    summary["AUPRC_delta_vs_best_literature_model"] = summary["AUPRC_mean"] - best_lit
    summary["AUPRC_ratio_vs_best_literature_model"] = summary["AUPRC_mean"] / best_lit
    summary = summary.sort_values("AUPRC_mean", ascending=False)
    per.to_csv(OUT / "major4_optimized_model_grid_per_fold.csv", index=False)
    summary.to_csv(OUT / "major4_optimized_model_grid_summary.csv", index=False)
    pd.DataFrame(oof).to_csv(OUT / "major4_optimized_model_grid_oof_scores.csv", index=False)
    return summary


def run_major5():
    t = pd.read_csv(TEMPORAL).merge(desc_features(), on="gene_name", how="left")
    t["text_sparse_composite"] = (
        (t["desc_word_len"] <= t["desc_word_len"].quantile(1 / 3)).astype(int)
        + (t["diabetes_keyword_count"] <= t["diabetes_keyword_count"].quantile(1 / 3)).astype(int)
        + (t["uncertainty_density"] >= t["uncertainty_density"].quantile(2 / 3)).astype(int)
    ) >= 2
    t["omics_support_tertile"] = pd.qcut(t["omics_strength"].rank(method="first"), 3, labels=["low", "medium", "high"])
    t["gene_score_tertile"] = pd.qcut(t["gene_score"].rank(method="first"), 3, labels=["low", "medium", "high"])
    rows = []
    for sparse in [True, False]:
        for tertile in ["low", "medium", "high"]:
            sub = t[t["text_sparse_composite"].eq(sparse) & t["omics_support_tertile"].eq(tertile)]
            if len(sub):
                rows.append(summarize_temporal(sub, f"text_sparse={sparse};omics_strength={tertile}"))
    for sparse in [True, False]:
        for tertile in ["low", "medium", "high"]:
            sub = t[t["text_sparse_composite"].eq(sparse) & t["gene_score_tertile"].eq(tertile)]
            if len(sub):
                rows.append(summarize_temporal(sub, f"text_sparse={sparse};gene_score={tertile}"))
    rows.append(summarize_temporal(t, "all_temporal"))
    t.to_csv(OUT / "major5_temporal_text_sparse_omics_supported_gene_level.csv", index=False)
    pd.DataFrame(rows).to_csv(OUT / "major5_temporal_text_sparse_omics_supported_summary.csv", index=False)


def summarize_temporal(sub, group):
    substantial = sub["abs_delta"] > 0.1
    return {
        "group": group,
        "n": len(sub),
        "desc_word_len_mean": sub["desc_word_len"].mean(),
        "diabetes_keyword_count_mean": sub["diabetes_keyword_count"].mean(),
        "text_score_mean": sub["text_score"].mean(),
        "gene_score_mean": sub["gene_score"].mean(),
        "full_score_mean": sub["full_score"].mean(),
        "positive_shift_fraction": (sub["delta"] > 0).mean(),
        "mean_shift": sub["delta"].mean(),
        "mean_abs_shift": sub["abs_delta"].mean(),
        "substantial_abs_gt_0_1_n": int(substantial.sum()),
        "substantial_up_fraction": (sub.loc[substantial, "delta"] > 0).mean() if substantial.any() else np.nan,
        "omics_strength_mean": sub["omics_strength"].mean(),
    }


def main():
    summary = run_major4()
    run_major5()
    print(summary.head(12).to_string(index=False))
    print(OUT)


if __name__ == "__main__":
    main()
