#!/usr/bin/env python3
"""Curated no-T2D-evidence target-like hard negatives and method comparison."""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SEED = 20260702
ROOT = Path("/root/autodl-tmp")
WORK = ROOT / "尤毅复现"
TADE = ROOT / "TADE"
YY_TADE = ROOT / "yyfuxian/zip_check_TADE/TADE"
COMP = ROOT / "test_data_gpt4o_mini/comparation_algorithm_folder"
OUT = WORK / "curated_no_t2d_hard_negative_outputs"

POS = YY_TADE / "datasets/druggable_gene/drugbank/pos_omics_text.csv"
ALL_OMICS = YY_TADE / "data_source/all_omics_text.csv"
ALL_DESC = YY_TADE / "data_source/all_gene_desc.csv"
OPEN_GENES = TADE / "data_source/open_genes/open_genes.tsv"
CANDIDATE_TABLE = TADE / "results/candidate_genes/Supplementary_Table_3_candidate_genes.csv"

METHOD_FILES = {
    "CTD": COMP / "CTD.csv",
    "DISEASES": COMP / "DISEASES.csv",
    "Know-GENE": COMP / "Know-GENE.csv",
    "ProphNet": COMP / "ProphNet.csv",
    "GUILDify": COMP / "GUILDify.csv",
    "TIGA": COMP / "TIGA.csv",
    "Geneshot": COMP / "Geneshot.csv",
    "PubMed-Score": COMP / "PubMed-Score.csv",
    "T2DKP-CVBF": COMP / "T2DKP-CVBF.csv",
    "T2DKP-RVBF": COMP / "T2DKP-RVBF.csv",
}

DIABETES_TERMS = [
    "type 2 diabetes",
    "t2d",
    "diabetes",
    "glucose",
    "insulin",
    "beta-cell",
    "beta cell",
    "pancreatic",
    "adipose",
    "obesity",
    "metabolic",
]


def numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0.0)


def count_terms(text: str) -> int:
    low = str(text).lower()
    return sum(len(re.findall(re.escape(term), low)) for term in DIABETES_TERMS)


def load_method_scores() -> dict[str, pd.DataFrame]:
    scores = {}
    for method, path in METHOD_FILES.items():
        df = pd.read_csv(path)
        if "gene" in df.columns and "gene_name" not in df.columns:
            df = df.rename(columns={"gene": "gene_name"})
        df = df[["gene_name", "score"]].copy()
        df["gene_name"] = df["gene_name"].astype(str)
        df["score"] = numeric(df["score"])
        df = df.groupby("gene_name", as_index=False)["score"].max()
        scores[method] = df
    return scores


def potential_t2d_gene_union(method_scores: dict[str, pd.DataFrame], all_genes: set[str]) -> tuple[set[str], pd.DataFrame]:
    pos = set(pd.read_csv(POS, usecols=["gene_name"])["gene_name"].astype(str))
    union = set(pos)
    sources = [{"gene_name": g, "source": "DrugBank_positive"} for g in pos]

    if CANDIDATE_TABLE.exists():
        cand = pd.read_csv(CANDIDATE_TABLE)
        gene_col = "gene_name" if "gene_name" in cand.columns else cand.columns[0]
        for g in cand[gene_col].astype(str):
            union.add(g)
            sources.append({"gene_name": g, "source": "TADE_candidate_table"})

    # Conservative disease-evidence union: any method score above its 90th
    # percentile among genes with a positive score is treated as potential T2D evidence.
    # T2DKP evidence is treated more strictly: any positive CVBF/RVBF score is excluded.
    for method, df in method_scores.items():
        df = df[df["gene_name"].isin(all_genes)].copy()
        positive = df[df["score"] > 0]
        if positive.empty:
            continue
        threshold = 0.0 if method.startswith("T2DKP") else float(positive["score"].quantile(0.90))
        hit = positive[positive["score"] >= threshold]
        for g in hit["gene_name"].astype(str):
            union.add(g)
            sources.append(
                {
                    "gene_name": g,
                    "source": f"{method}_potential_T2D_score_ge_{threshold:.6g}",
                }
            )
    source_df = pd.DataFrame(sources).drop_duplicates()
    return union, source_df


def build_candidate_pool() -> tuple[pd.DataFrame, pd.DataFrame]:
    all_omics_genes = set(pd.read_csv(ALL_OMICS, usecols=["gene_name"])["gene_name"].astype(str))
    method_scores = load_method_scores()
    potential_union, union_sources = potential_t2d_gene_union(method_scores, all_omics_genes)

    open_genes = pd.read_csv(OPEN_GENES, sep="\t").rename(columns={"Gene": "gene_name"})
    open_genes["gene_name"] = open_genes["gene_name"].astype(str)
    open_genes["Score_num"] = numeric(open_genes["Score"])
    for col in [
        "hasLigand",
        "hasSmallMoleculeBinder",
        "hasPocket",
        "hasHighQualityChemicalProbes",
        "isInMembrane",
        "isSecreted",
        "chembl",
        "maxClinicalTrialPhase",
        "europepmc",
    ]:
        open_genes[f"{col}_num"] = numeric(open_genes[col])

    desc = pd.read_csv(ALL_DESC)
    desc["gene_name"] = desc["gene_name"].astype(str)
    desc["gene_desc"] = desc["gene_desc"].fillna("")
    desc["desc_word_len"] = desc["gene_desc"].str.split().str.len()
    desc["diabetes_keyword_count"] = desc["gene_desc"].map(count_terms)

    df = open_genes.merge(desc[["gene_name", "desc_word_len", "diabetes_keyword_count"]], on="gene_name", how="left")
    for method, score_df in method_scores.items():
        df = df.merge(score_df.rename(columns={"score": f"{method}_score"}), on="gene_name", how="left")
        df[f"{method}_score"] = df[f"{method}_score"].fillna(0.0)

    drug_flags = [
        "hasLigand_num",
        "hasSmallMoleculeBinder_num",
        "hasPocket_num",
        "hasHighQualityChemicalProbes_num",
    ]
    df["druggability_flags"] = (df[drug_flags] > 0).sum(axis=1)
    df["druggability_flags"] += (df["chembl_num"] > 0).astype(int)
    df["druggability_flags"] += (df["maxClinicalTrialPhase_num"] > 0).astype(int)
    df["targetability_flags"] = df["druggability_flags"]
    df["targetability_flags"] += (df["isInMembrane_num"] > 0).astype(int)
    df["targetability_flags"] += (df["isSecreted_num"] > 0).astype(int)
    df["in_potential_t2d_union"] = df["gene_name"].isin(potential_union)

    pool = df[
        df["gene_name"].isin(all_omics_genes)
        & ~df["in_potential_t2d_union"]
        & (df["Score_num"] <= 0.0217)
        & (df["diabetes_keyword_count"] <= 10)
        & (df["druggability_flags"] >= 1)
    ].copy()
    return pool, union_sources


def select_negatives(pool: pd.DataFrame) -> pd.DataFrame:
    selected = pool.sort_values(
        ["druggability_flags", "europepmc_num", "targetability_flags", "Score_num"],
        ascending=[False, False, False, True],
    ).head(27)
    if len(selected) < 27:
        raise RuntimeError(f"Only {len(selected)} curated no-T2D negatives available")
    return selected


def top10_metrics(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    k = max(1, int(np.ceil(0.10 * len(y_true))))
    idx = np.argsort(scores)[::-1][:k]
    precision = float(np.mean(y_true[idx]))
    prevalence = float(np.mean(y_true))
    return precision, precision / prevalence if prevalence else np.nan


def eval_scores(dataset: str, method: str, y: np.ndarray, scores: np.ndarray) -> dict:
    top_p, top_lift = top10_metrics(y, scores)
    return {
        "dataset": dataset,
        "method": method,
        "AUROC": roc_auc_score(y, scores),
        "AUPRC": average_precision_score(y, scores),
        "top10_precision": top_p,
        "top10_lift": top_lift,
        "n_positive": int(y.sum()),
        "n_negative": int((1 - y).sum()),
    }


def tade_models() -> dict[str, object]:
    return {
        "TADE_full_logreg": Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(class_weight="balanced", solver="liblinear", random_state=SEED, max_iter=5000)),
            ]
        ),
        "TADE_full_extra_trees": ExtraTreesClassifier(n_estimators=150, min_samples_leaf=2, class_weight="balanced", random_state=SEED, n_jobs=-1),
        "TADE_full_gradient_boosting": GradientBoostingClassifier(n_estimators=120, learning_rate=0.03, max_depth=2, random_state=SEED),
    }


def cv_tade(data: pd.DataFrame, dataset: str) -> pd.DataFrame:
    feature_cols = [c for c in data.columns if c not in {"gene_name", "label"}]
    X = data[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)
    y = data["label"].to_numpy()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = []
    for method, model in tade_models().items():
        fold_rows = []
        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
            estimator = clone(model)
            estimator.fit(X[train_idx], y[train_idx])
            scores = estimator.predict_proba(X[test_idx])[:, 1]
            r = eval_scores(dataset, method, y[test_idx], scores)
            r["fold"] = fold
            fold_rows.append(r)
        fd = pd.DataFrame(fold_rows)
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "AUROC": fd["AUROC"].mean(),
                "AUROC_sd": fd["AUROC"].std(),
                "AUPRC": fd["AUPRC"].mean(),
                "AUPRC_sd": fd["AUPRC"].std(),
                "top10_precision": fd["top10_precision"].mean(),
                "top10_lift": fd["top10_lift"].mean(),
                "n_positive": int(y.sum()),
                "n_negative": int((1 - y).sum()),
                "evaluation_type": "5fold_cv_on_curated_set",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pool, union_sources = build_candidate_pool()
    union_sources.to_csv(OUT / "potential_t2d_gene_union_sources.csv", index=False)
    pool.to_csv(OUT / "curated_no_t2d_targetlike_candidate_pool.csv", index=False)
    selected = select_negatives(pool)
    selected.to_csv(OUT / "curated_no_t2d_targetlike_negative_27_manifest.csv", index=False)

    all_omics = pd.read_csv(ALL_OMICS).set_index("gene_name")
    neg_matrix = all_omics.loc[selected["gene_name"].astype(str).tolist()].copy()
    neg_matrix.to_csv(OUT / "curated_no_t2d_targetlike_negative_27_omics_text.csv")

    pos = pd.read_csv(POS).assign(label=1)
    neg = pd.read_csv(OUT / "curated_no_t2d_targetlike_negative_27_omics_text.csv").assign(label=0)
    data = pd.concat([pos, neg], ignore_index=True)
    y = data["label"].to_numpy()

    # External/static comparison methods on exactly the same genes.
    method_scores = load_method_scores()
    compare_rows = []
    genes = data["gene_name"].astype(str).tolist()
    for method, score_df in method_scores.items():
        score_map = dict(zip(score_df["gene_name"].astype(str), score_df["score"].astype(float)))
        scores = np.array([score_map.get(g, 0.0) for g in genes], dtype=float)
        row = eval_scores("curated_no_t2d_targetlike_27", method, y, scores)
        row["AUROC_sd"] = np.nan
        row["AUPRC_sd"] = np.nan
        row["evaluation_type"] = "fixed_external_gene_score"
        compare_rows.append(row)

    tade_summary = cv_tade(data, "curated_no_t2d_targetlike_27")
    external_summary = pd.DataFrame(compare_rows)
    summary = pd.concat([tade_summary, external_summary], ignore_index=True, sort=False)
    summary["date"] = date.today().isoformat()
    summary["negative_selection"] = (
        "Open Targets target-like/druggable genes minus potential T2D gene union from "
        "DrugBank positives, candidate table, CTD, DISEASES, Know-GENE, ProphNet, "
        "GUILDify, TIGA, Geneshot, PubMed-Score, T2DKP-CVBF and T2DKP-RVBF; "
        "then require Open Targets T2D Score<=0.0217, diabetes_keyword_count<=10, "
        "and druggability_flags>=1."
    )
    summary = summary.sort_values("AUPRC", ascending=False)
    summary.to_csv(OUT / "curated_no_t2d_hard_negative_method_comparison.csv", index=False)

    audit = pd.DataFrame(
        [
            {
                "negative_set": "curated_no_t2d_targetlike_27",
                "n": len(selected),
                "candidate_pool_n": len(pool),
                "potential_t2d_union_n": union_sources["gene_name"].nunique(),
                "overlap_with_positives": 0,
                "Score_mean": selected["Score_num"].mean(),
                "Score_max": selected["Score_num"].max(),
                "diabetes_keyword_count_mean": selected["diabetes_keyword_count"].mean(),
                "druggability_flags_mean": selected["druggability_flags"].mean(),
                "europepmc_mean": selected["europepmc_num"].mean(),
            }
        ]
    )
    audit.to_csv(OUT / "curated_no_t2d_hard_negative_audit.csv", index=False)
    print(summary.to_string(index=False))
    print("\nAUDIT")
    print(audit.to_string(index=False))


if __name__ == "__main__":
    main()
