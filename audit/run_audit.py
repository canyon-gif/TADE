from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


ROOT = Path("/root/autodl-tmp")
TADE = ROOT / "TADE"
YY = ROOT / "yyfuxian"
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)


def read_csv_any(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, **kwargs)


def summarize_matrix() -> dict:
    all_omics = YY / "zip_check_TADE/TADE/data_source/all_omics_text.csv"
    all_desc = YY / "zip_check_TADE/TADE/data_source/all_gene_desc.csv"
    df = read_csv_any(all_omics)
    desc = read_csv_any(all_desc)

    feature_cols = [c for c in df.columns if c != "gene_name"]
    omics_cols = feature_cols[:321]
    text_cols = feature_cols[321:]
    snp_pvalue_mean = [c for c in omics_cols if c.startswith("snp_pvalue_mean_")]
    snp_pvalue_var = [c for c in omics_cols if c.startswith("snp_pvalue_var_")]
    snp_cnt = [c for c in omics_cols if c.startswith("snp_cnt_")]
    snp_cadd_mean = [c for c in omics_cols if c.startswith("snp_cadd_mean_")]
    snp_cadd_var = [c for c in omics_cols if c.startswith("snp_cadd_var_")]
    desc_vectors = [c for c in text_cols if c.startswith("desc_vector_")]

    text_examples = desc.head(12).copy()
    text_examples.to_csv(OUT / "gpt_cached_description_examples_first12.csv", index=False)

    colspec = pd.DataFrame(
        {
            "block": [
                "snp_pvalue_mean",
                "snp_pvalue_var",
                "snp_cnt",
                "snp_cadd_mean",
                "snp_cadd_var",
                "3mer_or_nucleotide_context",
                "methylation",
                "expression",
                "bert_desc_vector",
            ],
            "count": [
                len(snp_pvalue_mean),
                len(snp_pvalue_var),
                len(snp_cnt),
                len(snp_cadd_mean),
                len(snp_cadd_var),
                32,
                2,
                2,
                len(desc_vectors),
            ],
        }
    )
    colspec.to_csv(OUT / "omics_feature_block_counts.csv", index=False)

    return {
        "all_omics_rows": len(df),
        "all_omics_cols": len(df.columns),
        "feature_cols": len(feature_cols),
        "omics_cols": len(omics_cols),
        "text_cols": len(text_cols),
        "all_omics_missing_values": int(df.isna().sum().sum()),
        "all_desc_rows": len(desc),
        "all_desc_cols": len(desc.columns),
        "all_desc_missing_values": int(desc.isna().sum().sum()),
        "desc_nonempty_rows": int(desc.astype(str).apply(lambda r: r.str.len().sum(), axis=1).gt(0).sum()),
        "desc_vector_cols": len(desc_vectors),
    }


def summarize_training_sets() -> dict:
    dg_base = TADE / "datasets/druggable_gene/drugbank"
    gdi = TADE / "datasets/gene_drug_interaction/drugbank/test.csv"
    pos = read_csv_any(dg_base / "pos_omics_text.csv", index_col=0)
    neg_files = sorted(dg_base.glob("neg_omics_text_random_10*.csv"))
    neg_summary = []
    pos_genes = set(pos.index.astype(str))
    for path in neg_files:
        neg = read_csv_any(path, index_col=0)
        neg_genes = set(neg.index.astype(str))
        neg_summary.append(
            {
                "file": path.name,
                "rows": len(neg),
                "cols": len(neg.columns),
                "unique_genes": len(neg_genes),
                "overlap_with_positive": len(pos_genes & neg_genes),
                "missing_values": int(neg.isna().sum().sum()),
            }
        )
    pd.DataFrame(neg_summary).to_csv(OUT / "negative_set_integrity_summary.csv", index=False)

    pair = read_csv_any(gdi)
    return {
        "druggable_pos_rows": len(pos),
        "druggable_pos_cols": len(pos.columns),
        "druggable_pos_missing": int(pos.isna().sum().sum()),
        "negative_sets": len(neg_summary),
        "gdi_rows": len(pair),
        "gdi_cols": len(pair.columns),
        "gdi_positive_pairs": int(pair["label"].sum()),
        "gdi_negative_pairs": int((pair["label"] == 0).sum()),
        "gdi_unique_genes": int(pair["gene_name"].nunique()),
        "gdi_unique_drugs": int(pair["drug_name"].nunique()),
        "gdi_missing_values": int(pair.isna().sum().sum()),
    }


def audit_original_gdi_stratified_overlap() -> dict:
    df = read_csv_any(TADE / "datasets/gene_drug_interaction/drugbank/test.csv")
    labels = df["label"].values
    skf = StratifiedKFold(n_splits=5, shuffle=False)
    rows = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(labels)), labels), 1):
        train = df.iloc[train_idx]
        test = df.iloc[test_idx]
        train_genes = set(train["gene_name"])
        test_genes = set(test["gene_name"])
        train_drugs = set(train["drug_name"])
        test_drugs = set(test["drug_name"])
        rows.append(
            {
                "fold": fold,
                "test_n": len(test),
                "test_pos": int(test["label"].sum()),
                "test_unique_genes": len(test_genes),
                "test_genes_seen_in_train": len(test_genes & train_genes),
                "test_gene_seen_fraction": len(test_genes & train_genes) / len(test_genes),
                "test_unique_drugs": len(test_drugs),
                "test_drugs_seen_in_train": len(test_drugs & train_drugs),
                "test_drug_seen_fraction": len(test_drugs & train_drugs) / len(test_drugs),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "gdi_stratified_split_overlap_audit.csv", index=False)
    return {
        "orig_gdi_gene_seen_fraction_mean": float(out["test_gene_seen_fraction"].mean()),
        "orig_gdi_gene_seen_fraction_min": float(out["test_gene_seen_fraction"].min()),
        "orig_gdi_gene_seen_fraction_max": float(out["test_gene_seen_fraction"].max()),
        "orig_gdi_drug_seen_fraction_mean": float(out["test_drug_seen_fraction"].mean()),
        "orig_gdi_drug_seen_fraction_min": float(out["test_drug_seen_fraction"].min()),
        "orig_gdi_drug_seen_fraction_max": float(out["test_drug_seen_fraction"].max()),
    }


def summarize_prediction_shift() -> dict:
    gw = read_csv_any(TADE / "code/train_val_test_draw/save/ablation/genome_wide_correction.csv")
    temporal = read_csv_any(TADE / "code/train_val_test_draw/save/ablation/temporal_gene_correction.csv")

    def find_shift_col(df):
        for c in ["correction", "prediction_shift", "shift", "delta"]:
            if c in df.columns:
                return c
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric) >= 2:
            return numeric[-1]
        raise ValueError("No shift-like numeric column found")

    gw_col = find_shift_col(gw)
    tmp_col = find_shift_col(temporal)
    rows = []
    for name, df, col in [("genome_wide", gw, gw_col), ("temporal", temporal, tmp_col)]:
        shift = df[col].astype(float)
        substantial = shift.abs() > 0.1
        rows.append(
            {
                "analysis": name,
                "shift_column": col,
                "n": len(df),
                "positive_shift_n": int((shift > 0).sum()),
                "positive_shift_fraction": float((shift > 0).mean()),
                "negative_shift_n": int((shift < 0).sum()),
                "mean_shift": float(shift.mean()),
                "mean_abs_shift": float(shift.abs().mean()),
                "substantial_abs_gt_0_1_n": int(substantial.sum()),
                "substantial_up_n": int(((shift > 0) & substantial).sum()),
                "substantial_up_fraction": float(((shift > 0) & substantial).sum() / substantial.sum()) if substantial.sum() else np.nan,
            }
        )
    pd.DataFrame(rows).to_csv(OUT / "prediction_shift_recomputed_summary.csv", index=False)
    return {f"{r['analysis']}_{k}": v for r in rows for k, v in r.items() if k != "analysis"}


def summarize_candidate_genes() -> dict:
    path = TADE / "results/candidate_genes/Supplementary_Table_3_candidate_genes.csv"
    df = read_csv_any(path)
    gene_col = "gene_name" if "gene_name" in df.columns else df.columns[0]
    dup = df[df[gene_col].duplicated(keep=False)].copy()
    dup.to_csv(OUT / "candidate_gene_duplicate_rows.csv", index=False)
    return {
        "candidate_table_rows": len(df),
        "candidate_unique_genes": int(df[gene_col].nunique()),
        "candidate_duplicate_gene_rows": len(dup),
        "candidate_duplicate_gene_names": ",".join(sorted(dup[gene_col].astype(str).unique())),
    }


def include_completed_results() -> dict:
    rep = read_csv_any(YY / "repeated_negative_druggable_gene_rerun_summary.csv")
    gdi = read_csv_any(YY / "gdi_group_cold_summary.csv")
    return {
        "repeated_negative_auc": float(rep.loc[0, "AUC"]),
        "repeated_negative_auc_sd": float(rep.loc[0, "AUC_std_across_negative_sets"]),
        "repeated_negative_auprc": float(rep.loc[0, "AUPRC"]),
        "repeated_negative_auprc_sd": float(rep.loc[0, "AUPRC_std_across_negative_sets"]),
        "gdi_gene_cold_auroc": float(gdi[gdi["split"] == "gene_name"]["AUROC_mean"].iloc[0]),
        "gdi_gene_cold_auprc": float(gdi[gdi["split"] == "gene_name"]["AUPRC_mean"].iloc[0]),
        "gdi_drug_cold_auroc": float(gdi[gdi["split"] == "drug_name"]["AUROC_mean"].iloc[0]),
        "gdi_drug_cold_auprc": float(gdi[gdi["split"] == "drug_name"]["AUPRC_mean"].iloc[0]),
    }


def main():
    summary = {}
    for fn in [
        summarize_matrix,
        summarize_training_sets,
        audit_original_gdi_stratified_overlap,
        summarize_prediction_shift,
        summarize_candidate_genes,
        include_completed_results,
    ]:
        summary.update(fn())
    pd.DataFrame([summary]).to_csv(OUT / "audit_summary.csv", index=False)
    for key in sorted(summary):
        print(f"{key}: {summary[key]}")


if __name__ == "__main__":
    main()
