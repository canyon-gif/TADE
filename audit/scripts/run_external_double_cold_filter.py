from pathlib import Path

import pandas as pd


ROOT = Path("/root/autodl-tmp")
TADE = ROOT / "TADE"
OUT = ROOT / "尤毅复现" / "remaining_experiment_outputs"
OUT.mkdir(exist_ok=True)


def main():
    train = pd.read_csv(TADE / "datasets/gene_drug_interaction/drugbank/test.csv")
    train_genes = set(train["gene_name"].astype(str))
    train_drugs = set(train["drug_name"].astype(str))
    rows = []
    combined_frames = []
    for name in ["ttd", "drugcentral"]:
        df = pd.read_csv(TADE / f"datasets/gene_drug_interaction/{name}/test.csv")
        gene_new = ~df["gene_name"].astype(str).isin(train_genes)
        drug_new = ~df["drug_name"].astype(str).isin(train_drugs)
        filtered = df[gene_new & drug_new].copy()
        filtered.insert(0, "source_dataset", name)
        filtered.to_csv(OUT / f"{name}_fully_external_double_cold_filtered.csv", index=False)
        combined_frames.append(filtered)
        rows.append(
            {
                "dataset": name,
                "original_pairs": len(df),
                "original_pos": int(df["label"].sum()),
                "original_neg": int((df["label"] == 0).sum()),
                "filtered_pairs": len(filtered),
                "filtered_pos": int(filtered["label"].sum()) if len(filtered) else 0,
                "filtered_neg": int((filtered["label"] == 0).sum()) if len(filtered) else 0,
                "filtered_unique_genes": int(filtered["gene_name"].nunique()) if len(filtered) else 0,
                "filtered_unique_drugs": int(filtered["drug_name"].nunique()) if len(filtered) else 0,
            }
        )
    combined = pd.concat(combined_frames, ignore_index=True)
    combined.to_csv(OUT / "external_fully_double_cold_filtered_combined.csv", index=False)
    rows.append(
        {
            "dataset": "combined",
            "original_pairs": None,
            "original_pos": None,
            "original_neg": None,
            "filtered_pairs": len(combined),
            "filtered_pos": int(combined["label"].sum()) if len(combined) else 0,
            "filtered_neg": int((combined["label"] == 0).sum()) if len(combined) else 0,
            "filtered_unique_genes": int(combined["gene_name"].nunique()) if len(combined) else 0,
            "filtered_unique_drugs": int(combined["drug_name"].nunique()) if len(combined) else 0,
        }
    )
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "external_fully_double_cold_filter_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
