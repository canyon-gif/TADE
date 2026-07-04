import os
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import auc, average_precision_score, roc_curve
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import MaxAbsScaler
import torch


PROJECT_DIR = Path("/root/autodl-tmp/TADE/code/train_val_test_draw")
SOURCE = PROJECT_DIR / "gene_drug_interaction.py"


def load_original_definitions():
    text = SOURCE.read_text()
    marker = "fg2emb = load('./save/gene_drug_interaction/fg2emb.pkl')"
    prefix = text.split(marker)[0]
    namespace = {"__file__": str(SOURCE)}
    old_cwd = os.getcwd()
    os.chdir(PROJECT_DIR)
    try:
        exec(compile(prefix, str(SOURCE), "exec"), namespace)
        namespace["fg2emb"] = load("./save/gene_drug_interaction/fg2emb.pkl")
    finally:
        os.chdir(old_cwd)
    return namespace


def run_group_cv(group_col: str, epochs: int = 18, k: int = 5) -> tuple[pd.DataFrame, dict]:
    ns = load_original_definitions()
    df = pd.read_csv("/root/autodl-tmp/TADE/datasets/gene_drug_interaction/drugbank/test.csv")
    genomic_data = df.iloc[:, 2:-2].values.astype(float)
    labels = df["label"].values
    groups = df[group_col].values

    gkf = GroupKFold(n_splits=k)
    rows = []
    for fold, (train_index, test_index) in enumerate(gkf.split(genomic_data, labels, groups), 1):
        df_train = df.iloc[train_index].copy()
        df_test = df.iloc[test_index].copy()

        scaler = MaxAbsScaler().fit(genomic_data[train_index])
        train_dataset = CachedDrugGeneDataset(df_train, scaler, ns)
        test_dataset = CachedDrugGeneDataset(df_test, scaler, ns)

        train_loader = ns["GraphDataLoader"](
            train_dataset, batch_size=32, shuffle=False, collate_fn=cached_collate
        )
        test_loader = ns["GraphDataLoader"](
            test_dataset, batch_size=32, shuffle=False, collate_fn=cached_collate
        )

        ns["seed_torch"](ns["seed"])
        model = ns["DrugGenePredictor"](ns["in_feats"], ns["hidden_feats"], ns["num_heads"], ns["genomic_feats"]).to(
            ns["device"]
        )
        optimizer = ns["torch"].optim.AdamW(model.parameters(), lr=6e-4)
        loss_fn = ns["nn"].BCELoss()

        for _ in range(epochs):
            ns["train_model"](model, train_loader, optimizer, loss_fn, ns["device"])

        all_labels, all_probas = ns["test_model"](model, test_loader, ns["device"])
        fpr, tpr, _ = roc_curve(all_labels, all_probas)
        auc_score = auc(fpr, tpr)
        auprc = average_precision_score(all_labels, all_probas)

        train_groups = set(df_train[group_col])
        test_groups = set(df_test[group_col])
        row = {
            "split": group_col,
            "fold": fold,
            "train_n": len(df_train),
            "test_n": len(df_test),
            "train_pos": int(df_train["label"].sum()),
            "test_pos": int(df_test["label"].sum()),
            "train_neg": int((df_train["label"] == 0).sum()),
            "test_neg": int((df_test["label"] == 0).sum()),
            "test_unique_groups": len(test_groups),
            "overlap_groups": len(train_groups & test_groups),
            "AUROC": float(auc_score),
            "AUPRC": float(auprc),
        }
        rows.append(row)
        print(
            f"{group_col} fold={fold} test_pos={row['test_pos']} auc={auc_score:.6f} auprc={auprc:.6f}",
            flush=True,
        )

    per_fold = pd.DataFrame(rows)
    summary = {
        "split": group_col,
        "AUROC_mean": float(per_fold["AUROC"].mean()),
        "AUROC_std": float(per_fold["AUROC"].std(ddof=1)),
        "AUPRC_mean": float(per_fold["AUPRC"].mean()),
        "AUPRC_std": float(per_fold["AUPRC"].std(ddof=1)),
        "folds": int(len(per_fold)),
    }
    return per_fold, summary


class CachedDrugGeneDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, scaler: MaxAbsScaler, ns: dict):
        self.items = []
        atom_featurizer = ns["CanonicalAtomFeaturizer"]()
        fg_keys = list(ns["fg2emb"].keys())
        for _, row in df.iterrows():
            smiles = row["canonical_smi"]
            feats = row.iloc[2:-2].astype(float).values
            feats = scaler.transform(feats.reshape(1, -1)).ravel()
            graph = ns["smiles_to_bigraph"](
                smiles,
                canonical_atom_order=False,
                node_featurizer=atom_featurizer,
            )
            mol = ns["Chem"].MolFromSmiles(smiles)
            fg_indices = [0]
            if mol is not None:
                for sm in ns["smart"]:
                    if mol.HasSubstructMatch(sm):
                        fg_name = ns["smart2name"][sm]
                        if fg_name in ns["fg2emb"]:
                            fg_indices.append(fg_keys.index(fg_name))
            fg_indices = fg_indices[:13] + [0] * max(0, 13 - len(fg_indices))
            self.items.append(
                (
                    graph,
                    torch.tensor(feats, dtype=torch.float),
                    torch.tensor(row["label"], dtype=torch.float),
                    torch.tensor(fg_indices, dtype=torch.long),
                )
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def cached_collate(batch):
    graphs, genomic_feats, labels, fg_indices = zip(*batch)
    import dgl

    return (
        dgl.batch(graphs),
        torch.stack(genomic_feats),
        torch.stack(labels),
        torch.stack(fg_indices),
    )


def main():
    out_dir = Path("/root/autodl-tmp/yyfuxian")
    all_folds = []
    summaries = []
    for group_col in ["gene_name", "drug_name"]:
        per_fold, summary = run_group_cv(group_col)
        all_folds.append(per_fold)
        summaries.append(summary)

    fold_df = pd.concat(all_folds, ignore_index=True)
    summary_df = pd.DataFrame(summaries)
    fold_df.to_csv(out_dir / "gdi_group_cold_per_fold.csv", index=False)
    summary_df.to_csv(out_dir / "gdi_group_cold_summary.csv", index=False)
    print("\nPer-fold:")
    print(fold_df.to_string(index=False))
    print("\nSummary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
