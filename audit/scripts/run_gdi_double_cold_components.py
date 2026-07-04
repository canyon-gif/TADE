import os
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from joblib import load
from sklearn.metrics import auc, average_precision_score, roc_curve
from sklearn.preprocessing import MaxAbsScaler


PROJECT_DIR = Path("/root/autodl-tmp/TADE/code/train_val_test_draw")
SOURCE = PROJECT_DIR / "gene_drug_interaction.py"
OUT = Path("/root/autodl-tmp/尤毅复现/remaining_experiment_outputs")
OUT.mkdir(exist_ok=True)


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


def add_component_ids(df: pd.DataFrame) -> pd.DataFrame:
    graph = defaultdict(set)
    for idx, row in df.iterrows():
        g = f"gene::{row['gene_name']}"
        d = f"drug::{row['drug_name']}"
        p = f"pair::{idx}"
        graph[g].add(p)
        graph[d].add(p)
        graph[p].update([g, d])

    seen = set()
    pair_to_comp = {}
    comp_id = 0
    for node in list(graph):
        if node in seen:
            continue
        comp_id += 1
        q = deque([node])
        seen.add(node)
        while q:
            cur = q.popleft()
            if cur.startswith("pair::"):
                pair_to_comp[int(cur.split("::", 1)[1])] = comp_id
            for nxt in graph[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    q.append(nxt)
    out = df.copy()
    out["double_cold_component"] = [pair_to_comp[i] for i in df.index]
    return out


def assign_component_folds(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    comp = (
        df.groupby("double_cold_component")
        .agg(pairs=("label", "size"), pos=("label", "sum"))
        .reset_index()
        .sort_values(["pos", "pairs"], ascending=False)
    )
    fold_pos = [0] * k
    fold_pairs = [0] * k
    comp_to_fold = {}
    for _, row in comp.iterrows():
        if row["pos"] > 0:
            fold = min(range(k), key=lambda i: (fold_pos[i], fold_pairs[i]))
        else:
            fold = min(range(k), key=lambda i: fold_pairs[i])
        comp_to_fold[int(row["double_cold_component"])] = fold + 1
        fold_pos[fold] += int(row["pos"])
        fold_pairs[fold] += int(row["pairs"])
    out = df.copy()
    out["double_cold_fold"] = out["double_cold_component"].map(comp_to_fold)
    comp.to_csv(OUT / "gdi_double_cold_component_distribution.csv", index=False)
    out[["gene_name", "drug_name", "label", "double_cold_component", "double_cold_fold"]].to_csv(
        OUT / "gdi_double_cold_fold_assignment.csv", index=False
    )
    return out


class CachedDrugGeneDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, scaler: MaxAbsScaler, ns: dict):
        self.items = []
        atom_featurizer = ns["CanonicalAtomFeaturizer"]()
        fg_keys = list(ns["fg2emb"].keys())
        for _, row in df.iterrows():
            smiles = row["canonical_smi"]
            feats = row.iloc[2:-4].astype(float).values
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


def run_double_cold(epochs: int = 18, k: int = 5):
    ns = load_original_definitions()
    df = pd.read_csv("/root/autodl-tmp/TADE/datasets/gene_drug_interaction/drugbank/test.csv")
    df = assign_component_folds(add_component_ids(df), k=k)
    feature_cols = df.columns[2:-4]
    genomic_data = df[feature_cols].values.astype(float)
    rows = []

    for fold in range(1, k + 1):
        train_index = df.index[df["double_cold_fold"] != fold].to_numpy()
        test_index = df.index[df["double_cold_fold"] == fold].to_numpy()
        df_train = df.loc[train_index].copy()
        df_test = df.loc[test_index].copy()

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
        train_genes = set(df_train["gene_name"])
        test_genes = set(df_test["gene_name"])
        train_drugs = set(df_train["drug_name"])
        test_drugs = set(df_test["drug_name"])
        row = {
            "split": "component_double_cold",
            "fold": fold,
            "train_n": len(df_train),
            "test_n": len(df_test),
            "train_pos": int(df_train["label"].sum()),
            "test_pos": int(df_test["label"].sum()),
            "train_neg": int((df_train["label"] == 0).sum()),
            "test_neg": int((df_test["label"] == 0).sum()),
            "gene_overlap": len(train_genes & test_genes),
            "drug_overlap": len(train_drugs & test_drugs),
            "component_overlap": len(set(df_train["double_cold_component"]) & set(df_test["double_cold_component"])),
            "AUROC": float(auc(fpr, tpr)),
            "AUPRC": float(average_precision_score(all_labels, all_probas)),
        }
        rows.append(row)
        print(
            f"double-cold fold={fold} test_pos={row['test_pos']} auc={row['AUROC']:.6f} auprc={row['AUPRC']:.6f}",
            flush=True,
        )

    per_fold = pd.DataFrame(rows)
    summary = pd.DataFrame(
        [
            {
                "split": "component_double_cold",
                "AUROC_mean": per_fold["AUROC"].mean(),
                "AUROC_std": per_fold["AUROC"].std(ddof=1),
                "AUPRC_mean": per_fold["AUPRC"].mean(),
                "AUPRC_std": per_fold["AUPRC"].std(ddof=1),
                "folds": len(per_fold),
                "max_gene_overlap": int(per_fold["gene_overlap"].max()),
                "max_drug_overlap": int(per_fold["drug_overlap"].max()),
                "max_component_overlap": int(per_fold["component_overlap"].max()),
            }
        ]
    )
    per_fold.to_csv(OUT / "gdi_double_cold_component_per_fold.csv", index=False)
    summary.to_csv(OUT / "gdi_double_cold_component_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    run_double_cold()
