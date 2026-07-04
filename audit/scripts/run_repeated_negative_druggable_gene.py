import os
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.stats import mannwhitneyu
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MaxAbsScaler
from torch.utils.data import DataLoader, TensorDataset

import sys

PROJECT_DIR = Path("/root/autodl-tmp/TADE/code/train_val_test_draw")
sys.path.insert(0, str(PROJECT_DIR))
from models.TADE_GENE import GenePredictor  # noqa: E402


def seed_torch(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def run_cur_druggable_gene(pos_file: Path, neg_file: Path, k: int = 5) -> dict:
    df_pos = pd.read_csv(pos_file, index_col=0)
    df_neg = pd.read_csv(neg_file, index_col=0)
    df = pd.concat([df_pos, df_neg], axis=0)
    data = df.values.astype(float)
    labels = np.concatenate([np.ones(len(df_pos)), np.zeros(len(df_neg))])

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=110)
    mean_fpr = np.linspace(0, 1, 100)
    tprs, aucs, prs, p_values = [], [], [], []
    thresholds_out = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for fold, (train_index, test_index) in enumerate(skf.split(data, labels)):
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        scaler = MaxAbsScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float),
            torch.tensor(y_train, dtype=torch.float).reshape(-1, 1),
        )
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

        seed_torch(0)
        model = GenePredictor(
            gene_dim=321,
            text_dim=768,
            dim=256,
            depth=3,
            heads=8,
            attn_dropout=0.1,
            ff_dropout=0.1,
            k_sum=8,
            k_prod=8,
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        for _ in range(110):
            model.train()
            for batch_data, batch_labels in train_loader:
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)
                outputs, loss = model(batch_data, batch_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test, dtype=torch.float).to(device)
            y_test_tensor = torch.tensor(y_test, dtype=torch.float).reshape(-1, 1).to(device)
            probas, _ = model(X_test_tensor, y_test_tensor)
            probas = probas.cpu().squeeze().numpy()

        pos_scores = probas[y_test == 1]
        neg_scores = probas[y_test == 0]
        _, p_value = mannwhitneyu(pos_scores, neg_scores, alternative="greater")
        p_values.append(p_value)

        precision, recall, thresholds = precision_recall_curve(y_test, probas)
        thresholds = np.append(thresholds, 1.0)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        thresholds_out.append(float(thresholds[np.argmax(f1_scores)]))

        fpr, tpr, _ = roc_curve(y_test, probas)
        fold_auc = auc(fpr, tpr)
        fold_pr = average_precision_score(y_test, probas)

        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        aucs.append(fold_auc)
        prs.append(fold_pr)
        print(
            f"{neg_file.name} fold={fold} auc={fold_auc:.6f} auprc={fold_pr:.6f} p={p_value:.3e}",
            flush=True,
        )

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    return {
        "neg_file": neg_file.name,
        "curve_mean_auc": float(auc(mean_fpr, mean_tpr)),
        "fold_mean_auc": float(np.mean(aucs)),
        "fold_std_auc": float(np.std(aucs, ddof=1)),
        "mean_auprc": float(np.mean(prs)),
        "std_auprc": float(np.std(prs, ddof=1)),
        "best_f1_threshold_mean": float(np.mean(thresholds_out)),
        "mannwhitney_p_min": float(np.min(p_values)),
        "pos_n": int(len(df_pos)),
        "neg_n": int(len(df_neg)),
    }


def main() -> None:
    seed_torch(0)
    base = Path("/root/autodl-tmp/TADE/datasets/druggable_gene/drugbank")
    pos_file = base / "pos_omics_text.csv"
    neg_files = [
        base / "neg_omics_text_random_10_extra_0.csv",
        base / "neg_omics_text_random_10_extra_1.csv",
        base / "neg_omics_text_random_10_extra_2.csv",
        base / "neg_omics_text_random_10_extra_3.csv",
        base / "neg_omics_text_random_10.csv",
    ]

    rows = [run_cur_druggable_gene(pos_file, neg_file) for neg_file in neg_files]
    df = pd.DataFrame(rows)
    summary = pd.DataFrame(
        [
            {
                "method": "TADE_rerun",
                "AUC": df["curve_mean_auc"].mean(),
                "AUC_std_across_negative_sets": df["curve_mean_auc"].std(ddof=1),
                "AUPRC": df["mean_auprc"].mean(),
                "AUPRC_std_across_negative_sets": df["mean_auprc"].std(ddof=1),
            }
        ]
    )

    out_dir = Path("/root/autodl-tmp/yyfuxian")
    df.to_csv(out_dir / "repeated_negative_druggable_gene_rerun_per_set.csv", index=False)
    summary.to_csv(out_dir / "repeated_negative_druggable_gene_rerun_summary.csv", index=False)
    print("\nPer-set results:")
    print(df.to_string(index=False))
    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
