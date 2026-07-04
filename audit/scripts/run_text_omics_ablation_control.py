import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import auc, average_precision_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MaxAbsScaler
from torch.utils.data import DataLoader, TensorDataset


PROJECT_DIR = Path("/root/autodl-tmp/TADE/code/train_val_test_draw")
sys.path.insert(0, str(PROJECT_DIR))
from models.ablation import SingleModal  # noqa: E402


OUT = Path("/root/autodl-tmp/尤毅复现/remaining_experiment_outputs")
OUT.mkdir(exist_ok=True)


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


def load_data(mode: str):
    pos = pd.read_csv("/root/autodl-tmp/TADE/datasets/druggable_gene/drugbank/pos_omics_text.csv", index_col=0)
    neg = pd.read_csv("/root/autodl-tmp/TADE/datasets/druggable_gene/drugbank/neg_omics_text_random_10.csv", index_col=0)
    df = pd.concat([pos, neg], axis=0)
    X = df.values.astype(float)
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    if mode == "text_only":
        X = X[:, 321:]
    elif mode == "omics_only":
        X = X[:, :321]
    else:
        raise ValueError(mode)
    return X, y


def run_mode(mode: str) -> pd.DataFrame:
    X, y = load_data(mode)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=110)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    input_dim = 768 if mode == "text_only" else 321
    dim = 256 if mode == "text_only" else 64
    epochs = 110 if mode == "text_only" else 84
    attn_dropout = 0.1 if mode == "text_only" else 0
    ff_dropout = 0.1 if mode == "text_only" else 0

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        scaler = MaxAbsScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        y_train = y[train_idx]
        y_test = y[test_idx]

        train_ds = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1),
        )
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=False)

        seed_torch(0)
        model = SingleModal(
            input_dim=input_dim,
            dim=dim,
            depth=3,
            heads=8,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            k_sum=8,
            k_prod=8,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        for _ in range(epochs):
            model.train()
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                _, loss = model(xb, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            probas, _ = model(
                torch.tensor(X_test, dtype=torch.float32).to(device),
                torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1).to(device),
            )
            probas = probas.cpu().squeeze().numpy()

        fpr, tpr, _ = roc_curve(y_test, probas)
        row = {
            "mode": mode,
            "fold": fold,
            "test_pos": int(y_test.sum()),
            "test_neg": int((y_test == 0).sum()),
            "AUROC": float(auc(fpr, tpr)),
            "AUPRC": float(average_precision_score(y_test, probas)),
        }
        rows.append(row)
        print(f"{mode} fold={fold} auc={row['AUROC']:.6f} auprc={row['AUPRC']:.6f}", flush=True)
    return pd.DataFrame(rows)


def main():
    per_fold = pd.concat([run_mode("text_only"), run_mode("omics_only")], ignore_index=True)
    summary = (
        per_fold.groupby("mode")
        .agg(
            AUROC_mean=("AUROC", "mean"),
            AUROC_std=("AUROC", "std"),
            AUPRC_mean=("AUPRC", "mean"),
            AUPRC_std=("AUPRC", "std"),
            folds=("fold", "count"),
        )
        .reset_index()
    )
    per_fold.to_csv(OUT / "text_omics_ablation_control_per_fold.csv", index=False)
    summary.to_csv(OUT / "text_omics_ablation_control_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
