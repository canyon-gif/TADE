from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from scipy import stats


OUT = Path("/root/autodl-tmp/尤毅复现/remaining_experiment_outputs")


def compute_midrank(x):
    order = np.argsort(x)
    sorted_x = x[order]
    n = len(x)
    midranks = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        midranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n, dtype=float)
    out[order] = midranks
    return out


def delong_auc_ci(y_true, scores, alpha=0.95):
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    m = len(pos)
    n = len(neg)
    if m < 2 or n < 2:
        return np.nan, np.nan, np.nan
    all_scores = np.concatenate([pos, neg])
    tx = compute_midrank(pos)
    ty = compute_midrank(neg)
    tz = compute_midrank(all_scores)
    auc = (tz[:m].sum() - m * (m + 1) / 2) / (m * n)
    v01 = (tz[:m] - tx) / n
    v10 = 1 - (tz[m:] - ty) / m
    sx = np.var(v01, ddof=1)
    sy = np.var(v10, ddof=1)
    se = np.sqrt(sx / m + sy / n)
    z = stats.norm.ppf(1 - (1 - alpha) / 2)
    return float(auc), float(max(0, auc - z * se)), float(min(1, auc + z * se))


def stratified_bootstrap(y, score, n_boot=10000, seed=123):
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    score = np.asarray(score).astype(float)
    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)
    auroc = np.empty(n_boot)
    auprc = np.empty(n_boot)
    auprc_lift = np.empty(n_boot)
    top_decile_precision = np.empty(n_boot)
    top_decile_lift = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.concatenate(
            [
                rng.choice(pos_idx, size=len(pos_idx), replace=True),
                rng.choice(neg_idx, size=len(neg_idx), replace=True),
            ]
        )
        yy = y[idx]
        ss = score[idx]
        prevalence = yy.mean()
        ap = average_precision_score(yy, ss)
        auroc[i] = roc_auc_score(yy, ss)
        auprc[i] = ap
        auprc_lift[i] = ap / prevalence
        k = max(1, int(np.ceil(0.1 * len(idx))))
        top = np.argsort(-ss)[:k]
        prec = yy[top].mean()
        top_decile_precision[i] = prec
        top_decile_lift[i] = prec / prevalence
    return {
        "AUROC_boot_low": float(np.percentile(auroc, 2.5)),
        "AUROC_boot_high": float(np.percentile(auroc, 97.5)),
        "AUPRC_boot_low": float(np.percentile(auprc, 2.5)),
        "AUPRC_boot_high": float(np.percentile(auprc, 97.5)),
        "AUPRC_lift_boot_low": float(np.percentile(auprc_lift, 2.5)),
        "AUPRC_lift_boot_high": float(np.percentile(auprc_lift, 97.5)),
        "top_decile_precision_boot_low": float(np.percentile(top_decile_precision, 2.5)),
        "top_decile_precision_boot_high": float(np.percentile(top_decile_precision, 97.5)),
        "top_decile_lift_boot_low": float(np.percentile(top_decile_lift, 2.5)),
        "top_decile_lift_boot_high": float(np.percentile(top_decile_lift, 97.5)),
        "bootstrap_n": n_boot,
    }


def permutation_pvalues(y, score, n_perm=5000, seed=321):
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    score = np.asarray(score).astype(float)
    obs_auc = roc_auc_score(y, score)
    obs_ap = average_precision_score(y, score)
    ge_auc = 0
    ge_ap = 0
    for _ in range(n_perm):
        yp = rng.permutation(y)
        ge_auc += roc_auc_score(yp, score) >= obs_auc
        ge_ap += average_precision_score(yp, score) >= obs_ap
    return {
        "AUROC_permutation_p": (ge_auc + 1) / (n_perm + 1),
        "AUPRC_permutation_p": (ge_ap + 1) / (n_perm + 1),
        "permutation_n": n_perm,
    }


def top_enrichment(y, score):
    y = np.asarray(y).astype(int)
    score = np.asarray(score).astype(float)
    n = len(y)
    positives = int(y.sum())
    prevalence = positives / n
    order = np.argsort(-score)
    rows = {}
    for frac in [0.05, 0.10, 0.20]:
        k = max(1, int(np.ceil(frac * n)))
        hit = int(y[order[:k]].sum())
        precision = hit / k
        rows[f"top_{int(frac*100)}pct_k"] = k
        rows[f"top_{int(frac*100)}pct_hits"] = hit
        rows[f"top_{int(frac*100)}pct_precision"] = precision
        rows[f"top_{int(frac*100)}pct_lift"] = precision / prevalence
        # Hypergeometric one-sided enrichment P(X >= hit)
        rows[f"top_{int(frac*100)}pct_hypergeom_p"] = float(stats.hypergeom.sf(hit - 1, n, positives, k))
    return rows


def summarize_dataset(name, path):
    df = pd.read_csv(path)
    y = df["label"].astype(int).to_numpy()
    score = df["score"].astype(float).to_numpy()
    auc, dl_low, dl_high = delong_auc_ci(y, score)
    ap = average_precision_score(y, score)
    prevalence = y.mean()
    row = {
        "dataset": name,
        "n": len(df),
        "positives": int(y.sum()),
        "negatives": int((y == 0).sum()),
        "prevalence": prevalence,
        "AUROC": auc,
        "AUROC_delong_low": dl_low,
        "AUROC_delong_high": dl_high,
        "AUPRC": ap,
        "AUPRC_lift_over_prevalence": ap / prevalence,
    }
    row.update(stratified_bootstrap(y, score))
    row.update(permutation_pvalues(y, score))
    row.update(top_enrichment(y, score))
    return row


def main():
    files = {
        "ttd_filtered": OUT / "ttd_fully_external_double_cold_predictions.csv",
        "drugcentral_filtered": OUT / "drugcentral_fully_external_double_cold_predictions.csv",
        "combined_filtered": OUT / "combined_fully_external_double_cold_predictions.csv",
    }
    rows = [summarize_dataset(name, path) for name, path in files.items()]
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "external_fully_double_cold_robust_statistics.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
