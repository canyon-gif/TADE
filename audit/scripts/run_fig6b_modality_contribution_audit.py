from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT = Path("/root/autodl-tmp/yyfuxian")
TADE = PROJECT / "zip_check_TADE/TADE"
OUT = PROJECT / "audit_outputs/fig6b_modality_contribution"
OUT.mkdir(parents=True, exist_ok=True)

SUPP1 = PROJECT / "new_submission_check/submission/TADE_NC_Submission/TADE_NC_SupplementaryTable1.xlsx"
SUPP3_CLEAN = PROJECT / "Supplementary_Table_3_candidate_genes_top280_clean.csv"
OPEN_GENES = TADE / "data_source/open_genes/open_genes.tsv"
CANDIDATE_GENES = TADE / "results/candidate_genes/genes.csv"


def percentile_rank_high_is_high(s: pd.Series) -> pd.Series:
    return s.rank(method="average", pct=True)


def read_supplementary_table1() -> pd.DataFrame:
    df = pd.read_excel(SUPP1, skiprows=1)
    expected = {"gene_name", "full_score", "text_score", "gene_score", "delta"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Supplementary Table 1 missing columns: {sorted(missing)}")
    df["gene_name"] = df["gene_name"].astype(str).str.upper()
    return df


def build_joined_table() -> tuple[pd.DataFrame, pd.DataFrame]:
    genome = read_supplementary_table1()
    candidate = pd.read_csv(SUPP3_CLEAN)
    candidate["Gene"] = candidate["Gene"].astype(str).str.upper()

    open_genes = pd.read_csv(OPEN_GENES, sep="\t")
    open_genes["Gene"] = open_genes["Gene"].astype(str).str.upper()
    open_score = open_genes[["Gene", "Score"]].rename(columns={"Score": "opentargets_score_from_open_genes"})

    # Keep the original Fig. 6b-like candidate table as the coordinate source.
    # Fill only missing Open Targets values from the full Open Targets export.
    df = candidate.merge(genome, left_on="Gene", right_on="gene_name", how="left")
    df = df.merge(open_score, on="Gene", how="left")
    df["opentargets_score_original"] = df["open_targets_score"]
    df["opentargets_score"] = df["open_targets_score"].fillna(df["opentargets_score_from_open_genes"])
    df["full_score"] = df["full_score"].fillna(df["Score"])
    df["delta_full_minus_text"] = df["full_score"] - df["text_score"]
    df["delta_full_minus_omics"] = df["full_score"] - df["gene_score"]

    genome = genome.copy()
    genome["full_score_percentile_genome"] = percentile_rank_high_is_high(genome["full_score"])
    genome["text_score_percentile_genome"] = percentile_rank_high_is_high(genome["text_score"])
    genome["omics_score_percentile_genome"] = percentile_rank_high_is_high(genome["gene_score"])
    genome["rank_in_full_TADE_genome"] = genome["full_score"].rank(method="min", ascending=False).astype(int)
    genome["rank_in_text_only_genome"] = genome["text_score"].rank(method="min", ascending=False).astype(int)
    genome["rank_in_omics_only_genome"] = genome["gene_score"].rank(method="min", ascending=False).astype(int)

    rank_cols = [
        "gene_name",
        "full_score_percentile_genome",
        "text_score_percentile_genome",
        "omics_score_percentile_genome",
        "rank_in_full_TADE_genome",
        "rank_in_text_only_genome",
        "rank_in_omics_only_genome",
    ]
    df = df.merge(genome[rank_cols], on="gene_name", how="left")
    df["full_score_percentile_candidate"] = percentile_rank_high_is_high(df["full_score"])
    df["opentargets_score_percentile_candidate"] = percentile_rank_high_is_high(df["opentargets_score"])
    df["rank_in_full_TADE_candidate"] = df["full_score"].rank(method="min", ascending=False).astype(int)
    df["rank_in_text_only_candidate"] = df["text_score"].rank(method="min", ascending=False).astype(int)
    df["rank_in_omics_only_candidate"] = df["gene_score"].rank(method="min", ascending=False).astype(int)
    return df, genome


def classify_quadrant(q: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    q = q.copy()
    labels = []
    reasons = []
    for _, r in q.iterrows():
        delta_high = (
            r["delta_full_minus_text"] >= thresholds["delta_absolute_cutoff"]
            or r["delta_full_minus_text"] >= thresholds["delta_top_quartile_within_quadrant"]
        )
        omics_high = (
            r["gene_score"] >= thresholds["omics_score_global_top25_cutoff"]
            and r["gene_score"] >= thresholds["omics_score_quadrant_median_cutoff"]
        )
        omics_moderate = r["gene_score"] >= thresholds["omics_score_global_top25_cutoff"]
        text_high = r["text_score"] >= thresholds["text_score_candidate_top25_cutoff"]

        if text_high and delta_high and omics_moderate:
            labels.append("both-supported")
            reasons.append(
                f"text_score >= candidate top25 cutoff ({thresholds['text_score_candidate_top25_cutoff']:.4f}); "
                f"delta_full_minus_text >= {thresholds['delta_absolute_cutoff']:.2f}; "
                "omics_score is above genome-wide top25 cutoff"
            )
        elif (not text_high) and delta_high and omics_high:
            labels.append("omics-driven")
            reasons.append(
                "text_score below candidate top25 cutoff; "
                f"delta_full_minus_text is high ({r['delta_full_minus_text']:.4f}); "
                "omics_score exceeds both genome-wide top25 and quadrant median cutoffs"
            )
        elif text_high and (not delta_high):
            labels.append("semantic-driven")
            reasons.append(
                "text_score is high and full_score adds little over text-only "
                f"(delta={r['delta_full_minus_text']:.4f})"
            )
        else:
            labels.append("uncertain/mixed")
            reasons.append(
                "high full score and low Open Targets score, but modality signals do not meet "
                "the prespecified omics-driven, semantic-driven, or both-supported thresholds"
            )
    q["modality_class"] = labels
    q["classification_reason"] = reasons
    return q


def summarize(q: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total = len(q)
    for klass in ["omics-driven", "semantic-driven", "both-supported", "uncertain/mixed"]:
        sub = q[q["modality_class"] == klass]
        rows.append({
            "modality_class": klass,
            "n": len(sub),
            "fraction": len(sub) / total if total else 0,
            "mean_full_score": sub["full_score"].mean(),
            "median_full_score": sub["full_score"].median(),
            "mean_opentargets_score": sub["opentargets_score"].mean(),
            "median_opentargets_score": sub["opentargets_score"].median(),
            "mean_delta_full_minus_text": sub["delta_full_minus_text"].mean(),
            "median_delta_full_minus_text": sub["delta_full_minus_text"].median(),
            "mean_text_score": sub["text_score"].mean(),
            "mean_omics_score": sub["gene_score"].mean(),
        })
    rows.append({
        "modality_class": "total_highTADE_lowOT",
        "n": total,
        "fraction": 1.0 if total else 0,
        "mean_full_score": q["full_score"].mean(),
        "median_full_score": q["full_score"].median(),
        "mean_opentargets_score": q["opentargets_score"].mean(),
        "median_opentargets_score": q["opentargets_score"].median(),
        "mean_delta_full_minus_text": q["delta_full_minus_text"].mean(),
        "median_delta_full_minus_text": q["delta_full_minus_text"].median(),
        "mean_text_score": q["text_score"].mean(),
        "mean_omics_score": q["gene_score"].mean(),
    })
    return pd.DataFrame(rows)


def plot_modality(df: pd.DataFrame, q: pd.DataFrame, thresholds: dict) -> None:
    plot_df = df[df["opentargets_score"].notna()].copy()
    class_map = dict(zip(q["Gene"], q["modality_class"]))
    plot_df["plot_class"] = plot_df["Gene"].map(class_map).fillna("other_candidate")
    colors = {
        "other_candidate": "#b8b8b8",
        "omics-driven": "#1b9e77",
        "semantic-driven": "#d95f02",
        "both-supported": "#7570b3",
        "uncertain/mixed": "#e7298a",
    }
    markers = {
        "other_candidate": "o",
        "omics-driven": "^",
        "semantic-driven": "s",
        "both-supported": "D",
        "uncertain/mixed": "X",
    }
    plt.figure(figsize=(8, 6))
    for klass, sub in plot_df.groupby("plot_class"):
        plt.scatter(
            sub["opentargets_score"],
            sub["full_score"],
            s=60 if klass != "other_candidate" else 18,
            c=colors.get(klass, "#333333"),
            marker=markers.get(klass, "o"),
            alpha=0.9 if klass != "other_candidate" else 0.35,
            label=klass,
            edgecolors="black" if klass != "other_candidate" else "none",
            linewidths=0.4,
        )
    plt.axhline(thresholds["high_tade_cutoff"], color="black", linestyle="--", linewidth=1)
    plt.axvline(thresholds["low_opentargets_cutoff"], color="black", linestyle="--", linewidth=1)
    for _, r in q.iterrows():
        plt.annotate(
            r["Gene"],
            (r["opentargets_score"], r["full_score"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
            color="black",
        )
    plt.xlabel("Open Targets T2D association score")
    plt.ylabel("TADE full score")
    plt.title("Fig. 6b high-TADE / low-Open-Targets modality audit")
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT / "fig6b_highTADE_lowOT_modality_colored.png", dpi=220)
    plt.close()


def md_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if max_rows is not None:
        df = df.head(max_rows)
    if df.empty:
        return "_No rows available._"

    def cell(v) -> str:
        if pd.isna(v):
            return ""
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(map(str, df.columns)) + " |",
        "| " + " | ".join(["---"] * len(df.columns)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(cell(row[c]) for c in df.columns) + " |")
    return "\n".join(lines)


def write_report(q: pd.DataFrame, summary: pd.DataFrame, thresholds: dict) -> None:
    report = OUT / "fig6b_highTADE_lowOT_modality_report.md"
    lines = [
        "# Fig. 6b 左上象限模态贡献审计报告",
        "",
        f"运行日期：{thresholds['run_date']}",
        "",
        "## 数据来源",
        "",
        f"- Fig. 6b-like candidate coordinate table: `{SUPP3_CLEAN}`",
        f"- Genome-wide full/text/omics scores: `{SUPP1}`",
        f"- Open Targets full export: `{OPEN_GENES}`",
        f"- Original TADE candidate ranking: `{CANDIDATE_GENES}`",
        "",
        "未重新调用 GPT，未重新生成 GPT 文本或 BERT embedding；本实验复用已提交 Supplementary Table 1 中的 full/text/omics 分数。",
        "",
        "## 阈值",
        "",
        f"- high TADE cutoff: top 10% within top-280 candidate table, full_score >= {thresholds['high_tade_cutoff']:.6f}",
        f"- low Open Targets cutoff: Open Targets score <= {thresholds['low_opentargets_cutoff']:.6f}",
        f"- candidate-table bottom-25% Open Targets cutoff was {thresholds['low_opentargets_bottom25_cutoff']:.6f}; using <=0.05 gives a slightly less sparse, still low-evidence left-upper quadrant.",
        f"- delta absolute cutoff: {thresholds['delta_absolute_cutoff']:.2f}",
        f"- delta top-quartile within quadrant: {thresholds['delta_top_quartile_within_quadrant']:.6f}",
        f"- text high cutoff: candidate-table text_score top25 >= {thresholds['text_score_candidate_top25_cutoff']:.6f}",
        f"- genome-wide text_score top10 cutoff was {thresholds['text_score_global_top10_cutoff']:.6f}; this is recorded but not used for semantic-high classification because it is too permissive for top-ranked candidates.",
        f"- omics high cutoff: max(genome-wide omics top25, quadrant omics median) = {thresholds['omics_score_effective_high_cutoff']:.6f}",
        "",
        "## 分类汇总",
        "",
        md_table(summary),
        "",
        "## 左上象限候选基因",
        "",
        md_table(q[[
            "Gene", "full_score", "text_score", "gene_score", "opentargets_score",
            "delta_full_minus_text", "delta_full_minus_omics", "modality_class",
            "classification_reason",
        ]]),
        "",
        "## 安全解释",
        "",
        "该审计用于说明 high-TADE / low-Open-Targets 区域内部并不全是同一种信号来源：部分候选是 text-only 分数已经较高的 semantic/both-supported 候选，部分候选则表现为 text-only 很低、full 相对 text-only 大幅上调且 omics-only 分数较高的 omics-driven 候选。该分析是模态贡献审计，不是外部生物学验证。",
    ]
    report.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    df, genome = build_joined_table()
    df = df[df["opentargets_score"].notna()].copy()

    high_tade_cutoff = float(df["full_score"].quantile(0.90))
    low_ot_bottom25 = float(df["opentargets_score"].quantile(0.25))
    low_ot_cutoff = 0.05

    q = df[(df["full_score"] >= high_tade_cutoff) & (df["opentargets_score"] <= low_ot_cutoff)].copy()
    if q.empty:
        raise RuntimeError("No high-TADE / low-Open-Targets candidate found with current thresholds.")

    thresholds = {
        "run_date": date.today().isoformat(),
        "data_scope": "Top-280 TADE candidate genes with Open Targets score available, matching Fig. 6b-like candidate coordinate table.",
        "fig6b_coordinate_source": str(SUPP3_CLEAN),
        "genome_wide_modality_score_source": str(SUPP1),
        "open_targets_source": str(OPEN_GENES),
        "candidate_ranking_source": str(CANDIDATE_GENES),
        "high_tade_rule": "full_score >= 90th percentile within candidate top-280 table",
        "high_tade_cutoff": high_tade_cutoff,
        "low_opentargets_rule": "Open Targets T2D association score <= 0.05; bottom-25 cutoff also recorded",
        "low_opentargets_cutoff": low_ot_cutoff,
        "low_opentargets_bottom25_cutoff": low_ot_bottom25,
        "quadrant_n_using_low_ot_0_05": int(len(q)),
        "quadrant_n_using_bottom25_low_ot": int(((df["full_score"] >= high_tade_cutoff) & (df["opentargets_score"] <= low_ot_bottom25)).sum()),
        "delta_absolute_cutoff": 0.10,
        "delta_top_quartile_within_quadrant": float(q["delta_full_minus_text"].quantile(0.75)),
        "text_score_global_top10_cutoff": float(genome["text_score"].quantile(0.90)),
        "text_score_candidate_top25_cutoff": float(df["text_score"].quantile(0.75)),
        "omics_score_global_top25_cutoff": float(genome["gene_score"].quantile(0.75)),
        "omics_score_quadrant_median_cutoff": float(q["gene_score"].median()),
        "classification_priority": [
            "both-supported if text_score is candidate-table top25 and delta >= 0.10 or quadrant top quartile",
            "omics-driven if text_score is not high, delta is high, and omics_score exceeds genome top25 plus quadrant median",
            "semantic-driven if text_score is high and delta is not high",
            "uncertain/mixed otherwise",
        ],
    }
    thresholds["omics_score_effective_high_cutoff"] = max(
        thresholds["omics_score_global_top25_cutoff"],
        thresholds["omics_score_quadrant_median_cutoff"],
    )

    q = classify_quadrant(q, thresholds)
    output_cols = [
        "Gene", "full_score", "text_score", "gene_score", "opentargets_score",
        "full_score_percentile_candidate", "opentargets_score_percentile_candidate",
        "full_score_percentile_genome", "text_score_percentile_genome",
        "omics_score_percentile_genome", "delta_full_minus_text",
        "delta_full_minus_omics", "rank_in_full_TADE_candidate",
        "rank_in_text_only_candidate", "rank_in_omics_only_candidate",
        "rank_in_full_TADE_genome", "rank_in_text_only_genome",
        "rank_in_omics_only_genome", "modality_class", "classification_reason",
    ]
    q = q.sort_values(["modality_class", "delta_full_minus_text", "gene_score"], ascending=[True, False, False])
    q[output_cols].to_csv(OUT / "fig6b_highTADE_lowOT_modality_contribution.csv", index=False)

    summary = summarize(q)
    summary.to_csv(OUT / "fig6b_highTADE_lowOT_modality_summary.csv", index=False)

    top_omics = q[q["modality_class"] == "omics-driven"].sort_values(
        ["delta_full_minus_text", "gene_score", "full_score"], ascending=False
    )
    top_omics[output_cols].to_csv(OUT / "fig6b_highTADE_lowOT_top_omics_driven_candidates.csv", index=False)

    with open(OUT / "fig6b_highTADE_lowOT_thresholds.json", "w", encoding="utf-8") as f:
        json.dump(thresholds, f, indent=2, ensure_ascii=False)

    plot_modality(df, q, thresholds)
    write_report(q, summary, thresholds)

    print(f"Output directory: {OUT}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
