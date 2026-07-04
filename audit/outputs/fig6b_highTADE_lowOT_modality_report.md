# Fig. 6b 左上象限模态贡献审计报告

运行日期：2026-07-02

## 数据来源

- Fig. 6b-like candidate coordinate table: `submitted_supplementary_tables/TADE_NC_SupplementaryTable3.xlsx`
- Genome-wide full/text/omics scores: `submitted_supplementary_tables/TADE_NC_SupplementaryTable1.xlsx`
- Open Targets full export: `fixed_inputs/data_source/open_genes/open_genes.tsv`
- Original TADE candidate ranking: `results/candidate_genes/genes.csv`

未重新调用 GPT，未重新生成 GPT 文本或 BERT embedding；本实验复用已提交 Supplementary Table 1 中的 full/text/omics 分数。

## 阈值

- high TADE cutoff: top 10% within top-280 candidate table, full_score >= 0.975492
- low Open Targets cutoff: Open Targets score <= 0.050000
- candidate-table bottom-25% Open Targets cutoff was 0.015873; using <=0.05 gives a slightly less sparse, still low-evidence left-upper quadrant.
- delta absolute cutoff: 0.10
- delta top-quartile within quadrant: 0.963570
- text high cutoff: candidate-table text_score top25 >= 0.699184
- genome-wide text_score top10 cutoff was 0.029899; this is recorded but not used for semantic-high classification because it is too permissive for top-ranked candidates.
- omics high cutoff: max(genome-wide omics top25, quadrant omics median) = 0.541653

## 分类汇总

| modality_class | n | fraction | mean_full_score | median_full_score | mean_opentargets_score | median_opentargets_score | mean_delta_full_minus_text | median_delta_full_minus_text | mean_text_score | mean_omics_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| omics-driven | 3 | 0.5000 | 0.9793 | 0.9809 | 0.0199 | 0.0153 | 0.9631 | 0.9658 | 0.0162 | 0.6716 |
| semantic-driven | 0 | 0.0000 |  |  |  |  |  |  |  |  |
| both-supported | 1 | 0.1667 | 0.9790 | 0.9790 | 0.0057 | 0.0057 | 0.1548 | 0.1548 | 0.8242 | 0.4221 |
| uncertain/mixed | 2 | 0.3333 | 0.9795 | 0.9795 | 0.0211 | 0.0211 | 0.9515 | 0.9515 | 0.0280 | 0.3343 |
| total_highTADE_lowOT | 6 | 1.0000 | 0.9793 | 0.9795 | 0.0179 | 0.0115 | 0.8245 | 0.9563 | 0.1548 | 0.5176 |

## 左上象限候选基因

| Gene | full_score | text_score | gene_score | opentargets_score | delta_full_minus_text | delta_full_minus_omics | modality_class | classification_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AGL | 0.9790 | 0.8242 | 0.4221 | 0.0057 | 0.1548 | 0.5570 | both-supported | text_score >= candidate top25 cutoff (0.6992); delta_full_minus_text >= 0.10; omics_score is above genome-wide top25 cutoff |
| CD109 | 0.9813 | 0.0146 | 0.6406 | 0.0022 | 0.9667 | 0.3407 | omics-driven | text_score below candidate top25 cutoff; delta_full_minus_text is high (0.9667); omics_score exceeds both genome-wide top25 and quadrant median cutoffs |
| CNTN6 | 0.9809 | 0.0150 | 0.6969 | 0.0421 | 0.9658 | 0.2839 | omics-driven | text_score below candidate top25 cutoff; delta_full_minus_text is high (0.9658); omics_score exceeds both genome-wide top25 and quadrant median cutoffs |
| ESRRG | 0.9758 | 0.0190 | 0.6772 | 0.0153 | 0.9569 | 0.2986 | omics-driven | text_score below candidate top25 cutoff; delta_full_minus_text is high (0.9569); omics_score exceeds both genome-wide top25 and quadrant median cutoffs |
| SLC7A2 | 0.9792 | 0.0235 | 0.4427 | 0.0078 | 0.9557 | 0.5365 | uncertain/mixed | high full score and low Open Targets score, but modality signals do not meet the prespecified omics-driven, semantic-driven, or both-supported thresholds |
| PSPH | 0.9799 | 0.0325 | 0.2259 | 0.0344 | 0.9474 | 0.7540 | uncertain/mixed | high full score and low Open Targets score, but modality signals do not meet the prespecified omics-driven, semantic-driven, or both-supported thresholds |

## 安全解释

该审计用于说明 high-TADE / low-Open-Targets 区域内部并不全是同一种信号来源：部分候选是 text-only 分数已经较高的 semantic/both-supported 候选，部分候选则表现为 text-only 很低、full 相对 text-only 大幅上调且 omics-only 分数较高的 omics-driven 候选。该分析是模态贡献审计，不是外部生物学验证。