#!/usr/bin/env python3
"""Regenerate GPT-4o gene descriptions with fixed decoding parameters.

Environment variables:
  OPENAI_API_KEY   API key for the OpenAI-compatible endpoint.
  OPENAI_BASE_URL  Optional base URL for an OpenAI-compatible endpoint.

Example:
  python regenerate_gene_descriptions_fixed_params.py \
    --input /root/autodl-tmp/yyfuxian/zip_check_TADE/TADE/data_source/all_gene_desc.csv \
    --output /root/autodl-tmp/yyfuxian/all_gene_desc_fixed_gpt4o.csv
"""

from __future__ import annotations

import argparse
import os
import time

import pandas as pd
from openai import OpenAI


PROMPT = (
    "Is there a relationship between {gene} gene and type 2 diabetes? "
    "If it exists, please describe the possible relevant mechanisms in a short paragraph, "
    "without mentioning whether the gene can be druggable, and try to be scientifically rigorous; "
    "If it does not exist, please answer that it is currently unknown."
)


def revise_phrase(text: str) -> str:
    text = text.replace("尾-cell", "beta-cell")
    text = text.replace("β-cell", "beta-cell")
    if "As of" in text:
        start_index = text.find("As of")
        end_index = text.find(",", start_index)
        if end_index != -1:
            text = text[:start_index] + text[end_index + 1 :].strip().capitalize()
    return text


def load_genes(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "gene_name" not in df.columns:
        first = df.columns[0]
        df = df.rename(columns={first: "gene_name"})
    if "gene_desc" not in df.columns:
        df["gene_desc"] = pd.NA
    return df[["gene_name", "gene_desc"]].copy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--sleep", type=float, default=0.0)
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set.")

    client = OpenAI(
        api_key=api_key,
        base_url=os.environ.get("OPENAI_BASE_URL") or None,
    )

    df = load_genes(args.input)
    if os.path.exists(args.output):
        cached = load_genes(args.output)
        cached_map = dict(zip(cached["gene_name"], cached["gene_desc"]))
        df["gene_desc"] = df["gene_name"].map(cached_map).combine_first(df["gene_desc"])

    for idx, row in df.iterrows():
        gene = row["gene_name"]
        if pd.notna(row["gene_desc"]) and str(row["gene_desc"]).strip():
            continue

        completion = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": PROMPT.format(gene=gene)}],
            temperature=0,
            top_p=1.0,
            max_tokens=512,
            frequency_penalty=0,
            presence_penalty=0,
        )
        df.at[idx, "gene_desc"] = revise_phrase(completion.choices[0].message.content or "")
        df.to_csv(args.output, index=False)
        if args.sleep:
            time.sleep(args.sleep)

    df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
