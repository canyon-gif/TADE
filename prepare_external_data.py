#!/usr/bin/env python3
"""Link or copy the external TADE data package into the repository layout.

Usage:
  python prepare_external_data.py /path/to/TADE_external_data_package
  python prepare_external_data.py /path/to/TADE_external_data_package --copy

By default, the script creates symlinks so the large files are not duplicated.
Use --copy on platforms or filesystems where symlinks are inconvenient.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


LINKS = [
    ("fixed_inputs/data_source/all_gene_desc.csv", "data_source/all_gene_desc.csv"),
    ("fixed_inputs/data_source/all_omics_text.csv", "data_source/all_omics_text.csv"),
    ("fixed_inputs/data_source/omics_data_source", "data_source/omics_data_source"),
    ("fixed_inputs/data_source/open_genes", "data_source/open_genes"),
    ("benchmarks/druggable_gene", "datasets/druggable_gene"),
    ("benchmarks/gene_drug_interaction", "datasets/gene_drug_interaction"),
    (
        "model_artifacts/gene_drug_interaction",
        "code/train_val_test_draw/save/gene_drug_interaction",
    ),
    ("results/candidate_genes", "results/candidate_genes"),
    ("results/docking_files", "results/docking_files"),
]


def remove_existing(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def install_one(src: Path, dst: Path, copy: bool, overwrite: bool) -> str:
    if not src.exists():
        return f"missing source: {src}"

    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return f"exists, skipped: {dst}"
        remove_existing(dst)

    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
        return f"copied: {dst}"

    os.symlink(src.resolve(), dst, target_is_directory=src.is_dir())
    return f"linked: {dst} -> {src.resolve()}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_package", help="Path to extracted TADE_external_data_package")
    parser.add_argument("--copy", action="store_true", help="copy files instead of creating symlinks")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing target files/directories/symlinks",
    )
    args = parser.parse_args()

    repo = Path(__file__).resolve().parent
    package = Path(args.data_package).expanduser().resolve()
    if not package.exists():
        raise SystemExit(f"Data package not found: {package}")

    for src_rel, dst_rel in LINKS:
        print(install_one(package / src_rel, repo / dst_rel, args.copy, args.overwrite))

    print("\nDone. The repository now has the data_source/, datasets/, results/, and save/ paths expected by the notebooks and scripts.")


if __name__ == "__main__":
    main()
