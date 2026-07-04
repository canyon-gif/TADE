#!/usr/bin/env python3
"""Summarize existing docking outputs as computational structural validation.

This does not rerun docking. It parses the submitted Vina score logs and
best-complex PDB files, then reports ligand-proximal protein contacts as a
pocket/contact proxy.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import pandas as pd


ROOT = Path("/root/autodl-tmp/尤毅复现")
DOCK_DIR = Path("/root/autodl-tmp/yyfuxian/zip_check_TADE/TADE/results/docking_files")
OUT_DIR = ROOT / "computational_validation_outputs"

POLAR_ELEMENTS = {"N", "O", "S"}


def parse_pair_from_name(path: Path) -> tuple[str, str]:
    stem = path.name.replace("_docking_scores.txt", "").replace("_best_complex.pdb", "")
    parts = stem.split("_", 1)
    if len(parts) != 2:
        raise ValueError(f"Cannot parse pair name from {path.name}")
    return parts[0], parts[1]


def parse_vina_scores(path: Path) -> dict:
    rows = []
    pattern = re.compile(r"^\s*(\d+)\s+(-?\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)")
    for line in path.read_text(errors="ignore").splitlines():
        m = pattern.match(line)
        if m:
            rows.append(
                {
                    "mode": int(m.group(1)),
                    "affinity_kcal_mol": float(m.group(2)),
                    "rmsd_lb": float(m.group(3)),
                    "rmsd_ub": float(m.group(4)),
                }
            )
    if not rows:
        return {
            "best_affinity_kcal_mol": math.nan,
            "n_modes": 0,
            "mean_top3_affinity": math.nan,
            "top3_affinity_range": math.nan,
        }
    df = pd.DataFrame(rows)
    top3 = df.head(3)
    return {
        "best_affinity_kcal_mol": float(df.iloc[0]["affinity_kcal_mol"]),
        "n_modes": int(len(df)),
        "mean_top3_affinity": float(top3["affinity_kcal_mol"].mean()),
        "top3_affinity_range": float(top3["affinity_kcal_mol"].max() - top3["affinity_kcal_mol"].min()),
    }


def pdb_atom_element(line: str) -> str:
    elem = line[76:78].strip()
    if elem:
        return elem.upper()
    name = line[12:16].strip()
    return re.sub(r"[^A-Za-z]", "", name)[:1].upper()


def parse_pdb_atoms(path: Path) -> tuple[list[dict], list[dict]]:
    ligand = []
    protein = []
    for line in path.read_text(errors="ignore").splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        try:
            atom = {
                "record": line[:6].strip(),
                "atom_name": line[12:16].strip(),
                "resname": line[17:20].strip(),
                "chain": line[21:22].strip(),
                "resseq": line[22:26].strip(),
                "x": float(line[30:38]),
                "y": float(line[38:46]),
                "z": float(line[46:54]),
                "element": pdb_atom_element(line),
            }
        except ValueError:
            continue
        # Docking complexes use UNL for ligand; protein residues are standard
        # amino-acid residue names. Some files mark ligand as ATOM rather than
        # HETATM, so residue name is the robust discriminator.
        if atom["resname"] == "UNL":
            ligand.append(atom)
        else:
            protein.append(atom)
    return ligand, protein


def dist(a: dict, b: dict) -> float:
    return math.sqrt((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2 + (a["z"] - b["z"]) ** 2)


def contact_summary(path: Path) -> dict:
    ligand, protein = parse_pdb_atoms(path)
    contacts4 = []
    contacts5 = []
    polar_contacts = []
    min_distance = math.nan
    if ligand and protein:
        min_distance = min(dist(l, p) for l in ligand for p in protein)
    for l in ligand:
        for p in protein:
            d = dist(l, p)
            resid = f"{p['resname']}{p['resseq']}{p['chain']}".strip()
            if d <= 4.0:
                contacts4.append((resid, p["resname"], p["element"], l["element"], d))
            if d <= 5.0:
                contacts5.append((resid, p["resname"], p["element"], l["element"], d))
            if d <= 3.5 and p["element"] in POLAR_ELEMENTS and l["element"] in POLAR_ELEMENTS:
                polar_contacts.append((resid, d))
    residues4 = sorted({x[0] for x in contacts4})
    residues5 = sorted({x[0] for x in contacts5})
    return {
        "complex_file": str(path),
        "ligand_atom_count": len(ligand),
        "protein_atom_count": len(protein),
        "min_ligand_protein_distance_A": min_distance,
        "contact_atom_pairs_4A": len(contacts4),
        "contact_residue_count_4A": len(residues4),
        "contact_residues_4A": ";".join(residues4[:30]),
        "contact_atom_pairs_5A": len(contacts5),
        "contact_residue_count_5A": len(residues5),
        "polar_contact_proxy_count_3p5A": len(polar_contacts),
    }


def structural_grade(best_affinity: float, contact_residue_count: int, polar_count: int) -> str:
    if math.isnan(best_affinity):
        return "not_assessable"
    if best_affinity <= -7.0 and contact_residue_count >= 8:
        return "strong_computational_structural_plausibility"
    if best_affinity <= -6.0 and contact_residue_count >= 5:
        return "moderate_computational_structural_plausibility"
    if best_affinity <= -5.0 and contact_residue_count >= 3:
        return "weak_computational_structural_plausibility"
    return "limited_computational_structural_plausibility"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for score_path in sorted(DOCK_DIR.glob("*_docking_scores.txt")):
        gene, drug = parse_pair_from_name(score_path)
        complex_path = DOCK_DIR / f"{gene}_{drug}_best_complex.pdb"
        row = {
            "gene": gene,
            "drug": drug,
            "docking_score_file": str(score_path),
        }
        row.update(parse_vina_scores(score_path))
        if complex_path.exists():
            row.update(contact_summary(complex_path))
        else:
            row.update(
                {
                    "complex_file": "",
                    "ligand_atom_count": 0,
                    "protein_atom_count": 0,
                    "min_ligand_protein_distance_A": math.nan,
                    "contact_atom_pairs_4A": 0,
                    "contact_residue_count_4A": 0,
                    "contact_residues_4A": "",
                    "contact_atom_pairs_5A": 0,
                    "contact_residue_count_5A": 0,
                    "polar_contact_proxy_count_3p5A": 0,
                }
            )
        row["structural_plausibility_grade"] = structural_grade(
            row["best_affinity_kcal_mol"],
            row["contact_residue_count_4A"],
            row["polar_contact_proxy_count_3p5A"],
        )
        row["manuscript_safe_interpretation"] = (
            "Computational docking/contact support only; not experimental binding or target-engagement validation."
        )
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values(["best_affinity_kcal_mol", "contact_residue_count_4A"])
    summary.to_csv(OUT_DIR / "existing_docking_structural_validation_summary.csv", index=False)

    # Merge the highlighted-pair evidence table when available.
    evidence_path = ROOT / "remaining_experiment_outputs/highlighted_pair_evidence_grading_curated.csv"
    if evidence_path.exists():
        evidence = pd.read_csv(evidence_path)
        merged = summary.merge(evidence, on=["gene", "drug"], how="left")
        merged.to_csv(OUT_DIR / "highlighted_pair_structural_evidence_summary.csv", index=False)

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
