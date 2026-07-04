#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
from pathlib import Path


BASE = Path("/root/autodl-tmp/尤毅复现/computational_validation_outputs")
PDB_DIR = BASE / "pdb_positive_controls"
OUT = BASE / "pdb_positive_control_structural_validation.csv"

CASES = [
    {
        "pdb_id": "4FFW",
        "gene": "DPP4",
        "drug": "Sitagliptin",
        "ligand": "715",
        "evidence_grade": "A",
        "structure_type": "experimentally resolved complex",
        "source": "RCSB PDB 4FFW",
    },
    {
        "pdb_id": "2XKW",
        "gene": "PPARG",
        "drug": "Pioglitazone",
        "ligand": "P1B",
        "evidence_grade": "A",
        "structure_type": "experimentally resolved complex",
        "source": "RCSB PDB 2XKW",
    },
    {
        "pdb_id": "7VSI",
        "gene": "SLC5A2",
        "drug": "Empagliflozin",
        "ligand": "7R3",
        "evidence_grade": "A",
        "structure_type": "experimentally resolved complex",
        "source": "RCSB PDB 7VSI",
    },
    {
        "pdb_id": "6JB3",
        "gene": "ABCC8",
        "drug": "Repaglinide",
        "ligand": "BJX",
        "evidence_grade": "A",
        "structure_type": "experimentally resolved complex",
        "source": "RCSB PDB 6JB3",
    },
    {
        "pdb_id": "6PZ9",
        "gene": "ABCC8",
        "drug": "Repaglinide",
        "ligand": "BJX",
        "evidence_grade": "A",
        "structure_type": "experimentally resolved complex",
        "source": "RCSB PDB 6PZ9",
    },
    {
        "pdb_id": "7TYS",
        "gene": "KCNJ11/ABCC8 complex",
        "drug": "Repaglinide",
        "ligand": "BJX",
        "evidence_grade": "A",
        "structure_type": "experimentally resolved complex",
        "source": "RCSB PDB 7TYS",
    },
]

POLAR_ELEMENTS = {"N", "O", "S", "P"}


def parse_atoms(path: Path):
    atoms = []
    for line in path.read_text(errors="ignore").splitlines():
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        try:
            atoms.append(
                {
                    "record": line[:6].strip(),
                    "atom": line[12:16].strip(),
                    "res": line[17:20].strip(),
                    "chain": line[21].strip(),
                    "seq": line[22:26].strip(),
                    "x": float(line[30:38]),
                    "y": float(line[38:46]),
                    "z": float(line[46:54]),
                    "elem": (line[76:78].strip() or line[12:16].strip()[0]).upper(),
                }
            )
        except Exception:
            continue
    return atoms


def distance(a, b):
    return math.sqrt((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2 + (a["z"] - b["z"]) ** 2)


rows = []
for case in CASES:
    path = PDB_DIR / f"{case['pdb_id']}.pdb"
    atoms = parse_atoms(path)
    ligand_atoms = [a for a in atoms if a["record"] == "HETATM" and a["res"] == case["ligand"]]
    protein_atoms = [a for a in atoms if a["record"] == "ATOM"]

    min_dist = None
    contact4 = 0
    contact5 = 0
    polar = 0
    residues4 = set()
    residues5 = set()

    for la in ligand_atoms:
        for pa in protein_atoms:
            d = distance(la, pa)
            if min_dist is None or d < min_dist:
                min_dist = d
            if d <= 4.0:
                contact4 += 1
                residues4.add(f"{pa['res']}{pa['seq']}{pa['chain']}")
            if d <= 5.0:
                contact5 += 1
                residues5.add(f"{pa['res']}{pa['seq']}{pa['chain']}")
            if d <= 3.5 and la["elem"] in POLAR_ELEMENTS and pa["elem"] in POLAR_ELEMENTS:
                polar += 1

    if len(residues4) >= 10 and polar >= 2:
        structural_grade = "strong experimental complex contact support"
    elif len(residues4) >= 5:
        structural_grade = "moderate experimental complex contact support"
    else:
        structural_grade = "limited experimental complex contact support"

    row = dict(case)
    row.update(
        {
            "pdb_file": str(path),
            "ligand_atom_count": len(ligand_atoms),
            "protein_atom_count": len(protein_atoms),
            "min_ligand_protein_distance_A": round(min_dist or float("nan"), 3),
            "contact_atom_pairs_4A": contact4,
            "contact_residue_count_4A": len(residues4),
            "contact_residues_4A": ";".join(sorted(residues4)),
            "contact_atom_pairs_5A": contact5,
            "contact_residue_count_5A": len(residues5),
            "polar_contact_proxy_count_3p5A": polar,
            "structural_grade": structural_grade,
            "manuscript_safe_interpretation": "Experimentally resolved drug-target complex used as a positive structural control.",
        }
    )
    rows.append(row)

with OUT.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print(OUT)
