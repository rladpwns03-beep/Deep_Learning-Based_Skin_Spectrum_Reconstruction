"""Helpers shared by the post hoc, exploratory experiments (BiGRU structure ablation, repeated experiments).

  - snapshot_existing_results: sha256 of every file under results/ outside the two new result folders, written once
    (never overwritten when a run is resumed) so that tests can show the stored results were not changed.
  - pairs_for: the General reconstruction test pairs of a set of test subjects (all ordered pairs i != j within a subject),
    in the same order as final_utils.test_pairs (which can only read data/splits_final).
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "scripts"))
from common.data_loader import Spectra                                    # noqa: E402

RES = ROOT / "results"
NEW_RESULT_FOLDERS = ("ablation_bigru_structure", "repeat_experiments")
SNAPSHOT_NAME = "existing_results_sha256.json"
# Not compared: this file stores the generation time of each figure and is rewritten whenever tests/test_figures.py (or make_figures.py) runs.
VOLATILE = ("figures_final/figure_manifest.csv",)


def changed_existing_results(snapshot: dict) -> list:
    """Files of the snapshot whose sha256 differs now (or that are gone), apart from VOLATILE."""
    now = existing_results_sha256()
    return [k for k, v in snapshot.items() if k not in VOLATILE and now.get(k) != v]
STATUS = "post hoc, exploratory"
FORBIDDEN_WORDS = ["significant", "significantly", "outperform", "outperforms", "outperformed", "superior", "best", "better", "worse", "novel",
                   "robust", "clearly"]


def existing_results_sha256() -> dict:
    return {str(p.relative_to(RES)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(RES.rglob("*")) if p.is_file() and p.relative_to(RES).parts[0] not in NEW_RESULT_FOLDERS}


def snapshot_existing_results(out_root: Path) -> Path:
    """Write the sha256 list once. A resumed run keeps the first snapshot."""
    path = out_root / SNAPSHOT_NAME
    if not path.exists():
        out_root.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(existing_results_sha256(), indent=0), encoding="utf-8")
    return path


def code_commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def pairs_for(data: Spectra, test_subjects):
    """(ii, jj, subject array): X = a[ii], truth = a[jj], target HbI = hbi[jj]. Subjects sorted; i outer, j inner, in file order."""
    ii, jj, ss = [], [], []
    for s in sorted(test_subjects):
        ix = np.flatnonzero(data.subject == s)
        for i in ix:
            for j in ix:
                if i != j:
                    ii.append(i); jj.append(j); ss.append(s)
    return np.array(ii), np.array(jj), np.array(ss)


def cohort_of(subject: str) -> str:
    return "External cohort" if subject >= "S100" else "Development cohort"
