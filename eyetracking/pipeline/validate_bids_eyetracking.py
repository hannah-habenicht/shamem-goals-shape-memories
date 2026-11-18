#!/usr/bin/env python
"""
Small validator for the eyetracking BIDS derivative dataset produced in this repo.

What it checks:
- derivative-style file paths with the Python `bids_validator` package
- required top-level derivative metadata
- required eye-tracking `_physio` metadata and column conventions
- required `_physioevents` metadata and column conventions
- compressed TSV.GZ files being header-less, as required by BIDS

Usage:
    python "Code/Eyetracking Pipeline/validate_bids_eyetracking.py"
    python "Code/Eyetracking Pipeline/validate_bids_eyetracking.py" --dataset "Data/derivatives/eyetracking"
    python "Code/Eyetracking Pipeline/validate_bids_eyetracking.py" --verbose
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

try:
    from bids_validator import BIDSValidator
except ImportError:  # pragma: no cover
    BIDSValidator = None


VALID_RECORDED_EYES = {"left", "right", "cyclopean"}
VALID_COORDINATE_SYSTEMS = {"gaze-on-screen", "eye-in-head", "gaze-in-world", "custom"}


@dataclass
class Finding:
    level: str
    path: str
    message: str


@dataclass
class Report:
    findings: list[Finding] = field(default_factory=list)

    def error(self, path: Path | str, message: str) -> None:
        self.findings.append(Finding("ERROR", str(path), message))

    def warning(self, path: Path | str, message: str) -> None:
        self.findings.append(Finding("WARNING", str(path), message))

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.level == "ERROR"]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.level == "WARNING"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate eyetracking BIDS derivative files.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("Data/derivatives/eyetracking"),
        help="Path to the derivative dataset root.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print warnings in addition to errors.",
    )
    return parser.parse_args()


def load_json(path: Path, report: Report) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        report.error(path, "Missing JSON file.")
    except json.JSONDecodeError as exc:
        report.error(path, f"Invalid JSON: {exc}")
    return None


def read_tsvgz_first_row(path: Path, report: Report) -> list[str] | None:
    try:
        with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
            reader = csv.reader(f, delimiter="\t")
            return next(reader, None)
    except FileNotFoundError:
        report.error(path, "Missing TSV.GZ file.")
    except OSError as exc:
        report.error(path, f"Cannot read gzip content: {exc}")
    return None


def validate_dataset_description(dataset_root: Path, report: Report) -> None:
    path = dataset_root / "dataset_description.json"
    data = load_json(path, report)
    if not data:
        return

    for key in ("Name", "BIDSVersion", "DatasetType", "GeneratedBy"):
        if key not in data:
            report.error(path, f"Missing required key `{key}`.")

    if data.get("DatasetType") != "derivative":
        report.error(path, "DatasetType should be `derivative` for this dataset.")

    generated_by = data.get("GeneratedBy")
    if generated_by is not None and not isinstance(generated_by, list):
        report.error(path, "`GeneratedBy` must be a list.")


def validate_paths_with_bids_validator(dataset_root: Path, report: Report) -> None:
    if BIDSValidator is None:
        report.warning(dataset_root, "Python package `bids_validator` is not installed; skipping path validation.")
        return

    validator = BIDSValidator()
    dataset_name = dataset_root.name

    for path in dataset_root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(dataset_root).as_posix()
        derivative_rel = f"/derivatives/{dataset_name}/{rel}"
        if not validator.is_bids(derivative_rel):
            report.error(path, f"Path is not BIDS-valid according to `bids_validator`: {derivative_rel}")


def validate_physio(dataset_root: Path, tsv_path: Path, report: Report) -> None:
    json_path = tsv_path.with_suffix("").with_suffix(".json")
    metadata = load_json(json_path, report)
    first_row = read_tsvgz_first_row(tsv_path, report)
    if metadata is None or first_row is None:
        return

    for key in ("SamplingFrequency", "StartTime", "Columns", "PhysioType", "RecordedEye", "SampleCoordinateSystem"):
        if key not in metadata:
            report.error(json_path, f"Missing required key `{key}`.")

    columns = metadata.get("Columns")
    if not isinstance(columns, list) or not all(isinstance(col, str) for col in columns):
        report.error(json_path, "`Columns` must be a list of strings.")
        return

    if metadata.get("PhysioType") != "eyetrack":
        report.error(json_path, "`PhysioType` should be `eyetrack` for eye-tracking recordings.")

    recorded_eye = metadata.get("RecordedEye")
    if recorded_eye is not None and recorded_eye not in VALID_RECORDED_EYES:
        report.error(json_path, f"`RecordedEye` must be one of {sorted(VALID_RECORDED_EYES)}.")

    coordinate_system = metadata.get("SampleCoordinateSystem")
    if coordinate_system is not None and coordinate_system not in VALID_COORDINATE_SYSTEMS:
        report.error(
            json_path,
            "`SampleCoordinateSystem` must be one of "
            f"{sorted(VALID_COORDINATE_SYSTEMS)}; `pixel` is a unit, not a coordinate system.",
        )

    if len(columns) < 3:
        report.error(json_path, "Eye-tracking physio files need at least `timestamp`, `x_coordinate`, and `y_coordinate`.")
    else:
        expected_prefix = ["timestamp", "x_coordinate", "y_coordinate"]
        if columns[:3] != expected_prefix:
            report.error(
                json_path,
                f"First three `Columns` entries must be {expected_prefix}, found {columns[:3]}.",
            )

    for axis_col in ("x_coordinate", "y_coordinate"):
        axis_meta = metadata.get(axis_col)
        if not isinstance(axis_meta, dict) or "Units" not in axis_meta:
            report.error(json_path, f"Column metadata `{axis_col}.Units` is required for eye-tracking recordings.")

    if first_row == columns:
        report.error(tsv_path, "TSV.GZ contains a header row. BIDS compressed tabular files must be header-less.")

    if len(first_row) != len(columns):
        report.error(tsv_path, f"First data row has {len(first_row)} fields but `Columns` lists {len(columns)}.")


def validate_physioevents(tsv_path: Path, report: Report) -> None:
    json_path = tsv_path.with_suffix("").with_suffix(".json")
    metadata = load_json(json_path, report)
    first_row = read_tsvgz_first_row(tsv_path, report)
    if metadata is None or first_row is None:
        return

    for key in ("Columns", "OnsetSource"):
        if key not in metadata:
            report.error(json_path, f"Missing required key `{key}`.")

    columns = metadata.get("Columns")
    if not isinstance(columns, list) or not all(isinstance(col, str) for col in columns):
        report.error(json_path, "`Columns` must be a list of strings.")
        return

    if not columns:
        report.error(json_path, "`Columns` must not be empty.")
    elif columns[0] != "onset":
        report.error(json_path, f"First `Columns` entry must be `onset`, found `{columns[0]}`.")

    if first_row == columns:
        report.error(tsv_path, "TSV.GZ contains a header row. BIDS compressed tabular files must be header-less.")

    if len(first_row) != len(columns):
        report.error(tsv_path, f"First data row has {len(first_row)} fields but `Columns` lists {len(columns)}.")


def iter_target_files(dataset_root: Path, suffix: str) -> Iterable[Path]:
    return sorted(p for p in dataset_root.rglob(f"*{suffix}") if p.is_file())


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset.resolve()
    report = Report()

    if not dataset_root.exists():
        report.error(dataset_root, "Dataset root does not exist.")
    elif not dataset_root.is_dir():
        report.error(dataset_root, "Dataset root is not a directory.")

    if report.errors:
        for finding in report.errors:
            print(f"[{finding.level}] {finding.path}: {finding.message}")
        return 2

    validate_dataset_description(dataset_root, report)
    validate_paths_with_bids_validator(dataset_root, report)

    physio_files = iter_target_files(dataset_root, "_physio.tsv.gz")
    physioevents_files = iter_target_files(dataset_root, "_physioevents.tsv.gz")

    if not physio_files:
        report.warning(dataset_root, "No `_physio.tsv.gz` files found.")
    if not physioevents_files:
        report.warning(dataset_root, "No `_physioevents.tsv.gz` files found.")

    for tsv_path in physio_files:
        validate_physio(dataset_root, tsv_path, report)

    for tsv_path in physioevents_files:
        validate_physioevents(tsv_path, report)

    if args.verbose:
        for finding in report.findings:
            print(f"[{finding.level}] {finding.path}: {finding.message}")
    else:
        grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
        for finding in report.errors:
            grouped[(finding.level, finding.message)].append(finding.path)

        for (level, message), paths in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0][1])):
            print(f"[{level}] {message}")
            print(f"  count: {len(paths)}")
            print(f"  example: {paths[0]}")

    print(
        f"Summary: {len(report.errors)} error(s), {len(report.warnings)} warning(s), "
        f"{len(physio_files)} physio file(s), {len(physioevents_files)} physioevents file(s)."
    )
    return 1 if report.errors else 0


if __name__ == "__main__":
    sys.exit(main())
