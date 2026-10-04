"""Validate the public release tree without requiring private behavior inputs."""
from __future__ import annotations

import ast
import json
import re
import struct
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TEXT_EXTENSIONS = {".py", ".md", ".txt", ".json", ".csv", ".tex", ".sh", ".yml", ".yaml", ".toml", ".ini"}
REQUIRED = (
    "README.md", "DATA.md", "LICENSE", "CONTRIBUTING.md",
    "docs/ARCHITECTURE.md", "docs/WORKFLOW.md", "docs/PAPER_ALIGNMENT.md",
    "daily_share_model.py",
    "preference_dataset.py", "preference_alignment_finetune.py", "SEIR/seir_timeseries.py",
    "SEIR/analysis/plot_figure12_10seed.py",
    "SEIR/results/policy_calendar_eta1_reproduction/provenance/calibration/best_effective_hazard_params.json",
    "SEIR/results/policy_calendar_eta1_reproduction/random_seed_table9/table9_10_seed_summary.csv",
    "SEIR/figures/fig15_seir_policy_compare_four_calendars_10seed.png",
)
PRIVATE_PATTERNS = (
    ("absolute Windows path", re.compile(r"[A-Za-z]:\\(?:Users|Study|Research)\\", re.I)),
    ("Linux private path", re.compile(r"/(?:root|home)/", re.I)),
    ("compute-environment name", re.compile(r"auto" + r"dl", re.I)),
    ("GitHub token", re.compile(r"gh" + r"_[pousr]_[A-Za-z0-9]{20,}")),
    ("private key", re.compile(r"BEGIN " + "PRIVATE KEY", re.I)),
)


def png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"invalid PNG: {path.relative_to(ROOT)}")
    return struct.unpack(">II", header[16:24])


def check_text(errors: list[str]) -> None:
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        rel = path.relative_to(ROOT).as_posix()
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            errors.append(f"non-UTF-8 text file: {rel}: {exc}")
            continue
        for label, pattern in PRIVATE_PATTERNS:
            if pattern.search(text):
                errors.append(f"{label} found in {rel}")
        if path.suffix.lower() == ".py":
            try:
                ast.parse(text, filename=rel)
            except SyntaxError as exc:
                errors.append(f"Python syntax error in {rel}: {exc}")
        if path.suffix.lower() == ".json":
            try:
                json.loads(text)
            except json.JSONDecodeError as exc:
                errors.append(f"invalid JSON in {rel}: {exc}")


def check_structure(errors: list[str]) -> None:
    for rel in REQUIRED:
        if not (ROOT / rel).is_file():
            errors.append(f"missing required release file: {rel}")
    forbidden_suffixes = {".pt", ".pth", ".ckpt", ".safetensors", ".bin", ".csv.gz"}
    forbidden = [p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*") if p.is_file() and p.suffix.lower() in forbidden_suffixes]
    if forbidden:
        errors.append(f"private/large artifact files remain: {forbidden}")
    for path in ROOT.rglob("*"):
        if path.is_file() and path.stat().st_size >= 100 * 1024 * 1024:
            errors.append(f"file exceeds GitHub 100 MiB limit: {path.relative_to(ROOT)}")


def check_calendars(errors: list[str]) -> None:
    for policy in ("early", "late", "short", "long"):
        path = ROOT / policy / "calendar.csv"
        if not path.is_file():
            errors.append(f"missing calendar: {policy}")
            continue
        frame = pd.read_csv(path)
        dates = pd.to_datetime(frame.get("date"), errors="coerce")
        if len(frame) != 184 or dates.isna().any() or dates.iloc[0] != pd.Timestamp("2020-03-01") or dates.iloc[-1] != pd.Timestamp("2020-08-31") or dates.duplicated().any():
            errors.append(f"invalid policy calendar: {policy}/calendar.csv")


def check_results(errors: list[str]) -> None:
    figure = ROOT / "SEIR" / "figures" / "fig15_seir_policy_compare_four_calendars_10seed.png"
    if figure.is_file():
        width, height = png_size(figure)
        if width < 1000 or height < 500:
            errors.append("ten-seed figure canvas is unexpectedly small")
    table = ROOT / "SEIR" / "results" / "policy_calendar_eta1_reproduction" / "random_seed_table9" / "table9_10_seed_summary.csv"
    if table.is_file() and len(pd.read_csv(table)) != 5:
        errors.append("ten-seed Table 9 summary should contain five policy rows")
    elif not table.is_file():
        errors.append("ten-seed Table 9 summary is missing")


def main() -> None:
    errors: list[str] = []
    check_text(errors)
    check_structure(errors)
    check_calendars(errors)
    check_results(errors)
    result = {"status": "pass" if not errors else "fail", "errors": errors}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()



