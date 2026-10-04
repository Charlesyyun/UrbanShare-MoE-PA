"""Validate the code-only public release."""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT_EXTENSIONS = {".py", ".md", ".txt", ".json", ".tex", ".sh", ".yml", ".yaml", ".toml", ".ini"}
REQUIRED = (
    "README.md",
    "DATA.md",
    "LICENSE",
    "CONTRIBUTING.md",
    "docs/ARCHITECTURE.md",
    "docs/WORKFLOW.md",
    "docs/PAPER_ALIGNMENT.md",
    "daily_share_model.py",
    "build_enriched_timeshare.py",
    "preference_dataset.py",
    "preference_scorer.py",
    "preference_alignment_finetune.py",
    "SEIR/seir_timeseries.py",
    "SEIR/calibration/auto_calibrate_seir.py",
    "SEIR/analysis/run_10_seed_table9.py",
    "SEIR/analysis/plot_figure12_10seed.py",
)
FORBIDDEN = (
    ("absolute Windows path", re.compile(r"[A-Za-z]:\\(?:Users|Study|Research)\\", re.I)),
    ("Linux private path", re.compile(r"/(?:root|home)/", re.I)),
    ("compute-environment name", re.compile(r"auto" + r"dl", re.I)),
    ("GitHub token", re.compile(r"gh" + r"_[pousr]_[A-Za-z0-9]{20,}")),
    ("private key", re.compile(r"BEGIN " + "PRIVATE KEY", re.I)),
)


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
        for label, pattern in FORBIDDEN:
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
            errors.append(f"missing required code file: {rel}")
    forbidden_names = {"baselines", "MoE-PA-ablations", "dataset", "results", "figures", "inputs"}
    for path in ROOT.rglob("*"):
        if path.is_dir() and path.name in forbidden_names:
            errors.append(f"data/result directory remains: {path.relative_to(ROOT)}")
        if path.is_file() and path.suffix.lower() in {".pt", ".pth", ".ckpt", ".safetensors", ".bin", ".csv", ".csv.gz"}:
            errors.append(f"data/result file remains: {path.relative_to(ROOT)}")


def main() -> None:
    errors: list[str] = []
    check_text(errors)
    check_structure(errors)
    result = {"status": "pass" if not errors else "fail", "errors": errors}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

