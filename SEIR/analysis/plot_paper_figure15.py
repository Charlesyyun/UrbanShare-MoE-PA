"""Compatibility entry point for the paper's ten-seed Figure 15."""
from __future__ import annotations

import argparse
from pathlib import Path

from plot_figure12_10seed import main as plot_ten_seed_figure


SEIR_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=SEIR_ROOT / "figures")
    parser.add_argument(
        "--scenario-dir", type=Path,
        help="deprecated; Figure 15 uses the archived ten-seed runs",
    )
    args = parser.parse_args()
    if args.scenario_dir:
        print("--scenario-dir is deprecated and ignored; using archived ten-seed runs.")
    plot_ten_seed_figure(["--output-dir", str(args.output_dir)])


if __name__ == "__main__":
    main()
