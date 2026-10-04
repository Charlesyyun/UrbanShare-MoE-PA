"""Export the frozen paper values used in Table 6 as CSV and LaTeX."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

SEIR_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=SEIR_ROOT / "figures")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({
        "Policy": ["Original policy", "Early lockdown", "Late lockdown", "Short lockdown", "Long lockdown"],
        "Peak I": [83, 59, 91, 86, 89],
        "Peak new E": [11.86, 8.43, 13.00, 12.29, 12.71],
        "Final R": [305, 231, 368, 325, 319],
        "Area(I)": [2133, 1615, 2574, 2273, 2231],
        "Area(new E)": [303, 229, 366, 323, 317],
        "O": [253.12, 247.00, 253.11, 255.79, 246.95],
        "Change in O (%)": ["--", -2.42, -0.01, 1.06, -2.44],
    })
    csv_path = args.output_dir / "table6_epidemic_activity_tradeoffs_paper.csv"
    tex_path = args.output_dir / "table6_epidemic_activity_tradeoffs_paper.tex"
    frame.to_csv(csv_path, index=False)
    tex_path.write_text(frame.to_latex(index=False, escape=True, caption="Epidemic and activity trade-offs.", label="tab:policy_tradeoff_models"), encoding="utf-8")
    print(f"[ok] {csv_path}")
    print(f"[ok] {tex_path}")


if __name__ == "__main__":
    main()
