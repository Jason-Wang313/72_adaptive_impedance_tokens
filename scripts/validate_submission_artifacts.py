from __future__ import annotations

import csv
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
PAPER = ROOT / "paper"
DOWNLOADS_PDF = Path(r"C:\Users\wangz\Downloads\72.pdf")
DESKTOP_PDF = Path(r"C:\Users\wangz\Desktop\72.pdf")

EXPECTED_ROWS = {
    "training_impedance_examples.csv": 2400,
    "training_summary.csv": 1,
    "impedance_raw.csv": 8640,
    "impedance_rollouts.csv": 8640,
    "raw_seed_metrics.csv": 1440,
    "metrics.csv": 180,
    "impedance_metrics.csv": 180,
    "pairwise_stats.csv": 168,
    "impedance_pairwise.csv": 168,
    "aggregate_metrics.csv": 45,
    "aggregate_pairwise_stats.csv": 28,
    "fixed_risk_metrics.csv": 45,
    "impedance_ablation_raw.csv": 1536,
    "ablation_metrics.csv": 48,
    "impedance_ablation.csv": 48,
    "ablation_aggregate_metrics.csv": 12,
    "stress_sweep_raw.csv": 4320,
    "stress_sweep.csv": 180,
    "negative_cases.csv": 12,
}

REQUIRED_FIGURES = [
    "impedance_success_by_split.png",
    "impedance_force_error_by_split.png",
    "impedance_safety_by_split.png",
    "impedance_ablation_success.png",
    "impedance_stress_sweep.png",
    "stress_curve_data.csv",
]

REQUIRED_GENERATED = [
    "result_macros.tex",
    "hard_aggregate_table.tex",
    "combined_aggregate_table.tex",
    "selected_split_table.tex",
    "fixed_risk_table.tex",
    "ablation_table.tex",
    "stress_table.tex",
    "negative_cases_table.tex",
    "full_metrics_longtable.tex",
    "full_aggregate_longtable.tex",
    "all_seed_metrics_longtable.tex",
    "full_pairwise_longtable.tex",
    "full_ablation_longtable.tex",
    "full_stress_longtable.tex",
]


def row_count(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.DictReader(f))


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def pdf_pages(path: Path) -> int:
    proc = subprocess.run(["pdfinfo", str(path)], check=True, capture_output=True, text=True)
    for line in proc.stdout.splitlines():
        if line.startswith("Pages:"):
            return int(line.split(":", 1)[1].strip())
    raise AssertionError("pdfinfo did not report page count")


def main() -> None:
    summary = (RESULTS / "summary.txt").read_text(encoding="utf-8")
    assert_true("Terminal decision: KILL_ARCHIVE" in summary, "terminal decision is not KILL_ARCHIVE")

    for name, expected in EXPECTED_ROWS.items():
        actual = row_count(RESULTS / name)
        assert_true(actual == expected, f"{name}: expected {expected} rows, found {actual}")

    for name in REQUIRED_FIGURES:
        path = FIGURES / name
        assert_true(path.exists() and path.stat().st_size > 0, f"missing or empty figure: {name}")

    for name in REQUIRED_GENERATED:
        path = PAPER / "generated" / name
        assert_true(path.exists() and path.stat().st_size > 0, f"missing generated TeX asset: {name}")

    main_tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    assert_true("citebordercolor={0 1 0}" in main_tex, "bright green citation boxes are not configured")
    assert_true("pdfborder={0 0 1.8}" in main_tex, "visible citation/link box width is not configured")
    assert_true("generated/result_macros.tex" in main_tex, "main.tex does not include generated result macros")
    assert_true("KILL" in main_tex and "ARCHIVE" in main_tex, "main.tex does not foreground the terminal decision")

    remote = subprocess.run(
        ["git", "remote", "get-url", "origin"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert_true("72_adaptive_impedance_tokens" in remote, f"unexpected Git remote: {remote}")

    assert_true(DOWNLOADS_PDF.exists(), "Downloads PDF is missing")
    assert_true(not DESKTOP_PDF.exists(), "Desktop PDF copy exists")
    pages = pdf_pages(DOWNLOADS_PDF)
    assert_true(pages >= 25, f"PDF is only {pages} pages")

    print("Paper 72 validation passed: counts, figures, TeX links, Downloads PDF, repo URL, and Desktop hygiene are OK.")


if __name__ == "__main__":
    main()
