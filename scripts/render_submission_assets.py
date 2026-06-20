from __future__ import annotations

import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PAPER = ROOT / "paper"
GENERATED = PAPER / "generated"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def tex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def label(value: str) -> str:
    return tex_escape(value.replace("_", " "))


def fmt(value: object, digits: int = 3) -> str:
    text = str(value)
    if text.lower() == "inf":
        return r"$\infty$"
    try:
        return f"{float(text):.{digits}f}"
    except ValueError:
        return tex_escape(text)


def table_cell(name: str, value: str) -> str:
    if name in {"method", "split", "comparison", "reference", "group", "lesson"}:
        return label(value)
    if name in {
        "case",
        "seed",
        "episode",
        "episodes",
        "episodes_per_seed",
        "reference_better_seeds",
        "seeds",
        "split_rows",
        "seed_split_rows",
    }:
        try:
            return str(int(float(value)))
        except ValueError:
            return tex_escape(value)
    return fmt(value)


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def parse_summary() -> tuple[str, str]:
    text = (RESULTS / "summary.txt").read_text(encoding="utf-8")
    decision = re.search(r"Terminal decision: (.+)", text)
    reason = re.search(r"Terminal reason: (.+)", text)
    return (
        decision.group(1).strip() if decision else "UNKNOWN",
        reason.group(1).strip() if reason else "No terminal reason found.",
    )


def count_rows(name: str) -> int:
    return len(read_csv(RESULTS / name))


def row_lookup(rows: list[dict[str, str]], **criteria: str) -> dict[str, str]:
    for row in rows:
        if all(row.get(key) == value for key, value in criteria.items()):
            return row
    raise KeyError(criteria)


def aggregate_table(rows: list[dict[str, str]], group: str, label_name: str) -> str:
    selected = [row for row in rows if row["group"] == group]
    selected = sorted(selected, key=lambda row: float(row["success_rate"]), reverse=True)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{label(group)} aggregate results. Higher success is better; lower diagnostic rates are better.}}",
        rf"\label{{{label_name}}}",
        r"\small",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Method & Succ. & Err. & Over. & Safe & Slip & Chat. \\",
        r"\midrule",
    ]
    for row in selected[:12]:
        lines.append(
            f"{label(row['method'])} & {fmt(row['success_rate'])} & {fmt(row['normalized_force_error'])} & "
            f"{fmt(row['peak_overshoot'])} & {fmt(row['safety_violation_rate'])} & "
            f"{fmt(row['slip_rate'])} & {fmt(row['chatter_rate'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def selected_split_table(metrics: list[dict[str, str]], split: str) -> str:
    wanted = {
        "impedance_token_policy_v5",
        "impedance_token_policy_v4",
        "token_no_memory_ablation",
        "admittance_switching_control",
        "adaptive_impedance_control",
        "conformal_safety_gain",
        "random_forest_gain_regressor",
        "learned_gain_regressor",
        "oracle_impedance",
    }
    rows = [row for row in metrics if row["split"] == split and row["method"] in wanted]
    rows = sorted(rows, key=lambda row: float(row["mean_success_rate"]), reverse=True)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Selected methods on {label(split)}.}}",
        r"\label{tab:selected-split}",
        r"\small",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Method & Succ. & Err. & Over. & Safe & Slip & Prog. \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{label(row['method'])} & {fmt(row['mean_success_rate'])} & "
            f"{fmt(row['mean_mean_normalized_force_error'])} & {fmt(row['mean_mean_peak_overshoot'])} & "
            f"{fmt(row['mean_mean_safety_violation_rate'])} & {fmt(row['mean_mean_slip_rate'])} & "
            f"{fmt(row['mean_mean_final_progress'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def fixed_risk_table(rows: list[dict[str, str]]) -> str:
    selected = [row for row in rows if row["budget"] == "0.10"]
    selected = sorted(selected, key=lambda row: float(row["success_at_budget"]), reverse=True)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Fixed-risk operating points at a 10 percent safety/slip/chatter budget over hard splits.}",
        r"\label{tab:fixed-risk}",
        r"\small",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Method & Succ. & Safety & Slip & Chatter \\",
        r"\midrule",
    ]
    for row in selected[:12]:
        lines.append(
            f"{label(row['method'])} & {fmt(row['success_at_budget'])} & "
            f"{fmt(row['mean_safety_violation_rate'])} & {fmt(row['mean_slip_rate'])} & "
            f"{fmt(row['mean_chatter_rate'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def ablation_table(rows: list[dict[str, str]]) -> str:
    selected = sorted(rows, key=lambda row: float(row["success"]), reverse=True)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Token-policy ablations aggregated over frozen ablation splits.}",
        r"\label{tab:ablation}",
        r"\small",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Ablation & Succ. & Err. & Safety & Slip & Chatter \\",
        r"\midrule",
    ]
    for row in selected:
        lines.append(
            f"{label(row['method'])} & {fmt(row['success'])} & {fmt(row['normalized_force_error'])} & "
            f"{fmt(row['safety_violation_rate'])} & {fmt(row['slip_rate'])} & "
            f"{fmt(row['chatter_rate'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def stress_table(rows: list[dict[str, str]]) -> str:
    chosen = {
        "adaptive_impedance_control",
        "conformal_safety_gain",
        "admittance_switching_control",
        "impedance_token_policy_v5",
        "impedance_token_policy_v4",
        "oracle_impedance",
    }
    selected = [
        row
        for row in rows
        if row["method"] in chosen and row["split"] == "combined_extreme_stress"
    ]
    selected = sorted(selected, key=lambda row: (float(row["stress_level"]), row["method"]))
    lines = [
        r"\begin{table}[p]",
        r"\centering",
        r"\caption{Combined-extreme stress sweep for selected controllers.}",
        r"\label{tab:stress}",
        r"\scriptsize",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Method & Level & Succ. & Err. & Over. & Safe & Slip \\",
        r"\midrule",
    ]
    for row in selected:
        lines.append(
            f"{label(row['method'])} & {fmt(row['stress_level'], 2)} & {fmt(row['mean_success_rate'])} & "
            f"{fmt(row['mean_normalized_force_error'])} & {fmt(row['mean_peak_overshoot'])} & "
            f"{fmt(row['mean_safety_violation_rate'])} & {fmt(row['mean_slip_rate'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def negative_cases_table(rows: list[dict[str, str]]) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Representative frozen negative cases for impedance-token policy v5.}",
        r"\label{tab:negative-cases}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.2pt}",
        r"\begin{tabular}{@{}l>{\raggedright\arraybackslash}p{0.18\linewidth}llrrrr>{\raggedright\arraybackslash}p{0.23\linewidth}@{}}",
        r"\toprule",
        r"Case & Split & Seed & Ep. & Succ. & Err. & Over. & Slip & Lesson \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['case']} & {label(row['split'])} & {row['seed']} & {row['episode']} & {row['success']} & "
            f"{fmt(row['normalized_force_error'])} & {fmt(row['peak_overshoot'])} & "
            f"{fmt(row['slip_rate'])} & {label(row['lesson'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def longtable(rows: list[dict[str, str]], columns: list[tuple[str, str, str]], caption: str, label_name: str) -> str:
    # Compact wrapped appendix tables keep the archive exhaustive without wide-page overflow.
    colspec = "@{}" + "".join(column[1] for column in columns) + "@{}"
    header = " & ".join(tex_escape(column[2]) for column in columns) + r" \\"
    lines = [
        r"\begingroup",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{0.65pt}",
        r"\renewcommand{\arraystretch}{0.86}",
        r"\setlength{\LTleft}{0pt}",
        r"\setlength{\LTright}{0pt}",
        rf"\begin{{longtable}}{{{colspec}}}",
        rf"\caption{{{tex_escape(caption)}}}\label{{{label_name}}}\\",
        r"\toprule",
        header,
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        header,
        r"\midrule",
        r"\endhead",
    ]
    for row in rows:
        lines.append(" & ".join(table_cell(name, row.get(name, "")) for name, _, _ in columns) + r" \\")
    lines += [r"\bottomrule", r"\end{longtable}", r"\endgroup", ""]
    return "\n".join(lines)


def render() -> None:
    GENERATED.mkdir(parents=True, exist_ok=True)
    decision, reason = parse_summary()
    metrics = read_csv(RESULTS / "metrics.csv")
    aggregate = read_csv(RESULTS / "aggregate_metrics.csv")
    fixed = read_csv(RESULTS / "fixed_risk_metrics.csv")
    ablation = read_csv(RESULTS / "ablation_aggregate_metrics.csv")
    stress = read_csv(RESULTS / "stress_sweep.csv")
    negative = read_csv(RESULTS / "negative_cases.csv")
    seed = read_csv(RESULTS / "raw_seed_metrics.csv")
    pairwise = read_csv(RESULTS / "pairwise_stats.csv")

    v5 = "impedance_token_policy_v5"
    hard_v5 = row_lookup(aggregate, group="hard_splits", method=v5)
    hard_best = max(
        [row for row in aggregate if row["group"] == "hard_splits" and row["method"] not in {v5, "oracle_impedance"}],
        key=lambda row: float(row["success_rate"]),
    )
    combined_v5 = row_lookup(aggregate, group="combined_and_extreme", method=v5)
    combined_best = max(
        [
            row
            for row in aggregate
            if row["group"] == "combined_and_extreme" and row["method"] not in {v5, "oracle_impedance"}
        ],
        key=lambda row: float(row["success_rate"]),
    )
    fixed_v5 = row_lookup(fixed, budget="0.10", method=v5)
    fixed_best = max(
        [row for row in fixed if row["budget"] == "0.10" and row["method"] not in {v5, "oracle_impedance"}],
        key=lambda row: float(row["success_at_budget"]),
    )
    full_ablation = row_lookup(ablation, method="token_full_v5")
    best_ablation = max(ablation, key=lambda row: float(row["success"]))
    max_stress_v5 = row_lookup(stress, split="combined_extreme_stress", stress_level="1.00", method=v5)
    max_stress_best = max(
        [
            row
            for row in stress
            if row["split"] == "combined_extreme_stress"
            and row["stress_level"] == "1.00"
            and row["method"] not in {v5, "oracle_impedance"}
        ],
        key=lambda row: float(row["mean_success_rate"]),
    )

    text_col = r">{\raggedright\arraybackslash}p{%s\linewidth}"
    num_col = r">{\raggedleft\arraybackslash}p{%s\linewidth}"

    macros = rf"""
\newcommand{{\PaperDecision}}{{{tex_escape(decision)}}}
\newcommand{{\PaperReason}}{{{tex_escape(reason)}}}
\newcommand{{\TrainingRows}}{{{count_rows('training_impedance_examples.csv'):,}}}
\newcommand{{\MainRows}}{{{count_rows('impedance_raw.csv'):,}}}
\newcommand{{\RolloutRows}}{{{count_rows('impedance_rollouts.csv'):,}}}
\newcommand{{\SeedRows}}{{{count_rows('raw_seed_metrics.csv'):,}}}
\newcommand{{\MetricRows}}{{{count_rows('metrics.csv'):,}}}
\newcommand{{\PairwiseRows}}{{{count_rows('pairwise_stats.csv'):,}}}
\newcommand{{\AblationRows}}{{{count_rows('impedance_ablation_raw.csv'):,}}}
\newcommand{{\StressRows}}{{{count_rows('stress_sweep_raw.csv'):,}}}
\newcommand{{\NegativeRows}}{{{count_rows('negative_cases.csv'):,}}}
\newcommand{{\VFiveHardSuccess}}{{{fmt(hard_v5['success_rate'])}}}
\newcommand{{\BestHardMethod}}{{{label(hard_best['method'])}}}
\newcommand{{\BestHardSuccess}}{{{fmt(hard_best['success_rate'])}}}
\newcommand{{\VFiveCombinedSuccess}}{{{fmt(combined_v5['success_rate'])}}}
\newcommand{{\BestCombinedMethod}}{{{label(combined_best['method'])}}}
\newcommand{{\BestCombinedSuccess}}{{{fmt(combined_best['success_rate'])}}}
\newcommand{{\VFiveFixedRiskSuccess}}{{{fmt(fixed_v5['success_at_budget'])}}}
\newcommand{{\BestFixedRiskMethod}}{{{label(fixed_best['method'])}}}
\newcommand{{\BestFixedRiskSuccess}}{{{fmt(fixed_best['success_at_budget'])}}}
\newcommand{{\FullAblationSuccess}}{{{fmt(full_ablation['success'])}}}
\newcommand{{\BestAblationMethod}}{{{label(best_ablation['method'])}}}
\newcommand{{\BestAblationSuccess}}{{{fmt(best_ablation['success'])}}}
\newcommand{{\VFiveMaxStressSuccess}}{{{fmt(max_stress_v5['mean_success_rate'])}}}
\newcommand{{\BestMaxStressMethod}}{{{label(max_stress_best['method'])}}}
\newcommand{{\BestMaxStressSuccess}}{{{fmt(max_stress_best['mean_success_rate'])}}}
""".strip()
    write(GENERATED / "result_macros.tex", macros + "\n")
    write(GENERATED / "hard_aggregate_table.tex", aggregate_table(aggregate, "hard_splits", "tab:hard-aggregate"))
    write(
        GENERATED / "combined_aggregate_table.tex",
        aggregate_table(aggregate, "combined_and_extreme", "tab:combined-aggregate"),
    )
    write(GENERATED / "selected_split_table.tex", selected_split_table(metrics, "combined_extreme_stress"))
    write(GENERATED / "fixed_risk_table.tex", fixed_risk_table(fixed))
    write(GENERATED / "ablation_table.tex", ablation_table(ablation))
    write(GENERATED / "stress_table.tex", stress_table(stress))
    write(GENERATED / "negative_cases_table.tex", negative_cases_table(negative))

    write(
        GENERATED / "full_metrics_longtable.tex",
        longtable(
            metrics,
            [
                ("method", text_col % "0.160", "Method"),
                ("split", text_col % "0.150", "Split"),
                ("mean_success_rate", num_col % "0.050", "Succ."),
                ("ci95_success_rate", num_col % "0.050", "CI"),
                ("mean_mean_normalized_force_error", num_col % "0.055", "Err."),
                ("mean_mean_peak_overshoot", num_col % "0.055", "Over."),
                ("mean_mean_safety_violation_rate", num_col % "0.055", "Safe"),
                ("mean_mean_slip_rate", num_col % "0.055", "Slip"),
                ("mean_mean_chatter_rate", num_col % "0.055", "Chat."),
                ("mean_mean_final_progress", num_col % "0.055", "Prog."),
            ],
            "Full split-method metrics.",
            "tab:full-metrics",
        ),
    )
    write(
        GENERATED / "full_aggregate_longtable.tex",
        longtable(
            aggregate,
            [
                ("group", text_col % "0.155", "Group"),
                ("method", text_col % "0.175", "Method"),
                ("success_rate", num_col % "0.050", "Succ."),
                ("ci95_success_rate", num_col % "0.050", "CI"),
                ("normalized_force_error", num_col % "0.055", "Err."),
                ("peak_overshoot", num_col % "0.055", "Over."),
                ("safety_violation_rate", num_col % "0.055", "Safe"),
                ("slip_rate", num_col % "0.055", "Slip"),
                ("chatter_rate", num_col % "0.055", "Chat."),
                ("energy", num_col % "0.055", "Energy"),
            ],
            "Full aggregate metrics.",
            "tab:full-aggregate",
        ),
    )
    write(
        GENERATED / "all_seed_metrics_longtable.tex",
        longtable(
            seed,
            [
                ("method", text_col % "0.145", "Method"),
                ("split", text_col % "0.130", "Split"),
                ("seed", num_col % "0.035", "Seed"),
                ("success_rate", num_col % "0.055", "Succ."),
                ("mean_normalized_force_error", num_col % "0.055", "Err."),
                ("mean_peak_overshoot", num_col % "0.055", "Over."),
                ("mean_safety_violation_rate", num_col % "0.055", "Safe"),
                ("mean_slip_rate", num_col % "0.055", "Slip"),
                ("mean_chatter_rate", num_col % "0.055", "Chat."),
                ("mean_settling_latency", num_col % "0.055", "Lat."),
                ("mean_final_progress", num_col % "0.055", "Prog."),
            ],
            "All seed-level metrics.",
            "tab:all-seeds",
        ),
    )
    write(
        GENERATED / "full_pairwise_longtable.tex",
        longtable(
            pairwise,
            [
                ("split", text_col % "0.160", "Split"),
                ("comparison", text_col % "0.220", "Comparison"),
                ("paired_success_diff", num_col % "0.055", "dSucc."),
                ("ci95_success_diff", num_col % "0.055", "CI"),
                ("paired_force_error_reduction", num_col % "0.055", "dErr."),
                ("paired_safety_reduction", num_col % "0.055", "dSafe"),
                ("paired_slip_reduction", num_col % "0.055", "dSlip"),
                ("paired_chatter_reduction", num_col % "0.055", "dChat."),
                ("reference_better_seeds", num_col % "0.040", "Wins"),
            ],
            "Full paired seed differences versus impedance-token policy v5.",
            "tab:full-pairwise",
        ),
    )
    write(
        GENERATED / "full_ablation_longtable.tex",
        longtable(
            read_csv(RESULTS / "ablation_metrics.csv"),
            [
                ("method", text_col % "0.190", "Method"),
                ("split", text_col % "0.160", "Split"),
                ("mean_success_rate", num_col % "0.050", "Succ."),
                ("mean_mean_normalized_force_error", num_col % "0.055", "Err."),
                ("mean_mean_peak_overshoot", num_col % "0.055", "Over."),
                ("mean_mean_safety_violation_rate", num_col % "0.055", "Safe"),
                ("mean_mean_slip_rate", num_col % "0.055", "Slip"),
                ("mean_mean_chatter_rate", num_col % "0.055", "Chat."),
            ],
            "Full ablation metrics.",
            "tab:full-ablation",
        ),
    )
    write(
        GENERATED / "full_stress_longtable.tex",
        longtable(
            stress,
            [
                ("method", text_col % "0.170", "Method"),
                ("split", text_col % "0.145", "Split"),
                ("stress_level", num_col % "0.040", "Lvl"),
                ("mean_success_rate", num_col % "0.050", "Succ."),
                ("mean_normalized_force_error", num_col % "0.050", "Err."),
                ("mean_peak_overshoot", num_col % "0.050", "Over."),
                ("mean_safety_violation_rate", num_col % "0.050", "Safe"),
                ("mean_slip_rate", num_col % "0.050", "Slip"),
                ("mean_chatter_rate", num_col % "0.050", "Chat."),
                ("mean_final_progress", num_col % "0.050", "Prog."),
            ],
            "Full stress sweep metrics.",
            "tab:full-stress",
        ),
    )

    terminal = f"""# Paper 72 Expanded Terminal Decision

Decision: {decision}

Reason: {reason}

Training rows: {count_rows('training_impedance_examples.csv')}
Main method-evaluation rows: {count_rows('impedance_raw.csv')}
Rollout rows: {count_rows('impedance_rollouts.csv')}
Seed summary rows: {count_rows('raw_seed_metrics.csv')}
Split-method metric rows: {count_rows('metrics.csv')}
Pairwise rows: {count_rows('pairwise_stats.csv')}
Ablation rows: {count_rows('impedance_ablation_raw.csv')}
Stress rows: {count_rows('stress_sweep_raw.csv')}
Negative cases: {count_rows('negative_cases.csv')}

This decision is generated from frozen CSV artifacts, not hand-transcribed table values.
"""
    write(ROOT / "docs" / "paper72_expanded_terminal_decision_20260621.md", terminal)


if __name__ == "__main__":
    render()
