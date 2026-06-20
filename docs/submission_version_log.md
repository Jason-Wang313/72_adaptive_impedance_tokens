# Submission Version Log

## v1 - Generated Draft

- Original continuation-batch generated paper and toy single-seed experiment.

## v2 - Submission Hardening

- Added hostile reviewer attack log and response docs.
- Replaced the toy experiment with seven-seed synthetic metrics, stronger synthetic baselines, ablations, stress tests, and negative cases.
- Narrowed claims to diagnostic evidence.
- Terminal decision: WORKSHOP_ONLY.

## v3 - ICLR Main Gate Archive

- Applied the stricter ICLR-main-conference standard.
- Determined that the existing local artifacts were insufficient for main-track submission.
- Recompiled the canonical PDF with `Submission-hardening version: v3`.
- Terminal decision: KILL_ARCHIVE.

## v4 - Real MuJoCo Rebuild

- Replaced the synthetic scaffold with a MuJoCo contact-control benchmark.
- Implemented fixed impedance, gain scheduling, adaptive impedance, admittance switching, robust MPC-style impedance, trained learned gain regression, impedance tokens, and oracle impedance.
- Ran seven seeds, 12 episodes per seed/split, ablations, stress sweeps, uncertainty intervals, paired comparisons, figures, and negative cases.
- Found that `impedance_token_policy` loses to `learned_gain_regressor` on combined stress: 0.488 +/- 0.120 versus 0.929 +/- 0.056 success.
- Terminal decision remains: KILL_ARCHIVE.

## v5 - Expanded ICLR-Main Kill Archive

- Expanded the benchmark to 8 seeds, 12 main splits, 15 main methods, 12 ablation methods, 12 stress methods, 8,640 main rows, 1,536 ablation rows, 4,320 stress rows, 1,440 seed summaries, and 168 paired comparisons.
- Added stronger learned, random-forest, histogram-gradient, uncertainty, risk-averse, conformal, no-memory, and oracle comparisons.
- Froze a hostile-review protocol before the full run, including hard-split, paired, combined/extreme, fixed-risk, max-stress, and ablation-necessity gates.
- Found that `impedance_token_policy_v5` reaches 0.000 success on hard and combined/extreme aggregates while `admittance_switching_control` reaches 0.941 on hard splits and `conformal_safety_gain` reaches 0.969 on combined/extreme splits.
- Found that full v5 reaches 0.000 ablation success while `learned_only_token_replacement` reaches 0.875.
- Rebuilt a 54-page ICLR-style PDF with bright boxed citations and exhaustive generated appendices.
- Terminal decision remains: KILL_ARCHIVE.
