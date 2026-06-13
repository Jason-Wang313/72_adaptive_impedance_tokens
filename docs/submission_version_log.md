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
