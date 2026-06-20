# Submission Readiness Decision

Decision: KILL_ARCHIVE

ICLR main-conference readiness: NO.

Submission-hardening version: v5 expanded archive.

## Evidence Used

The v5 rebuild expands the real MuJoCo contact-control benchmark into a hostile-review audit. It includes fixed impedance, gain scheduling, adaptive impedance, admittance switching, robust MPC-style impedance, learned gain regression, random-forest gain regression, histogram-gradient gain regression, uncertainty ensemble, risk-averse impedance, conformal safety gain, previous token replay, full token v5, no-memory token ablation, and oracle diagnostics.

The frozen run contains 8,640 main rows, 1,536 ablation rows, 4,320 stress rows, 1,440 seed summaries, 168 paired comparisons, fixed-risk analysis, stress sweeps, figures, negative cases, and a 54-page PDF with boxed clickable citations.

## Gate Result

Frozen gates all fail:

- Hard splits: `impedance_token_policy_v5` = 0.000 success; strongest non-oracle baseline `admittance_switching_control` = 0.941.
- Paired lower bound against `admittance_switching_control`: not positive, reported as -0.941 +/- 0.025.
- Combined/extreme splits: `impedance_token_policy_v5` = 0.000; strongest non-oracle baseline `conformal_safety_gain` = 0.969.
- Fixed-risk 10 percent budget: `impedance_token_policy_v5` = 0.000; `token_no_memory_ablation` = 0.051.
- Maximum combined-extreme stress: `impedance_token_policy_v5` = 0.000; `adaptive_impedance_control` = 0.958.
- Ablations: full v5 = 0.000; `learned_only_token_replacement` = 0.875.

The proposed token mechanism is not submission-ready. The result is not a near miss; it is a clear falsification under the frozen evidence package.

## Terminal Action

Archive/kill for ICLR main. Do not submit this paper as an ICLR main paper.

Revival condition: redesign the token mechanism, add a credible change-point or continuous fallback strategy, validate on real robot or public contact-rich benchmarks, and show statistically reliable gains over conformal, learned, adaptive, and admittance baselines under a frozen protocol.
