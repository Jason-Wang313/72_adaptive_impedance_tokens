# Paper 72 Expanded Submission Plan

Date: 2026-06-21

Target venue standard: ICLR main-conference hostile-review readiness.

Current terminal state before this pass: KILL_ARCHIVE.

Core rule: do not optimize for pretty results. Optimize for a result that survives hostile review. Use strong baselines and stress tests to expose weaknesses, improve the method during development, then freeze the final protocol and report all predefined results honestly.

## Goal

Rebuild Paper 72 from a four-page v4 negative archive into a 25+ page submission-style evidence package for adaptive impedance tokens in contact-rich control. The rebuilt paper may only become STRONG_REVISE if the token mechanism survives strong learned, classical, robust, risk-aware, and ablated competitors under frozen gates. If it fails, produce a polished negative archive with complete evidence, theory, clickable bright-box citations, and a Downloads-only numbered PDF.

## Current Weaknesses

- The current PDF is only 4 pages and cannot survive a serious submission-readiness review.
- The current evaluation has only five splits and one combined-stress regime.
- The token policy loses badly to `learned_gain_regressor` on the v4 decisive split.
- The ablation table already suggests that `token_no_memory` can beat the full token mechanism.
- The paper lacks a formal theory section that distinguishes discrete impedance-token memory from ordinary continuous gain adaptation.
- The bibliography is too weak for submission polish and needs real robotics/control references.
- The current PDF has no generated appendix tables, no fixed-risk/contact-safety gates, and no bright boxed citation links.

## Expanded Research Question

Can discrete impedance tokens with contact-outcome memory provide a necessary and empirically superior representation for contact-rich force control, compared with continuous learned gain adaptation, classical adaptive impedance/admittance, robust safety-biased controllers, conformal/risk filters, ensembles, and oracle upper bounds?

The null hypothesis is strong: a continuous learned or risk-aware gain adapter can absorb the useful signal without discrete token memory.

## Method Development Scope

Allowed development before freeze:

- Add a better v5 token method with bounded memory, contact-phase context, slip-aware transitions, tail-risk penalties, and calibration-aware token scores.
- Add CPU-light learned baselines such as ridge, random forest, histogram-gradient regressors, ensemble uncertainty, and/or residual online calibration.
- Add stronger classical controllers: variable impedance, adaptive admittance/impedance switching, robust conservative impedance, passivity/safety-filter variants, and gain-scheduled controls.
- Add diagnostics: force error, post-shift error, overshoot, safety violation, slip, chatter, energy, settling latency, progress, token switches, calibration error, fixed-risk success, and max-stress success.
- Add stress regimes that expose stiffness jumps, friction jumps, delayed contact transitions, target-force changes, actuator saturation, sensor noise, stick-slip, surface discontinuity, and combined extreme shifts.

Not allowed after freeze:

- Changing splits, seed count, gates, thresholds, methods, or reported metrics to make results prettier.
- Removing hostile baselines or ablations because they win.
- Hiding failed predefined results.

## CPU and RAM Policy

- CPU only.
- One worker by default.
- Keep memory light by streaming rows and using shallow scikit-learn models.
- No deep neural networks or GPU-required training.
- Do not compromise evidence quality for RAM; instead reduce memory by deterministic row generation, compact CSVs, and simple model classes.

## Required Experiment Matrix

Minimum expanded final matrix:

- At least 12 main methods including weak baselines, strong classical controls, strong learned controls, token v4, token v5, ablations-as-methods where relevant, risk/safety variants, and oracle.
- At least 12 contact splits including nominal, stiffness shift, friction shift, contact transition, target-force shift, actuator saturation, sensor-noise burst, stick-slip, surface discontinuity, delayed mode switch, combined stress, and combined extreme stress.
- At least 8 seeds.
- At least 6 episodes per seed/split/method for main evaluation.
- At least 4 ablation episodes per seed/split for multiple hostile ablation splits.
- At least 5 stress levels for several stress splits.
- At least 2,400 synthetic training examples for learned gain baselines.

## Required Baselines

The final run must include, if implementation remains numerically stable:

- `fixed_impedance`
- `gain_scheduled_impedance`
- `adaptive_impedance_control`
- `admittance_switching_control`
- `robust_mpc_impedance`
- `learned_gain_regressor`
- `random_forest_gain_regressor`
- `hist_gradient_gain_regressor`
- `ensemble_uncertainty_gain`
- `risk_averse_impedance`
- `conformal_safety_gain`
- `impedance_token_policy_v4`
- `impedance_token_policy_v5`
- `token_no_memory_ablation`
- `oracle_impedance`

## Required Ablations

The final ablation table must test whether the mechanism is necessary:

- full v5 token method
- no token memory
- no discrete tokens
- no force update
- no transition planner
- no safety penalty
- no slip context
- no tail-risk objective
- no calibration guard
- no phase memory
- token v4
- learned-only token replacement

## Frozen Decision Gates

After the dev probe and before the full run, freeze a protocol file with exact command and thresholds.

The final terminal decision can be STRONG_REVISE only if all gates pass:

- Hard-regime gate: v5 token success beats the strongest non-oracle baseline by at least 0.030 on hard regimes.
- Paired gate: paired lower bound against the strongest hard-regime baseline is positive.
- Combined/extreme gate: v5 token success beats the strongest non-oracle baseline by at least 0.030 on combined/extreme regimes.
- Fixed-risk gate: at the 0.10 safety-violation budget, v5 token fixed-risk success beats all non-oracle baselines.
- Max-stress gate: v5 token does not lose the highest stress level to a non-oracle baseline.
- Ablation-necessity gate: every core removal is worse than full v5 on the predefined ablation aggregate.
- Safety gate: v5 token does not buy success by increasing safety violation, slip, or chatter beyond predefined tolerances.

If any gate fails, the correct terminal decision is KILL_ARCHIVE or STRONG_REVISE only if the failure is narrow and the remaining evidence is genuinely promising. No ICLR-main-ready claim is allowed without all core mechanism and validation gates.

## Paper Requirements

- 25+ pages without padding.
- New theory section explaining when discrete impedance-token memory can help and when it is not identifiable beyond continuous gain adaptation.
- Expanded methods section with controller equations and contact metrics.
- Generated tables directly from final CSVs.
- Appendix with full split-method, seed-level, paired, ablation, stress, and negative-case tables.
- Bright boxed clickable citations using `hyperref` border settings.
- Real robotics/control references, not placeholder prior-work tokens.
- Terminal decision stated honestly in abstract and conclusion.

## Artifact Requirements

- Final PDF path: `C:/Users/wangz/Downloads/72.pdf`.
- No `C:/Users/wangz/Desktop/72.pdf`.
- Public GitHub repo remains: `https://github.com/Jason-Wang313/72_adaptive_impedance_tokens`.
- Update `README.md`, `child_status.md`, `docs/submission_readiness_decision.md`, root `GLOBAL_POOL_STATUS.md`, `BATCH_STATUS.md`, `MASTER_REPORT.md`, `SUBMISSION_STATUS.md`, `MASTER_SUBMISSION_REPORT.md`, and `SUBMISSION_AUDIT_MATRIX.csv`.

## Execution Sequence

1. Patch ignore rules for temporary outputs and full-run logs.
2. Expand `src/run_experiment.py` with v5 methods, baselines, splits, metrics, fixed-risk analysis, stress aggregation, and frozen gate output.
3. Run a tiny dev probe to catch crashes and verify the gates can fail honestly.
4. Write the protocol-freeze document with exact full-run command.
5. Run the full frozen CPU-only experiment.
6. Generate manuscript assets from final CSVs.
7. Rebuild the PDF to `Downloads/72.pdf`.
8. Validate counts, links, PDF placement, Desktop hygiene, GitHub URL, and final logs.
9. Render PDF pages to PNGs, inspect title/results/appendix/references pages, and clean temporary renders.
10. Commit, push, verify public repo, then update root ledgers.
