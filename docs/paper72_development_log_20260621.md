# Paper 72 Development Log

Date: 2026-06-21

## Expanded Runner Changes

- Replaced the v4 five-split runner with an expanded v5 MuJoCo contact-control runner.
- Added 12 main splits, including combined and combined-extreme stress.
- Added stronger baselines: random forest gain regression, histogram-gradient gain regression, ensemble uncertainty gains, risk-averse impedance, conformal safety gains, token v4, token v5, no-memory token ablation, and oracle.
- Added hostile ablations for memory, discrete tokens, force updates, transition planning, safety penalty, slip context, tail-risk objective, calibration guard, phase memory, and learned-only token replacement.
- Added aggregate metrics, aggregate pairwise statistics, fixed-risk metrics, ablation aggregate metrics, stress summaries, and negative cases.
- Added a frozen gate function that emits `Terminal decision:` and `Terminal reason:`.

## Runtime Fixes Before Freeze

The first tiny probe timed out because random-forest and histogram-gradient predictions were called every physics tick. This was a runner/runtime defect, not an evidence result. The fix was to update learned controllers at a controller-rate interval using cached predictions, reduce the episode tick count to 96, and reduce tree-ensemble sizes while keeping the learned baselines present.

The second probe exposed nested summary-column names in the ablation aggregator and plotting code. Both were fixed.

The third probe showed all-zero hard-split success because the binary success gate was over-tight on transient force overshoot and slip. The success criterion was recalibrated before protocol freeze to require progress, force error, safety, slip, chatter, and bounded transient overshoot without discarding all continuous signal.

## Dev Probe Command

```powershell
python src\run_experiment.py --seeds 2 --episodes 1 --ablation-episodes 1 --stress-episodes 1 --train-scenes 200 --splits combined_stress combined_extreme_stress --ablation-splits combined_stress combined_extreme_stress --stress-splits combined_extreme_stress --stress-levels 0.0 1.0 --results-dir results\dev_probe --figures-dir figures\dev_probe --workers 1
```

## Dev Probe Outcome

The final dev probe completed and produced a KILL_ARCHIVE terminal decision. This is acceptable for development because the purpose was to verify that the runner exposes failures instead of forcing a positive result.

Dev terminal reason:

`v5 does not beat strongest hard-regime baseline admittance_switching_control by 0.030 (v5=0.000, best=1.000); paired lower bound against admittance_switching_control is not positive (-1.000+/-0.000); v5 does not beat strongest combined/extreme baseline admittance_switching_control by 0.030 (v5=0.000, best=1.000); fixed-risk gate fails at budget 0.10 (v5=0.000, best=admittance_switching_control 0.250); maximum-stress gate fails (v5=0.000, best=adaptive_impedance_control 1.000); ablation gate fails because ablate_no_calibration_guard, ablate_no_discrete_tokens, ablate_no_force_update, ablate_no_memory, ablate_no_phase_memory, ablate_no_safety_penalty, ablate_no_slip_context, ablate_no_tail_risk_objective, ablate_no_transition_planner, impedance_token_policy_v4, learned_only_token_replacement matches or beats full v5`

No full-run method, split, seed, threshold, or gate may be changed after the protocol freeze except to fix a crash that prevents artifact generation.
