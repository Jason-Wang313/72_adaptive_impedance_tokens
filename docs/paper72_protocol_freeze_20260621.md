# Paper 72 Protocol Freeze

Date: 2026-06-21

This protocol is frozen before the expanded full run.

## Frozen Command

```powershell
python src\run_experiment.py --seeds 8 --episodes 6 --ablation-episodes 4 --stress-episodes 3 --train-scenes 2400 --splits nominal_surface_tracking stiffness_shift friction_slip_shift contact_transition target_force_jump actuator_saturation sensor_noise_burst stick_slip_cycle surface_discontinuity delayed_mode_switch combined_stress combined_extreme_stress --ablation-splits combined_stress combined_extreme_stress stick_slip_cycle actuator_saturation --stress-splits combined_stress combined_extreme_stress friction_slip_shift --stress-levels 0.0 0.25 0.5 0.75 1.0 --results-dir results --figures-dir figures --workers 1
```

## Frozen Scale

- Main rows: 12 splits x 8 seeds x 6 episodes x 15 methods = 8,640.
- Ablation rows: 4 ablation splits x 8 seeds x 4 episodes x 12 methods = 1,536.
- Stress rows: 3 stress splits x 5 stress levels x 8 seeds x 3 episodes x 12 methods = 4,320.
- Training examples: 2,400.

## Frozen Decision Gates

The terminal decision can only improve if all core gates pass:

- Hard-regime gate: `impedance_token_policy_v5` beats the strongest non-oracle hard-regime baseline by at least 0.030 success.
- Paired gate: paired lower bound against the strongest hard-regime baseline is positive.
- Combined/extreme gate: v5 beats the strongest non-oracle combined/extreme baseline by at least 0.030 success.
- Fixed-risk gate: at safety budget 0.10, v5 fixed-risk success beats all non-oracle baselines by at least 0.030.
- Maximum-stress gate: v5 does not lose the highest stress level to any non-oracle baseline.
- Ablation gate: no core ablation matches or beats full v5 on the predefined ablation aggregate.
- Safety gate: v5 cannot trade success for unacceptable safety, slip, or chatter.

If any frozen gate fails, report the failure honestly in `results/summary.txt`, the manuscript, and root ledgers.

## Frozen Artifact Rules

- Final PDF must be `C:/Users/wangz/Downloads/72.pdf`.
- No `C:/Users/wangz/Desktop/72.pdf`.
- Generated tables must be built from final CSVs.
- Bright boxed citation links must be enabled in the PDF.
- Visual PDF QA must inspect rendered PNG pages before final commit.
