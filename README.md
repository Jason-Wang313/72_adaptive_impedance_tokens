# 72 Adaptive Impedance Tokens

Submission-hardening version: v5 expanded ICLR-main kill archive

Terminal decision: KILL_ARCHIVE for ICLR main conference.

This repository contains the expanded Paper 72 rebuild: a CPU-only MuJoCo contact-control audit with implemented impedance, admittance, robust, learned, conformal, token, ablation, and oracle controllers. The evidence does not support submission. The proposed `impedance_token_policy_v5` collapses under the frozen hostile-review gates while simpler classical and learned controllers remain strong.

## Frozen Evidence

Full run:

- Main evaluation rows: 8,640.
- Ablation rows: 1,536.
- Stress rows: 4,320.
- Seed summaries: 1,440.
- Pairwise rows: 168.
- Training examples: 2,400.
- Seeds: 0 through 7.
- Runtime: 4,034.50 seconds.

Headline gate failures:

- Hard splits: `impedance_token_policy_v5` = 0.000 success; best non-oracle baseline `admittance_switching_control` = 0.941.
- Combined/extreme splits: `impedance_token_policy_v5` = 0.000; best non-oracle baseline `conformal_safety_gain` = 0.969.
- Fixed-risk 10 percent budget: `impedance_token_policy_v5` = 0.000; best non-oracle baseline `token_no_memory_ablation` = 0.051.
- Maximum combined-extreme stress: `impedance_token_policy_v5` = 0.000; best non-oracle baseline `adaptive_impedance_control` = 0.958.
- Ablations: full v5 = 0.000; `learned_only_token_replacement` = 0.875.

The paper is retained as a reproducible negative-result archive, not as a submission.

## Reproduce

Frozen full protocol:

```powershell
python src\run_experiment.py --seeds 8 --episodes 6 --ablation-episodes 4 --stress-episodes 3 --train-scenes 2400 --splits nominal_surface_tracking stiffness_shift friction_slip_shift contact_transition target_force_jump actuator_saturation sensor_noise_burst stick_slip_cycle surface_discontinuity delayed_mode_switch combined_stress combined_extreme_stress --ablation-splits combined_stress combined_extreme_stress stick_slip_cycle actuator_saturation --stress-splits combined_stress combined_extreme_stress friction_slip_shift --stress-levels 0.0 0.25 0.5 0.75 1.0 --results-dir results --figures-dir figures --workers 1
```

Rebuild paper assets and PDF:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_submission_pdf.ps1
python scripts\validate_submission_artifacts.py
```

Canonical local PDF: `C:/Users/wangz/Downloads/72.pdf`

No PDF is copied to the visible Desktop.
