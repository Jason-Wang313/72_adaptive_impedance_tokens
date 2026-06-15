# Paper 72 Terminal Audit - 2026-06-15

Paper: `adaptive_impedance_tokens`
Decision: `KILL_ARCHIVE`
ICLR-main ready: no

## Verification Performed

1. Source compile gate passed with `python -m py_compile src/run_experiment.py`.
2. CSV integrity gate passed for all result CSVs: files are present, nonempty, finite, and schema-readable. Blank `stress_level` values are expected only in non-stress rollout tables.
3. Evidence scale matched the reported claims:
   - Main rollouts: 3,360
   - Ablation rollouts: 420
   - Stress rollouts: 2,016
   - Seeds: 0, 1, 2, 3, 4, 5, 6
4. Baselines were present in the main evidence: `fixed_impedance`, `gain_scheduled_impedance`, `adaptive_impedance_control`, `admittance_switching_control`, `robust_mpc_impedance`, `learned_gain_regressor`, and `oracle_impedance`.
5. PDF rebuild completed and `C:/Users/wangz/Downloads/72.pdf` was refreshed.
6. BibTeX sort warnings were repaired by adding stable `key` fields to the local reference entries.
7. No visible Desktop copy of `72.pdf` was present after the audit.

## Fatal Evidence

The proposed token mechanism fails the ICLR-main decision rule decisively. On combined stress, `impedance_token_policy` reaches 0.488 success, while `learned_gain_regressor` reaches 0.929. The paired token-minus-learned success difference is -0.440 +/- 0.141.

The token policy also loses to a simpler classical-style baseline: `gain_scheduled_impedance` reaches 0.738 success, with token-minus-gain-scheduled paired success difference -0.250 +/- 0.050.

The ablation evidence falsifies the claimed token-memory mechanism. On the ablation combined-stress grid, `token_full` reaches 0.457 success while `token_no_memory` reaches 0.600. Removing memory improves performance, so the central mechanism is not supported.

At stress level 1.00, both `impedance_token_policy` and several non-oracle baselines collapse to 0.000 success, while `oracle_impedance` reaches 0.589. This confirms the local benchmark is hard, but it does not rescue the token policy.

## Decision

Paper 72 remains `KILL_ARCHIVE`. It is a reproducible negative result, not an honest ICLR-main submission candidate.

## Revival Requirements

To revive this paper, a future version would need a redesigned token mechanism that beats learned gain adaptation and gain scheduling under combined stress, demonstrates that memory and token transitions help in ablations, and validates contact-rich impedance control on hardware or a public benchmark.

