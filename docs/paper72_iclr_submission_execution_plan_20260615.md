# Paper 72 ICLR-Main Submission-Readiness Execution Plan

Date: 2026-06-15
Paper: 72 - `adaptive_impedance_tokens`
Target venue posture: ICLR main only if supported by decisive evidence
Current terminal label entering audit: `KILL_ARCHIVE`

## Goal

Rebuild and audit Paper 72 as a real submission candidate rather than a cosmetic manuscript. The audit must decide whether the MuJoCo contact-control evidence can honestly support an ICLR-main adaptive-impedance-token submission, or whether the paper remains a terminal negative result.

## Decision Rule

Upgrade from `KILL_ARCHIVE` only if all of the following are true:

1. `impedance_token_policy` decisively beats the strongest non-oracle baseline, especially `learned_gain_regressor`, on combined stress.
2. The paired confidence interval supports a real positive effect rather than a loss or ambiguous mean.
3. Stress-level results remain favorable under the hardest force/contact settings.
4. Ablations show the claimed token memory and token-update mechanism are necessary; removed-component variants should not beat the full method.
5. Safety, force-error, and chatter metrics do not reveal a hidden regression.
6. The evidence is reproducible from checked-in code, raw CSVs, and a clean PDF build.

If any of these gates fail, preserve `KILL_ARCHIVE` and document the exact failure mode.

## Evidence Gates

Run these checks before changing the decision:

1. Code integrity: compile the experiment source with `python -m py_compile src/run_experiment.py`.
2. Result integrity: verify all required CSVs exist, are nonempty, finite, and schema-valid.
3. Scale check: confirm the recorded evidence includes 7 seeds, 3,360 main rollouts, 420 ablation rollouts, and 2,016 stress rollouts.
4. Baseline check: verify `fixed_impedance`, `gain_scheduled_impedance`, `adaptive_impedance_control`, `admittance_switching_control`, `robust_mpc_impedance`, `learned_gain_regressor`, and `oracle_impedance` are present.
5. Stress check: confirm stress-level results are represented and compare the proposed method against the strongest non-oracle stress baseline.
6. Ablation check: confirm whether `token_full` beats removed-component variants, especially `token_no_memory`.
7. Paper build: run LaTeX/BibTeX to produce a clean PDF and copy only the numbered PDF to `C:/Users/wangz/Downloads/72.pdf`.
8. Artifact hygiene: confirm no numbered PDF is copied to the visible Desktop.
9. GitHub hygiene: confirm the matching public GitHub repository exists and the local commit is pushed.
10. Root-report hygiene: update `GLOBAL_POOL_STATUS.md`, `BATCH_STATUS.md`, `SUBMISSION_STATUS.md`, `MASTER_REPORT.md`, and `MASTER_SUBMISSION_REPORT.md`.

## Expected Risk

The existing evidence summary reports that `impedance_token_policy` reaches 0.488 combined-stress success while `learned_gain_regressor` reaches 0.929, with paired token-minus-learned success difference -0.440 +/- 0.141. Unless direct verification contradicts that result, Paper 72 cannot honestly become submission-ready in this pass.

## Execution Order

1. Re-check repository cleanliness and result inventory.
2. Run code and CSV integrity gates.
3. Rebuild the paper PDF and repair recoverable build warnings.
4. Write a terminal audit with exact evidence and rejection rationale.
5. Update child status, local audit docs, and root reports.
6. Commit and push the Paper 72 repository.
7. Verify `Downloads/72.pdf`, no Desktop copy, public GitHub visibility, clean git state, and root report consistency.

