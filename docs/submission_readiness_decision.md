# Submission Readiness Decision

Decision: KILL_ARCHIVE

ICLR main-conference readiness: NO.

Submission-hardening version: v4.

## Evidence Used

The v4 rebuild replaced the old synthetic scaffold with a MuJoCo contact-control benchmark. It includes implemented controllers, a trained learned gain regressor, the proposed token controller, an oracle controller, seven seeds, 12 evaluation episodes per seed/split, ablations, stress sweeps, uncertainty intervals, pairwise statistics, figures, and negative cases.

## Gate Result

On the decisive `combined_stress` split:

- `learned_gain_regressor`: 0.929 +/- 0.056 success.
- `oracle_impedance`: 0.881 +/- 0.079 success.
- `gain_scheduled_impedance`: 0.738 +/- 0.115 success.
- `impedance_token_policy`: 0.488 +/- 0.120 success.
- Paired token-minus-learned success: -0.440 +/- 0.141.

The proposed token mechanism does not clear the strongest non-oracle baseline. The ablation table is also hostile: `token_no_memory` reaches 0.600 success while `token_full` reaches 0.457, so the claimed memory/update mechanism is not supported.

## Terminal Action

Archive/kill for ICLR main. Do not submit this paper as an ICLR main paper.

Revival condition: redesign the token mechanism, validate it on real robot or public contact-rich benchmarks, and show statistically reliable gains over learned gain adaptation and strong classical impedance/admittance baselines.
