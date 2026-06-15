# Submission Attack Log

Paper: 72 adaptive_impedance_tokens

This v4 pass applies the ICLR main-conference bar after rebuilding the evidence package.

## 2026-06-15 Continuation Audit
Attack: A real MuJoCo contact-control benchmark does not rescue the paper unless the token policy beats learned gain adaptation and the token-memory ablation story holds.

Verdict: Fatal. The verified CSVs contain 3,360 main rollouts, 420 ablation rollouts, and 2,016 stress rollouts across 7 seeds. On combined stress, `impedance_token_policy` reaches 0.488 success versus 0.929 for `learned_gain_regressor`, with paired token-minus-learned success difference -0.440 +/- 0.141. It also loses to `gain_scheduled_impedance` by -0.250 +/- 0.050. The ablation grid undercuts the mechanism: `token_full` reaches 0.457 success while `token_no_memory` reaches 0.600.

Action: Keep KILL_ARCHIVE and preserve the reproducible negative result without claiming ICLR-main readiness.

## Attack 1: Does the proposed method beat the strongest implemented baseline?

Verdict: No. `impedance_token_policy` reaches 0.488 +/- 0.120 combined-stress success. `learned_gain_regressor` reaches 0.929 +/- 0.056.

Action: Kill/archive.

## Attack 2: Is the paired comparison favorable?

Verdict: No. Token-minus-learned paired success difference is -0.440 +/- 0.141 over seven seeds.

Action: Kill/archive.

## Attack 3: Do ablations support the token-memory mechanism?

Verdict: No. `token_no_memory` reaches 0.600 success, above `token_full` at 0.457.

Action: Treat the mechanism as falsified in this implementation.

## Attack 4: Are the baselines real enough for a first submission gate?

Verdict: Improved but not sufficient for acceptance. The repo now has implemented controllers and a trained learned baseline, but lacks real robot and external public benchmark validation.

Action: Keep as a local negative-result archive.

## Attack 5: Is the stress sweep polished?

Verdict: Partially. Raw and continuous metrics are reported, but low-stress cases reveal force-undertracking artifacts, so this is not a benchmark contribution.

Action: Report honestly as a limitation.

## Attack 6: Does hostile prior work leave enough novelty?

Verdict: Not in the current evidence. Adaptive impedance, admittance switching, robust force tracking, and learned gain adaptation are crowded areas. The proposed token mechanism would need decisive empirical gains to matter.

Action: Kill/archive.

## Attack 7: Could text polishing rescue the paper?

Verdict: No. The central empirical claim fails.

Action: Stop at v4 negative-result package.
