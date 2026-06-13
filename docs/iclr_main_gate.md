# ICLR Main Gate

Paper: 72 adaptive_impedance_tokens

Hardening version: v4

Gate verdict: KILL_ARCHIVE

Evidence digest: 918ae8152a0d4b64

## Why It Fails

The v4 rebuild produced real local evidence, but the central claim fails:

- The proposed `impedance_token_policy` achieves 0.488 +/- 0.120 success on `combined_stress`.
- The strongest non-oracle baseline, `learned_gain_regressor`, achieves 0.929 +/- 0.056.
- The paired success difference is -0.440 +/- 0.141 against the token policy.
- `token_no_memory` outperforms the full token method in the ablation table, weakening the intended mechanism story.

## Remaining Main-Track Blockers

- No real-robot evaluation.
- No external public benchmark such as a standard manipulation/contact suite.
- The best implemented non-oracle baseline already beats the proposed mechanism.
- The stress sweep exposes a low-stress force-undertracking artifact; continuous metrics are reported, but this is not a polished benchmark contribution.
- The hostile prior-work set around adaptive impedance, admittance switching, and contact-force tracking remains crowded.

The only honest main-conference-safe decision is to archive rather than overclaim.
