# Hostile Reviewer Response

Paper: 72 Adaptive Impedance Tokens

## Strongest Technical Threats

- A hybrid control strategy for grinding and polishing robot based on adaptive impedance control (2021)
- Adaptive Admittance-Impedance Switching Control for Safe Mobile Manipulation: A Door Opening Application with Predictive Anti-Slip Logic (2026)
- Force Tracking in Impedance Control (1997)
- Smooth adaptive hybrid impedance control for robotic contact force tracking in dynamic environments (2020)
- Stable Robust Adaptive Impedance Control of a Prosthetic Leg (2015)
- Adaptive variable impedance control for dynamic contact force tracking in uncertain environment (2018)
- Passivity-Based Skill Motion Learning in Stiffness-Adaptive Unified Force-Impedance Control (2022)
- Adaptive Impedance Control for the Haptic Shared Driving Task Based on Nonlinear MPC (2020)

## Hostile ICLR Review

A hostile reviewer should reject this as an ICLR-main submission. The v4 rebuild is much stronger than the previous archive: it implements a MuJoCo contact benchmark, baselines, ablations, stress sweeps, and uncertainty estimates. But the proposed mechanism loses to a trained learned gain regressor on the decisive split.

The strongest objection is not merely "needs more experiments." It is that the current experiments already falsify the main claim. The full token policy reaches 0.488 combined-stress success, while the learned gain regressor reaches 0.929, and the paired gap is -0.440 +/- 0.141. The ablation table is also damaging because `token_no_memory` outperforms the full token method.

## Honest Action

Mark `KILL_ARCHIVE`. Keep the repository as a negative-result package and do not submit it as an ICLR main paper.

## What Would Be Needed To Revive

- Redesign the token mechanism so memory and token updates help rather than hurt.
- Train or infer tokens from contact data instead of using hand-coded token scores.
- Validate on real robot hardware or a public contact-rich manipulation benchmark.
- Beat learned gain adaptation, classical adaptive impedance, admittance switching, robust control, and oracle sanity checks where applicable.
