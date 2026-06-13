# Paper 72 Rebuild Plan: Adaptive Impedance Tokens

Date: 2026-06-13

## Goal

Rebuild Paper 72 into a real ICLR-main-target robotics submission candidate, or terminate it honestly as `STRONG_REVISE` / `KILL_ARCHIVE` if real evidence does not justify submission. The central question is whether representing impedance choices as adaptive action tokens improves contact-rich control compared with classic adaptive impedance/admittance controllers, robust controllers, and learned policies.

## Core Claim To Test

Impedance should be treated as an action-level object that a robot policy can select and adapt from contact outcomes. A useful token mechanism should choose stiffness, damping, and force targets in response to contact mode, rather than relying on a single fixed impedance schedule or a black-box policy.

## High-Fidelity Benchmark

Build a MuJoCo contact-control benchmark with a planar or 2D manipulator interacting with compliant surfaces and contact tasks:

- Nominal surface tracking: maintain a target force while sliding along a wall/surface.
- Stiffness shift: surface stiffness changes at test time.
- Friction/slip shift: friction changes and the robot must avoid chatter/slip.
- Contact transition: free-space approach, impact, sustained contact, and release.
- Combined stress: stiffness shift plus friction shift, sensor noise, actuator limits, and force-target change.

The benchmark must log MuJoCo states, contact force, penetration, slip velocity, selected impedance token, controller gains, force error, tracking success, overshoot, chatter, safety violations, and energy/work.

## Methods To Implement

- `fixed_impedance`: fixed stiffness/damping controller.
- `gain_scheduled_impedance`: classic hand-coded gain schedule from contact force and penetration.
- `adaptive_impedance_control`: continuous adaptive stiffness/damping update.
- `admittance_switching_control`: switches between free-space admittance and contact impedance.
- `robust_mpc_impedance`: robust contact controller with conservative gains.
- `learned_gain_regressor`: supervised model predicting continuous gains from recent contact history.
- `impedance_token_policy`: proposed method; selects discrete impedance tokens, updates token belief from contact outcomes, and plans over token transitions.
- `oracle_impedance`: upper bound with access to surface/contact mode.

## Metrics

- Contact task success.
- Force tracking error.
- Peak impact overshoot.
- Chatter/slip rate.
- Safety/penetration violation rate.
- Settling time after contact transition.
- Energy/work cost.
- Tail risk under combined stress.

## Experimental Rigor

- Use at least 5 random seeds; target 7 if runtime stays manageable.
- Evaluate held-out stiffness/friction/contact-transition combinations.
- Report mean, 95 percent confidence intervals, and paired comparisons against the strongest non-oracle baseline.
- Include ablations: no token memory, no discrete tokens, no force-outcome update, no transition planner, no safety penalty.
- Include stress sweeps over stiffness, damping, friction, sensor noise, actuator limit, and target force.
- Save raw per-episode rollouts and per-seed summaries for auditability.

## Submission Gate

The paper can only move above archive if `impedance_token_policy` beats the best non-oracle classic or learned baseline on combined stress with a meaningful paired effect, lower overshoot/chatter, and no safety regression. If classic adaptive impedance/admittance or a learned gain regressor matches or beats it, the paper remains `KILL_ARCHIVE` or at best `STRONG_REVISE`.

## Deliverables

- Replace the synthetic script with a reproducible MuJoCo contact-control benchmark runner.
- Generate raw rollout CSVs, metrics, pairwise statistics, ablations, stress sweep tables, negative cases, and figures.
- Rewrite README, claims, reviewer attacks, novelty boundary, final audit, ICLR gate, and submission decision around actual evidence.
- Rewrite `paper/main.tex` as either a real negative-result paper or a submission-candidate manuscript.
- Compile `paper/main.pdf`, copy exactly to `C:/Users/wangz/Downloads/72.pdf`, and do not copy any PDF to Desktop.
- Commit and push the final Paper 72 repo, then update shared root reports before moving to Paper 73.
