# Paper 72 Expanded Terminal Decision

Decision: KILL_ARCHIVE

Reason: v5 does not beat strongest hard-regime baseline admittance_switching_control by 0.030 (v5=0.000, best=0.941); paired lower bound against admittance_switching_control is not positive (-0.941+/-0.025); v5 does not beat strongest combined/extreme baseline conformal_safety_gain by 0.030 (v5=0.000, best=0.969); fixed-risk gate fails at budget 0.10 (v5=0.000, best=token_no_memory_ablation 0.051); maximum-stress gate fails (v5=0.000, best=adaptive_impedance_control 1.000); ablation gate fails because ablate_no_calibration_guard, ablate_no_discrete_tokens, ablate_no_force_update, ablate_no_memory, ablate_no_phase_memory, ablate_no_safety_penalty, ablate_no_slip_context, ablate_no_tail_risk_objective, ablate_no_transition_planner, impedance_token_policy_v4, learned_only_token_replacement matches or beats full v5

Training rows: 2400
Main method-evaluation rows: 8640
Rollout rows: 8640
Seed summary rows: 1440
Split-method metric rows: 180
Pairwise rows: 168
Ablation rows: 1536
Stress rows: 4320
Negative cases: 12

This decision is generated from frozen CSV artifacts, not hand-transcribed table values.
