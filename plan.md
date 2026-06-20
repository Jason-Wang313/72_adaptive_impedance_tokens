# Plan

Paper 72 was rebuilt as a real MuJoCo contact-control study before terminal packaging.

1. Implement a high-fidelity-ish MuJoCo planar contact benchmark for force-tracking surface sliding.
2. Compare fixed impedance, gain scheduling, adaptive impedance, admittance switching, robust MPC-style impedance, a trained learned gain regressor, the proposed impedance-token policy, and an oracle.
3. Run seven-seed main evaluation, token ablations, stress sweeps, uncertainty intervals, pairwise tests, and negative-case extraction.
4. Decide the ICLR-main gate from evidence.
5. Package the archive manuscript, numbered Downloads PDF, and public GitHub repo.

Outcome: KILL_ARCHIVE. The token mechanism loses to the learned gain regressor and its ablations do not support the core claim.

## 2026-06-21 Expanded-Standard Plan

Paper 72 now enters the same expanded-submission protocol used for Papers 61-71. The goal is not to decorate the existing 4-page archive; it is to test whether a redesigned impedance-token v5 can survive strong classical, learned, risk-aware, and ablated baselines under frozen gates.

Required additions:

1. Expand the contact-control benchmark from five v4 splits to a larger hostile set covering stiffness, friction, target-force, actuator, sensor-noise, stick-slip, surface discontinuity, delayed mode switch, combined, and combined-extreme regimes.
2. Add stronger baselines: random forest and histogram-gradient gain regressors, ensemble uncertainty gains, risk-averse impedance, conformal/safety-filter gains, token v4, token v5, and oracle.
3. Add hostile ablations for memory, discrete tokens, force updates, transition planning, safety penalties, slip context, tail-risk objective, calibration guard, phase memory, and learned-only token replacement.
4. Freeze hard-margin, paired, combined/extreme, fixed-risk, max-stress, ablation-necessity, and safety gates before the final run.
5. Build a 25+ page ICLR-style manuscript with new theory, generated appendix tables, bright boxed clickable citations, validation scripts, and a Downloads-only numbered PDF.
6. Preserve KILL_ARCHIVE unless the frozen evidence actually clears all core mechanism gates.

## 2026-06-15 Continuation Plan

1. Re-audit the real MuJoCo impedance-control evidence before making any submission-readiness claim.
2. Confirm the experiment source compiles and all raw CSVs are present, finite, and at the claimed scale.
3. Rebuild the PDF, repair recoverable LaTeX/BibTeX issues, and copy only `72.pdf` to Downloads.
4. Preserve `KILL_ARCHIVE` unless `impedance_token_policy` decisively beats learned gain adaptation and ablations support token memory.
5. Update child docs, root reports, and GitHub state before moving to Paper 73.
