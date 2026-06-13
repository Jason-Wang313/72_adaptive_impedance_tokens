# Plan

Paper 72 was rebuilt as a real MuJoCo contact-control study before terminal packaging.

1. Implement a high-fidelity-ish MuJoCo planar contact benchmark for force-tracking surface sliding.
2. Compare fixed impedance, gain scheduling, adaptive impedance, admittance switching, robust MPC-style impedance, a trained learned gain regressor, the proposed impedance-token policy, and an oracle.
3. Run seven-seed main evaluation, token ablations, stress sweeps, uncertainty intervals, pairwise tests, and negative-case extraction.
4. Decide the ICLR-main gate from evidence.
5. Package the archive manuscript, numbered Downloads PDF, and public GitHub repo.

Outcome: KILL_ARCHIVE. The token mechanism loses to the learned gain regressor and its ablations do not support the core claim.
