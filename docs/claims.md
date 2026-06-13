# Claims

- Mechanism claim tested: impedance choices can be represented as adaptive action tokens grounded in contact outcomes.
- Evidence claim: a seven-seed MuJoCo contact-control benchmark compares the token policy against fixed, scheduled, adaptive, admittance-switching, robust, learned, and oracle impedance controllers.
- Result claim: the current token policy fails the main gate; it reaches 0.488 combined-stress success versus 0.929 for the trained learned gain regressor.
- Ablation claim: token memory is not supported; `token_no_memory` outperforms the full token variant in the combined-stress ablation.
- Scope claim: the repository is a reproducible negative-result archive, not an ICLR-main-ready submission.
- Unsupported claim explicitly avoided: no claim of state-of-the-art robot contact control.
