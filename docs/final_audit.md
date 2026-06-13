# Final Audit

1. Chosen thesis: impedance choices can be represented as adaptive action tokens grounded in contact outcomes.
2. ICLR-main decision: KILL_ARCHIVE.
3. Submission-hardening version: v4.
4. Evidence: MuJoCo contact-control benchmark with 3360 main rows, 420 ablation rows, 2016 stress rows, seven seeds, implemented baselines, uncertainty intervals, paired statistics, figures, and negative cases.
5. Decisive result: `impedance_token_policy` loses to `learned_gain_regressor` on `combined_stress`, 0.488 +/- 0.120 versus 0.929 +/- 0.056 success.
6. Paired result: token-minus-learned success difference is -0.440 +/- 0.141.
7. Ablation result: `token_no_memory` reaches 0.600 success, above `token_full` at 0.457, so the claimed memory-based token adaptation is not supported.
8. Closest hostile prior work: see `docs/hostile_prior_work.md`, `docs/hostile_prior_work_100_cards.csv`, and `docs/hostile_reviewer_response.md`.
9. Reproducibility: `python src/run_experiment.py` regenerates results and figures; code uses MuJoCo, NumPy, Matplotlib, and scikit-learn.
10. Claim-validity status: main-conference claims killed; reproducible negative-result archive retained.
11. Exact Downloads PDF path: `C:/Users/wangz/Downloads/72.pdf`
12. GitHub URL: https://github.com/Jason-Wang313/72_adaptive_impedance_tokens
13. Confirmation: no visible Desktop copy was requested or made.
