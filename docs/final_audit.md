# Final Audit

1. Chosen thesis: impedance choices can be represented as adaptive action tokens grounded in contact outcomes.
2. ICLR-main decision: KILL_ARCHIVE.
3. Submission-hardening version: v5 expanded archive.
4. Evidence: CPU-only MuJoCo contact-control benchmark with 8,640 main rows, 1,536 ablation rows, 4,320 stress rows, 1,440 seed summaries, 168 paired comparisons, 8 seeds, 12 main splits, 15 main methods, 12 ablation methods, 12 stress methods, figures, fixed-risk analysis, and negative cases.
5. Decisive hard-split result: `impedance_token_policy_v5` reaches 0.000 success; `admittance_switching_control` reaches 0.941.
6. Decisive combined/extreme result: `impedance_token_policy_v5` reaches 0.000 success; `conformal_safety_gain` reaches 0.969.
7. Decisive fixed-risk result: `impedance_token_policy_v5` reaches 0.000 at a 10 percent risk budget; `token_no_memory_ablation` reaches 0.051.
8. Decisive ablation result: full v5 reaches 0.000; `learned_only_token_replacement` reaches 0.875.
9. Closest hostile prior work: see `docs/hostile_prior_work.md`, `docs/hostile_prior_work_100_cards.csv`, and `docs/hostile_reviewer_response.md`; the v5 manuscript uses verified impedance-control, variable-impedance, MuJoCo, learned-baseline, MPC, and conformal-prediction references.
10. Reproducibility: `scripts/build_submission_pdf.ps1` rebuilds generated TeX assets and the PDF; `scripts/validate_submission_artifacts.py` validates row counts, figures, TeX link boxes, Downloads PDF placement, repo URL, page count, and Desktop hygiene.
11. PDF scale: 54 pages.
12. Exact Downloads PDF path: `C:/Users/wangz/Downloads/72.pdf`
13. GitHub URL: https://github.com/Jason-Wang313/72_adaptive_impedance_tokens
14. Confirmation: no visible Desktop copy was requested or made.
15. Visual QA: rendered representative PDF pages and inspected title/abstract, main results, figure pages, long appendix tables, and references.
16. Claim-validity status: main-conference claims killed; reproducible negative-result archive retained.
