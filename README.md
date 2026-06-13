# 72 Adaptive Impedance Tokens

Submission-hardening version: v4

Terminal decision: KILL_ARCHIVE for ICLR main conference.

This repository now contains a real Paper 72 rebuild: a MuJoCo contact-control benchmark, implemented impedance/admittance/robust/learned/token baselines, seven-seed evaluation, uncertainty intervals, ablations, stress sweeps, negative cases, figures, and a rewritten archive manuscript. The evidence does not support an ICLR-main submission. The proposed `impedance_token_policy` loses decisively to a trained `learned_gain_regressor` on the combined-stress split.

## Main Result

Full run:

- Main evaluation rows: 3360.
- Ablation rows: 420.
- Stress rows: 2016.
- Seeds: 0 through 6.
- Episodes per seed and split: 12.
- Runtime: 717.59 seconds.

Combined-stress success:

- `learned_gain_regressor`: 0.929 +/- 0.056.
- `oracle_impedance`: 0.881 +/- 0.079.
- `gain_scheduled_impedance`: 0.738 +/- 0.115.
- `impedance_token_policy`: 0.488 +/- 0.120.

Paired comparison against the strongest non-oracle baseline:

- Token policy versus learned gain regressor: -0.440 +/- 0.141 success.

This fails the submission gate. The paper is retained as a reproducible negative-result archive, not as a submission.

## Reproduce

```powershell
python src\run_experiment.py
```

Outputs are written under `results/` and `figures/`.

## Rebuild PDF

```powershell
cd paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Canonical local PDF: `C:/Users/wangz/Downloads/72.pdf`

No PDF is copied to the visible Desktop.
