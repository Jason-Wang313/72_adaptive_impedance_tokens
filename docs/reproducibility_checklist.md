# Reproducibility Checklist

## What Reproduces

- [x] `python src/run_experiment.py`
- [x] MuJoCo contact-control rollouts in `results/impedance_raw.csv`
- [x] Main metrics in `results/metrics.csv` and `results/impedance_metrics.csv`
- [x] Per-seed summaries in `results/raw_seed_metrics.csv`
- [x] Pairwise tests in `results/pairwise_stats.csv`
- [x] Token ablations in `results/ablation_metrics.csv`
- [x] Stress sweep raw and summary tables in `results/stress_sweep_raw.csv` and `results/stress_sweep.csv`
- [x] Negative cases in `results/negative_cases.csv`
- [x] Figures in `figures/`
- [x] Manuscript source in `paper/main.tex`
- [x] Canonical PDF: `C:/Users/wangz/Downloads/72.pdf`

## What Does Not Reproduce

- [ ] Real robot experiments.
- [ ] External public contact-rich benchmark results.
- [ ] A learned token policy trained end to end from robot data.
- [ ] Video/visual rollout artifacts.

This repository is reproducible as a local MuJoCo negative-result archive, not as an ICLR-main-ready robotics submission.
