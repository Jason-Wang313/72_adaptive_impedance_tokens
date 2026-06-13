# Experiment Rigor Checklist

## v4 Evidence

- [x] MuJoCo contact-control benchmark.
- [x] Seven random seeds.
- [x] 12 evaluation episodes per seed/split.
- [x] Implemented classical impedance/admittance baselines.
- [x] Implemented robust controller baseline.
- [x] Trained learned gain regressor baseline.
- [x] Proposed impedance-token controller.
- [x] Oracle upper-bound controller.
- [x] Paired comparisons against the proposed method.
- [x] Token ablations.
- [x] Stress sweeps.
- [x] Negative-case extraction.
- [x] Paper-specific figures.

## ICLR Main Bar

- [ ] Proposed method beats the strongest non-oracle baseline.
- [ ] Ablations support the central mechanism.
- [ ] Real-robot validation.
- [ ] External public benchmark validation.
- [ ] Manual full-paper related-work synthesis deep enough for submission.
- [ ] Qualitative rollouts or videos.

Decision: fail ICLR-main empirical-rigor gate; archive.
