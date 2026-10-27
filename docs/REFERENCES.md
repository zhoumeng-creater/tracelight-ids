# Sources and scope

This file identifies implementation sources, not a claim to reproduce published scores.

- [UNSW-NB15 official dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset): split files, attack categories, provenance and required dataset citations. Consult the original dataset papers listed by the authors when writing the dissertation; older dataset citations remain necessary even when related-work discovery focuses on 2025–2026.
- [scikit-learn common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html): fit preprocessing on training data only.
- [scikit-learn precision-recall evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall-f-measure-metrics): distinguish average precision from ROC-AUC and operating-point metrics.
- [PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html): seeded execution and limitations across software/hardware versions. Determinism is scoped to a recorded environment, not a cross-device bitwise guarantee.
- [PyTorch previous-version installation](https://pytorch.org/get-started/previous-versions/): explicit 2.6.0 CPU / CUDA 12.6 wheels used by this initial implementation.
- [NVIDIA RTX 4090 D](https://www.nvidia.cn/geforce/graphics-cards/40-series/rtx-4090-d/): target desktop GPU specification, 24GB. Hardware availability and driver compatibility still require a lab preflight.

The initial implementation includes LR, RF, histogram gradient boosting and MLP. It does not claim to reproduce a 2025–2026 paper, win a benchmark, provide production SOC coverage, or perform online packet extraction. Broader literature and product comparison belong to the research planning materials; no private university documents are distributed here.
