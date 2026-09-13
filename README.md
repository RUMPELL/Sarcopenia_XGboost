# Sarcopenia Severity Classification with XGBoost and Feature Selection

[![CI](https://github.com/RUMPELL/Sarcopenia_XGboost/actions/workflows/ci.yml/badge.svg)](https://github.com/RUMPELL/Sarcopenia_XGboost/actions/workflows/ci.yml)

An interpretable classical-ML pipeline for classifying sarcopenia severity into three
classes — **Normal (N)**, **Severe (S)**, **Very Severe (VS)** — from high-dimensional
blood protein-expression data. This repository organises in code the methodology of the
master's thesis *"Blood Protein Biomarker Signature for Sarcopenia Severity Stratification
via Machine Learning"*: univariate feature-selection methods are compared, XGBoost is
trained under leave-one-out cross-validation (LOOCV), and a soft-voting ensemble of the
LOOCV models is evaluated on held-out data.

> The protein-expression dataset is not distributed. The scripts expect a CSV you supply.

---

## Problem

- **Input:** 5,420 blood protein-expression features per sample, from a small clinical
  cohort.
- **Output:** one of three severity classes (`N`, `S`, `VS`).
- **Challenge:** the feature count vastly exceeds the sample count, so feature selection
  and the validation design are the central methodological questions.

## Method

| Step | Implementation |
|---|---|
| Scaling | `MinMaxScaler` (`scripts/preprocessing.py`) |
| Feature selection | `SelectKBest` with **ANOVA F-test**, **χ²**, or **mutual information**, top-*k* (default 35) (`scripts/feature_selection.py`) |
| Model | XGBoost `multi:softprob`, `max_depth=10`, `eta=0.1`, up to 1000 rounds, early stopping 30 (`main.py`) |
| Validation | LOOCV — one XGBoost model per left-out sample. Scaling and feature selection are fitted inside each fold on the training rows only, so held-out samples never influence preprocessing (`scripts/fold_preprocessing.py`, `scripts/training.py`) |
| Inference | Soft-voting ensemble: each fold's saved scaler + selector is applied to the raw test matrix, per-model softmax over raw margins, averaged, argmax (`scripts/validation.py`) |
| Interpretation | SHAP biomarker ranking (thesis analysis; SHAP code is not included in this repository) |

## Results (as reported in the thesis)

| Setting | Metric |
|---|---|
| 35-biomarker signature, LOOCV | AUROC 0.930 |
| Independent external cohort (13 overlapping biomarkers) | Accuracy 78.6 % |

Top biomarkers by SHAP: SERTAD2, HOXD8, IFTAP, PTPRA.

These figures are the values reported in the thesis. The dataset is not part of this
repository, so they cannot be regenerated here.

## Running the pipeline

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# LOOCV training with each feature-selection method
python main.py train --data train.csv --label Label --k 35 \
  --methods anova chi2 mutual_info --out results

# ensemble evaluation on a raw test CSV (same columns as training)
python main.py validate --data test.csv --label Label \
  --model_dir results/anova_k35/models --n_models <number of training samples>

# tests (synthetic data only)
python -m unittest discover -s tests -t .
```

`train` writes, per method, under `results/<method>_k<k>/`:
`loocv_losses.csv` (per-fold loss and selected feature indices),
`loocv_oof_predictions.csv` (out-of-fold class probabilities), `models/model_{i}.json`,
and `preproc/preproc_{i}.json`.

`main.py validate` applies each fold's saved scaler and selector to the raw test matrix by
default; pass `--preprocessed` if the CSV is already scaled and reduced to the selected
features. `extra_validation.py` evaluates an already-preprocessed matrix with every model
found in a directory.

## Repository structure

```
main.py                       train / validate CLI
extra_validation.py           directory-based ensemble evaluation (already-preprocessed input)
scripts/preprocessing.py      CSV loading; scaler factory
scripts/feature_selection.py  ANOVA / chi2 / mutual-information score functions
scripts/fold_preprocessing.py per-fold scaler + selector, with save/load
scripts/training.py           LOOCV training with per-fold preprocessing, model/preprocessor saving
scripts/validation.py         model + preprocessor loading, soft-voting ensemble, report
tests/                        synthetic unit tests (17)
requirements.txt
```

## Status

Research code accompanying the thesis. Unit tests (17) run on synthetic data in CI.
