# Sarcopenia Severity Classification using XGBoost and Feature Selection

This repository provides the implementation of an interpretable machine learning model for classifying sarcopenia severity (**Normal, Severe Sarcopenia, Very Severe Sarcopenia**) using protein-expression data. To overcome the high dimensionality of protein-expression features, we compared multiple feature-selection techniques and validated the model performance via Leave-One-Out Cross-Validation (LOOCV) and independent validation.

---

## Overview

We developed a robust **XGBoost-based classification model** enhanced by systematic feature selection methods: **ANOVA F-test**, **Chi-square**, and **Mutual Information**. The optimal feature subset (35 features) was selected based on LOOCV performance and interpretability. Our approach identified key predictive biomarkers that accurately discriminate sarcopenia severity classes.

---

## Dataset and Methods

- **Dataset**:
  - 5,420 protein-expression features
  - Three sarcopenia severity classes:
    - **N** (Normal)
    - **S** (Severe Sarcopenia)
    - **VS** (Very Severe Sarcopenia)

- **Validation**:
  - Leave-One-Out Cross-Validation (LOOCV)
  - Independent external validation set

---

## Key Features

- **Multiple Feature Selection Methods**: ANOVA F-test, Chi-square, Mutual Information
- **LOOCV-Based Training**: Ensures unbiased model evaluation
- **SHAP Analysis**: For model interpretability and biomarker identification
- **External Validation**: Independent validation demonstrating generalizability

---

## Results Summary

| Metric                      | Performance |
|-----------------------------|-------------|
| Accuracy                    | 72.7%       |
| Macro-average F1-score      | 0.7116      |
| Macro-average Precision     | 0.7190      |
| Macro-average Recall        | 0.7202      |

### Key Biomarkers (SHAP Analysis):

- **SERTA domain-containing protein 2 (SERTAD2)**
- **Homeobox protein Hox-D8 (HOXD8)**
- **Intraflagellar transport-associated protein (IFTAP)**
- **Receptor-type tyrosine-protein phosphatase alpha (PTPRA)**

---

## Repository Structure

```
sarcopenia-xgboost/
├── scripts/
│   ├── preprocessing.py        # Data loading and preprocessing (scaling)
│   ├── feature_selection.py    # Feature selection methods
│   ├── training.py             # LOOCV training & model saving
│   └── validation.py           # Ensemble predictions & evaluation
│
├── main.py                     # Main script for training and validation
├── extra_validation.py         # Additional validation script
├── requirements.txt            # Python dependencies
└── .gitignore                  # Git ignore file
```

---

## Installation & Usage

### Installation

```bash
git clone https://github.com/<your_username>/sarcopenia-xgboost.git
cd sarcopenia-xgboost

# Set up virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Model Training (LOOCV)

```bash
python main.py train \
  --data path/to/train.csv \
  --label Label \
  --k 35 \
  --methods anova chi2 mutual_info \
  --out results/
```

### Model Validation

```bash
python main.py validate \
  --data path/to/test.csv \
  --label Label \
  --model_dir results/anova_k35/models \
  --labels N S VS
```

### Extra Independent Validation

```bash
python extra_validation.py \
  --test_csv path/to/extra_test.csv \
  --label_col Label \
  --model_dir results/anova_k35/models \
  --class_labels N S VS
```

---

## Citation

If you use this repository for your research, please cite our paper:

```bibtex
@article{YourPaper2025,
  title   = {Your Paper Title},
  author  = {Your Name},
  journal = {Journal Name},
  year    = {2025},
  doi     = {10.xxxx/xxxxx}
}
```

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.
