# scripts/validation.py

import os

import numpy as np
import xgboost as xgb
from scipy.special import softmax
from sklearn.metrics import accuracy_score, log_loss, classification_report, confusion_matrix

from scripts.fold_preprocessing import FoldPreprocessor


def load_models(model_dir, n_models):
    """Loads n_models number of XGBoost models from the directory."""
    models = []
    for i in range(n_models):
        bst = xgb.Booster()
        bst.load_model(f"{model_dir}/model_{i}.json")
        models.append(bst)
    return models


def load_preprocessors(preproc_dir, n_models):
    """Load the per-fold preprocessing artifacts saved next to the models, or None if absent."""
    if not preproc_dir or not os.path.isdir(preproc_dir):
        return None
    return [FoldPreprocessor.load(f"{preproc_dir}/preproc_{i}.json") for i in range(n_models)]


def ensemble_predict(models, X, preprocessors=None):
    """Soft Voting Ensemble: the raw margin predictions of each model are softmaxed and then averaged.

    If `preprocessors` is given (one per model), `X` is treated as the raw
    feature matrix and each model receives its own fold's scaled, selected view.
    Otherwise `X` must already be scaled and reduced to the selected features.
    """
    if preprocessors is None:
        dtest = xgb.DMatrix(X)
        probs = [softmax(m.predict(dtest, output_margin=True), axis=1) for m in models]
    else:
        if len(preprocessors) != len(models):
            raise ValueError("preprocessors and models must have the same length")
        probs = [
            softmax(m.predict(xgb.DMatrix(pp.transform(X)), output_margin=True), axis=1)
            for m, pp in zip(models, preprocessors)
        ]
    avg_probs = np.mean(probs, axis=0)
    preds = np.argmax(avg_probs, axis=1)
    return preds, avg_probs


def evaluate(y_true, y_pred, y_probs, class_labels):
    """Print Accuracy, Log Loss, Classification Report, Confusion Matrix."""
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Log Loss:", log_loss(y_true, y_probs))
    print("\nClassification Report:\n", 
          classification_report(y_true, y_pred, target_names=class_labels, digits=4))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
