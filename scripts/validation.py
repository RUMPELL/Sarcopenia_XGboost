# scripts/validation.py

import numpy as np
import xgboost as xgb
from scipy.special import softmax
from sklearn.metrics import accuracy_score, log_loss, classification_report, confusion_matrix

def load_models(model_dir, n_models):
    """Loads n_models number of XGBoost models from the directory."""
    import os
    models = []
    for i in range(n_models):
        bst = xgb.Booster()
        bst.load_model(f"{model_dir}/model_{i}.json")
        models.append(bst)
    return models

def ensemble_predict(models, X):
    """Soft Voting Ensemble: The raw margin predictions of each model are softmaxed and then averaged."""
    dtest = xgb.DMatrix(X)
    probs = [softmax(m.predict(dtest, output_margin=True), axis=1) for m in models]
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
