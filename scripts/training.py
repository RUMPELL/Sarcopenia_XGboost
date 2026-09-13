
import os

import numpy as np
import xgboost as xgb
from sklearn.model_selection import LeaveOneOut

from scripts.fold_preprocessing import FoldPreprocessor


def train_one_fold(X_tr, y_tr, X_va, y_va, params, rounds=1000, stop_rounds=30):
    """Train one XGBoost model with early stopping on the held-out sample.

    Returns (validation mlogloss, booster).
    """
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_va, label=y_va)
    res = {}
    bst = xgb.train(
        params, dtrain,
        num_boost_round=rounds,
        evals=[(dtrain, "train"), (dval, "eval")],
        early_stopping_rounds=stop_rounds,
        evals_result=res,
        verbose_eval=False
    )
    loss = res["eval"]["mlogloss"][-1]
    return loss, bst


def train_loocv(X, y, params, method="anova", k=35, rounds=1000, stop_rounds=30):
    """LOOCV over a *raw* feature matrix with per-fold preprocessing.

    For every held-out sample the scaler and the top-k feature selector are
    fitted on the remaining training rows only, then applied to the held-out
    row, so the held-out sample never influences preprocessing.

    Returns:
        avg_loss        mean validation mlogloss over folds
        fold_losses     list of per-fold mlogloss
        models          list of trained boosters (one per fold)
        preprocessors   list of fitted FoldPreprocessor (one per fold)
        oof_probs       (n_samples, n_classes) out-of-fold class probabilities
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    n_classes = int(params["num_class"])
    loo = LeaveOneOut()
    fold_losses, models, preprocessors = [], [], []
    oof_probs = np.zeros((len(y), n_classes), dtype=float)

    for train_idx, val_idx in loo.split(X):
        pp = FoldPreprocessor(method=method, k=k).fit(X[train_idx], y[train_idx])
        X_tr = pp.transform(X[train_idx])
        X_va = pp.transform(X[val_idx])

        loss, bst = train_one_fold(X_tr, y[train_idx], X_va, y[val_idx], params, rounds, stop_rounds)
        oof_probs[val_idx] = bst.predict(xgb.DMatrix(X_va))

        fold_losses.append(loss)
        models.append(bst)
        preprocessors.append(pp)

    return float(np.mean(fold_losses)), fold_losses, models, preprocessors, oof_probs


def save_models(models, out_dir):
    """Saves the model list in JSON format to the specified directory."""
    os.makedirs(out_dir, exist_ok=True)
    for i, m in enumerate(models):
        m.save_model(f"{out_dir}/model_{i}.json")


def save_preprocessors(preprocessors, out_dir):
    """Save one preprocessing artifact per fold, aligned with model_{i}.json."""
    os.makedirs(out_dir, exist_ok=True)
    for i, pp in enumerate(preprocessors):
        pp.save(f"{out_dir}/preproc_{i}.json")
