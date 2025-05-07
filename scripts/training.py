
import numpy as np
import xgboost as xgb
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import log_loss

def train_loocv(X, y, params, rounds=1000, stop_rounds=30):
    """
    Train LOOCV-based XGBoost,
    returns the average validation loss, a list of losses for each fold, and a list of trained models..
    """
    loo = LeaveOneOut()
    fold_losses, models = [], []

    for train_idx, val_idx in loo.split(X):
        dtrain = xgb.DMatrix(X[train_idx], label=y[train_idx])
        dval   = xgb.DMatrix(X[val_idx],   label=y[val_idx])
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
        fold_losses.append(loss)
        models.append(bst)

    return np.mean(fold_losses), fold_losses, models

def save_models(models, out_dir):
    """Saves the model list in JSON format to the specified directory."""
    import os
    os.makedirs(out_dir, exist_ok=True)
    for i, m in enumerate(models):
        m.save_model(f"{out_dir}/model_{i}.json")
