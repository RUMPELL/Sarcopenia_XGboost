
import os
import argparse
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, log_loss, classification_report, confusion_matrix
from tqdm import tqdm

def load_models(model_dir, extension='.json'):
    """
    Load all XGBoost models from a directory.
    """
    model_files = [f for f in os.listdir(model_dir) if f.endswith(extension)]
    models = []
    for mf in tqdm(model_files, desc="Loading models"):
        bst = xgb.Booster()
        bst.load_model(os.path.join(model_dir, mf))
        models.append(bst)
    return models

def ensemble_predict(models, X):
    """
    Perform soft-voting ensemble: average per-class probabilities.
    """
    dmat = xgb.DMatrix(X)
    preds = []
    for model in tqdm(models, desc="Predicting"):
        preds.append(model.predict(dmat))
    probs = np.array(preds)  # shape (n_models, n_samples, n_classes)
    avg_probs = np.mean(probs, axis=0)
    y_pred = np.argmax(avg_probs, axis=1)
    return y_pred, avg_probs

def main():
    parser = argparse.ArgumentParser(
        description="Extra validation: ensemble predict and evaluate on a test set"
    )
    parser.add_argument("--test_csv",    required=True,
                        help="Path to CSV file for validation (must include label column)")
    parser.add_argument("--label_col",   default="Label",
                        help="Name of the label column in the CSV")
    parser.add_argument("--model_dir",   required=True,
                        help="Directory containing saved model_*.json files")
    parser.add_argument("--n_models",    type=int, default=50,
                        help="Number of models to load and ensemble")
    parser.add_argument("--class_labels", nargs="+",
                        default=["N","S","VS"],
                        help="Names of classes for reporting")
    args = parser.parse_args()

    # Load test data
    df = pd.read_csv(args.test_csv)
    X_test = df.drop(columns=[args.label_col]).values
    y_test = df[args.label_col].values

    # Load and ensemble predict
    models = load_models(args.model_dir)
    y_pred, y_probs = ensemble_predict(models, X_test)

    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    loss = log_loss(y_test, y_probs)
    report = classification_report(y_test, y_pred,
                                   target_names=args.class_labels, digits=4)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy:  {acc:.4f}")
    print(f"Log Loss:   {loss:.4f}\\n")
    print("Classification Report:\\n", report)
    print("Confusion Matrix:\\n", cm)

if __name__ == "__main__":
    main()
