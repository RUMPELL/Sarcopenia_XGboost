#!/usr/bin/env python
# main.py

import argparse, os, numpy as np, pandas as pd
from scripts.preprocessing    import load_and_scale
from scripts.feature_selection import FS_METHODS
from scripts.training         import train_loocv, save_models
from scripts.validation       import load_models, ensemble_predict, evaluate

def train_cmd(args):
    X, y = load_and_scale(args.data, args.label)
    for method in args.methods:
        if method not in FS_METHODS:
            print(f"[WARN] Unknown FS method: {method}")
            continue

        # Feature Selection
        sel_fn = FS_METHODS[method]
        Xk, idx = sel_fn(X, y, args.k)
        print(f"\n[{method}] Selected top-{args.k} indices: {idx}")

        # LOOCV Training
        params = {
            "objective":   "multi:softprob",
            "num_class":   len(np.unique(y)),
            "max_depth":   args.max_depth,
            "eta":         args.eta,
            "eval_metric":"mlogloss",
            "tree_method":"hist",
            "seed":        args.seed
        }
        avg_loss, losses, models = train_loocv(
            Xk, y, params,
            rounds=args.rounds,
            stop_rounds=args.stop
        )
        print(f"[{method}] LOOCV Avg Loss: {avg_loss:.4f}")

        # Save Results
        out_dir = os.path.join(args.out, f"{method}_k{args.k}")
        os.makedirs(out_dir, exist_ok=True)

        pd.DataFrame({
            "fold": list(range(1, len(losses)+1)),
            "loss": losses
        }).to_csv(f"{out_dir}/loocv_losses.csv", index=False)

        save_models(models, f"{out_dir}/models")
        print(f"[{method}] Results saved to {out_dir}")

def validate_cmd(args):
    df = pd.read_csv(args.data)
    X_test = df.drop(columns=[args.label]).values
    y_test = df[args.label].values

    models = load_models(args.model_dir, args.n_models)
    y_pred, y_probs = ensemble_predict(models, X_test)
    evaluate(y_test, y_pred, y_probs, args.labels)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Sarcopenia XGBoost Pipeline")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    # train
    p_train = subparsers.add_parser("train", help="LOOCV train with multiple FS methods")
    p_train.add_argument("--data",      required=True, help="train CSV path")
    p_train.add_argument("--label",     default="Label", help="label column name")
    p_train.add_argument("--k",         type=int, default=35, help="number of features")
    p_train.add_argument("--methods", nargs="+",
                         default=["anova","chi2","mutual_info"],
                         help="FS methods to compare")
    p_train.add_argument("--max_depth", type=int, default=10)
    p_train.add_argument("--eta",       type=float, default=0.1)
    p_train.add_argument("--rounds",    type=int,   default=1000)
    p_train.add_argument("--stop",      type=int,   default=30)
    p_train.add_argument("--seed",      type=int,   default=42)
    p_train.add_argument("--out",       default="results", help="output root dir")
    p_train.set_defaults(func=train_cmd)

    # validate
    p_val = subparsers.add_parser("validate", help="Ensemble validate on test set")
    p_val.add_argument("--data",      required=True, help="test CSV path")
    p_val.add_argument("--label",     default="Label", help="label column name")
    p_val.add_argument("--model_dir", required=True, help="directory of saved models")
    p_val.add_argument("--n_models",  type=int, default=50, help="number of models")
    p_val.add_argument("--labels",    nargs="+", default=["N","S","VS"],
                       help="class label names")
    p_val.set_defaults(func=validate_cmd)

    args = parser.parse_args()
    args.func(args)
