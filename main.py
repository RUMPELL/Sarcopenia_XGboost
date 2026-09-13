#!/usr/bin/env python
# main.py

import argparse, os, numpy as np, pandas as pd
from scripts.preprocessing      import load_raw
from scripts.feature_selection  import SCORE_FUNCS
from scripts.training           import train_loocv, save_models, save_preprocessors
from scripts.validation         import load_models, load_preprocessors, ensemble_predict, evaluate

def train_cmd(args):
    # Raw features: scaling and feature selection happen inside each LOOCV fold.
    X, y = load_raw(args.data, args.label)
    for method in args.methods:
        if method not in SCORE_FUNCS:
            print(f"[WARN] Unknown FS method: {method}")
            continue

        params = {
            "objective":   "multi:softprob",
            "num_class":   len(np.unique(y)),
            "max_depth":   args.max_depth,
            "eta":         args.eta,
            "eval_metric":"mlogloss",
            "tree_method":"hist",
            "seed":        args.seed
        }
        avg_loss, losses, models, preprocessors, oof_probs = train_loocv(
            X, y, params,
            method=method, k=args.k,
            rounds=args.rounds,
            stop_rounds=args.stop
        )
        oof_pred = np.argmax(oof_probs, axis=1)
        print(f"[{method}] LOOCV Avg Loss: {avg_loss:.4f}  |  out-of-fold accuracy: {np.mean(oof_pred == y):.4f}")

        # Save Results
        out_dir = os.path.join(args.out, f"{method}_k{args.k}")
        os.makedirs(out_dir, exist_ok=True)

        pd.DataFrame({
            "fold": list(range(1, len(losses)+1)),
            "loss": losses,
            "selected_indices": [" ".join(map(str, pp.selected_indices)) for pp in preprocessors],
        }).to_csv(f"{out_dir}/loocv_losses.csv", index=False)

        oof = pd.DataFrame(oof_probs, columns=[f"prob_{c}" for c in range(oof_probs.shape[1])])
        oof.insert(0, "y_true", y)
        oof.insert(1, "y_pred", oof_pred)
        oof.to_csv(f"{out_dir}/loocv_oof_predictions.csv", index=False)

        save_models(models, f"{out_dir}/models")
        save_preprocessors(preprocessors, f"{out_dir}/preproc")
        print(f"[{method}] Results saved to {out_dir}")

def validate_cmd(args):
    df = pd.read_csv(args.data)
    X_test = df.drop(columns=[args.label]).values
    y_test = df[args.label].values

    models = load_models(args.model_dir, args.n_models)
    preprocessors = None
    if not args.preprocessed:
        preproc_dir = args.preproc_dir or os.path.join(os.path.dirname(args.model_dir.rstrip("/")), "preproc")
        preprocessors = load_preprocessors(preproc_dir, args.n_models)
        if preprocessors is None:
            raise SystemExit(
                f"No preprocessing artifacts found in {preproc_dir}. Either point --preproc_dir at the "
                "directory written by `train`, or pass --preprocessed if the CSV is already scaled and reduced."
            )
    y_pred, y_probs = ensemble_predict(models, X_test, preprocessors)
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
    p_train.add_argument("--seed",      type=int, default=42)
    p_train.add_argument("--out",       default="results", help="output root dir")
    p_train.set_defaults(func=train_cmd)

    # validate
    p_val = subparsers.add_parser("validate", help="Ensemble validate on test set")
    p_val.add_argument("--data",      required=True, help="test CSV path (raw features by default)")
    p_val.add_argument("--label",     default="Label", help="label column name")
    p_val.add_argument("--model_dir", required=True, help="directory of saved models (results/<method>_k<k>/models)")
    p_val.add_argument("--preproc_dir", default=None,
                       help="directory of per-fold preprocessing artifacts; defaults to <model_dir>/../preproc")
    p_val.add_argument("--preprocessed", action="store_true",
                       help="the CSV is already scaled and reduced to the selected features")
    p_val.add_argument("--n_models",  type=int, default=50, help="number of models")
    p_val.add_argument("--labels",    nargs="+", default=["N","S","VS"],
                       help="class label names")
    p_val.set_defaults(func=validate_cmd)

    args = parser.parse_args()
    args.func(args)
