"""End-to-end LOOCV on a tiny synthetic matrix: fold isolation, shapes, and inference path.

Deliberately small (n=12, 5 boosting rounds) so the whole file runs in a few seconds.
"""
import os
import tempfile
import unittest

import numpy as np

from scripts.training import save_models, save_preprocessors, train_loocv
from scripts.validation import ensemble_predict, load_models, load_preprocessors

PARAMS = {
    "objective": "multi:softprob", "num_class": 3, "max_depth": 2, "eta": 0.3,
    "eval_metric": "mlogloss", "tree_method": "hist", "seed": 0,
}


def synthetic(n=12, p=15, seed=1):
    rng = np.random.RandomState(seed)
    X = rng.rand(n, p)
    y = np.arange(n) % 3
    X[:, 0] += y
    return X, y


class TestTrainLoocv(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y = synthetic()
        cls.X[5, 9] = 1e4  # an outlier that must never reach fold 5's preprocessing
        cls.avg, cls.losses, cls.models, cls.pps, cls.oof = train_loocv(
            cls.X, cls.y, PARAMS, method="anova", k=4, rounds=5, stop_rounds=3
        )

    def test_one_model_and_preprocessor_per_sample(self):
        n = len(self.y)
        self.assertEqual(len(self.models), n)
        self.assertEqual(len(self.pps), n)
        self.assertEqual(len(self.losses), n)
        self.assertAlmostEqual(self.avg, float(np.mean(self.losses)))

    def test_out_of_fold_probabilities_are_distributions(self):
        self.assertEqual(self.oof.shape, (len(self.y), 3))
        np.testing.assert_allclose(self.oof.sum(axis=1), 1.0, atol=1e-5)
        self.assertTrue((self.oof >= 0).all())

    def test_held_out_sample_never_enters_its_folds_preprocessing(self):
        # Fold 5 holds out row 5; its scaler must not have seen the 1e4 outlier in column 9.
        self.assertLess(self.pps[5].scaler.data_max_[9], 10.0)
        # Every other fold trained on row 5 and therefore did see it.
        for i in (0, 1, 11):
            self.assertGreater(self.pps[i].scaler.data_max_[9], 1e3)

    def test_each_fold_selects_k_features(self):
        for pp in self.pps:
            self.assertEqual(len(pp.selected_indices), 4)

    def test_saved_artifacts_reproduce_raw_inference(self):
        with tempfile.TemporaryDirectory() as d:
            save_models(self.models, os.path.join(d, "models"))
            save_preprocessors(self.pps, os.path.join(d, "preproc"))
            self.assertEqual(len(os.listdir(os.path.join(d, "preproc"))), len(self.y))
            models = load_models(os.path.join(d, "models"), len(self.y))
            pps = load_preprocessors(os.path.join(d, "preproc"), len(self.y))
        pred_a, prob_a = ensemble_predict(self.models, self.X, self.pps)
        pred_b, prob_b = ensemble_predict(models, self.X, pps)
        np.testing.assert_allclose(prob_a, prob_b, atol=1e-6)
        np.testing.assert_array_equal(pred_a, pred_b)
        self.assertEqual(prob_a.shape, (len(self.y), 3))
        self.assertEqual(pred_a.shape, (len(self.y),))

    def test_missing_preprocessor_dir_returns_none(self):
        self.assertIsNone(load_preprocessors("/nonexistent/dir", 3))

    def test_mismatched_preprocessor_count_raises(self):
        with self.assertRaises(ValueError):
            ensemble_predict(self.models, self.X, self.pps[:-1])


if __name__ == "__main__":
    unittest.main()
