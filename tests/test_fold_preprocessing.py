"""Per-fold preprocessing isolation on small synthetic matrices (no real data)."""
import os
import tempfile
import unittest

import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import MinMaxScaler

from scripts.fold_preprocessing import FoldPreprocessor


def synthetic(n=30, p=40, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.rand(n, p) * 10
    y = rng.randint(0, 3, size=n)
    # make a few features informative so selection is not arbitrary
    X[:, 0] += y * 5
    X[:, 1] -= y * 3
    return X, y


class TestFitUsesTrainingRowsOnly(unittest.TestCase):
    def setUp(self):
        self.X, self.y = synthetic()
        self.held = 7
        self.train_idx = np.array([i for i in range(len(self.y)) if i != self.held])

    def test_scaler_statistics_ignore_the_held_out_row(self):
        X = self.X.copy()
        X[self.held, 3] = 1e6          # extreme value only in the held-out row
        pp = FoldPreprocessor("anova", k=5).fit(X[self.train_idx], self.y[self.train_idx])
        self.assertLess(pp.scaler.data_max_[3], 1e5)
        np.testing.assert_allclose(pp.scaler.data_max_, X[self.train_idx].max(axis=0))
        np.testing.assert_allclose(pp.scaler.data_min_, X[self.train_idx].min(axis=0))

    def test_selector_matches_a_selector_fitted_on_train_rows_only(self):
        pp = FoldPreprocessor("anova", k=5).fit(self.X[self.train_idx], self.y[self.train_idx])
        ref_scaler = MinMaxScaler().fit(self.X[self.train_idx])
        ref = SelectKBest(f_classif, k=5).fit(ref_scaler.transform(self.X[self.train_idx]), self.y[self.train_idx])
        np.testing.assert_array_equal(pp.selected_indices, ref.get_support(indices=True))

    def test_selection_can_change_when_the_held_out_row_changes(self):
        # sanity: the selector is genuinely refitted per fold (different train sets may pick different columns)
        X, y = self.X.copy(), self.y.copy()
        picks = set()
        for held in range(6):
            idx = np.array([i for i in range(len(y)) if i != held])
            X2 = X.copy()
            X2[idx[0], 2 + held] += 50 * (y[idx[0]] + 1)  # perturb a different column each time
            picks.add(tuple(FoldPreprocessor("anova", k=5).fit(X2[idx], y[idx]).selected_indices))
        self.assertGreater(len(picks), 1)

    def test_transform_before_fit_raises(self):
        with self.assertRaises(RuntimeError):
            FoldPreprocessor().transform(self.X)

    def test_unknown_method_raises(self):
        with self.assertRaises(ValueError):
            FoldPreprocessor("pca", k=5)


class TestShapesAndPersistence(unittest.TestCase):
    def setUp(self):
        self.X, self.y = synthetic()

    def test_transform_shapes(self):
        pp = FoldPreprocessor("chi2", k=6).fit(self.X, self.y)
        self.assertEqual(pp.transform(self.X).shape, (30, 6))
        self.assertEqual(pp.transform(self.X[0]).shape, (1, 6))
        self.assertEqual(len(pp.selected_indices), 6)

    def test_k_is_capped_at_feature_count(self):
        pp = FoldPreprocessor("anova", k=100).fit(self.X, self.y)
        self.assertEqual(pp.transform(self.X).shape[1], self.X.shape[1])

    def test_held_out_row_is_transformed_with_training_statistics(self):
        train = self.X[:-1]
        pp = FoldPreprocessor("anova", k=5).fit(train, self.y[:-1])
        out = pp.transform(self.X[-1])
        expected = MinMaxScaler().fit(train).transform(self.X[-1:])[:, pp.selected_indices]
        np.testing.assert_allclose(out, expected)

    def test_all_three_methods_run(self):
        for m in ("anova", "chi2", "mutual_info"):
            self.assertEqual(FoldPreprocessor(m, k=4).fit(self.X, self.y).transform(self.X).shape, (30, 4))

    def test_save_load_roundtrip_reproduces_transform(self):
        pp = FoldPreprocessor("anova", k=5).fit(self.X[:-3], self.y[:-3])
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "preproc_0.json")
            pp.save(path)
            loaded = FoldPreprocessor.load(path)
        np.testing.assert_array_equal(loaded.selected_indices, pp.selected_indices)
        np.testing.assert_allclose(loaded.transform(self.X[-3:]), pp.transform(self.X[-3:]))
        self.assertEqual((loaded.method, loaded.k), ("anova", 5))


if __name__ == "__main__":
    unittest.main()
