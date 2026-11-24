"""Fast unit tests; no dataset download is required."""

from __future__ import annotations

import unittest

import numpy as np

from src.data_loader import bandpass_and_car
from src.download_data import parse_subjects
from src.features import LogVariance, OneVsRestCSP, make_csp_classifier


class DownloadParsingTests(unittest.TestCase):
    def test_subject_ranges_and_duplicates(self):
        self.assertEqual(parse_subjects(["1-3", "3", "9"]), [1, 2, 3, 9])


class PreprocessingTests(unittest.TestCase):
    def test_preprocessing_shape_and_car(self):
        rng = np.random.default_rng(42)
        signal = rng.normal(size=(1000, 22))
        processed = bandpass_and_car(signal, fs=250)
        self.assertEqual(processed.shape, signal.shape)
        np.testing.assert_allclose(processed.mean(axis=1), 0, atol=1e-10)

    def test_invalid_band_rejected(self):
        with self.assertRaises(ValueError):
            bandpass_and_car(np.zeros((100, 22)), fs=250, low_freq=30, high_freq=8)


class FeatureTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.labels = np.repeat(np.arange(4), 40)
        sources = rng.normal(size=(160, 4, 240))
        for class_index in range(4):
            mask = self.labels == class_index
            sources[mask, class_index, :] *= 4
        mixing = rng.normal(size=(6, 4))
        self.epochs = np.einsum("cs,nst->nct", mixing, sources)
        self.epochs += rng.normal(scale=0.05, size=self.epochs.shape)

    def test_csp_output_shape_and_finite_values(self):
        csp = OneVsRestCSP(n_components=2).fit(self.epochs, self.labels)
        features = csp.transform(self.epochs)
        self.assertEqual(features.shape, (160, 16))
        self.assertTrue(np.isfinite(features).all())
        self.assertEqual(csp.spatial_patterns_.shape, (4, 6, 4))

    def test_csp_trace_normalization_gives_scale_invariance(self):
        original_csp = OneVsRestCSP(n_components=2).fit(self.epochs, self.labels)
        scaled_csp = OneVsRestCSP(n_components=2).fit(self.epochs * 100, self.labels)
        np.testing.assert_allclose(original_csp.eigenvalues_, scaled_csp.eigenvalues_, rtol=1e-8, atol=1e-8)
        original = original_csp.transform(self.epochs)
        scaled = scaled_csp.transform(self.epochs * 100)
        np.testing.assert_allclose(original, scaled, rtol=1e-8, atol=1e-8)

    def test_log_variance_baseline(self):
        features = LogVariance().fit_transform(self.epochs)
        self.assertEqual(features.shape, (160, 6))
        self.assertTrue(np.isfinite(features).all())

    def test_end_to_end_classifier_on_synthetic_spatial_patterns(self):
        train = np.arange(160) % 4 != 0
        test = ~train
        model = make_csp_classifier(n_components=2)
        model.fit(self.epochs[train], self.labels[train])
        accuracy = np.mean(model.predict(self.epochs[test]) == self.labels[test])
        self.assertGreater(accuracy, 0.90)


if __name__ == "__main__":
    unittest.main()
