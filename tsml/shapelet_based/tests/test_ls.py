"""Tests for the Learning Shapelets classifier wrapper."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from tsml.shapelet_based import LearningShapeletClassifier
from tsml.utils.testing import generate_3d_test_data
from tsml.utils.validation import _check_optional_dependency


@pytest.mark.skipif(
    not _check_optional_dependency(
        "tslearn", "tslearn", "LearningShapeletClassifier", False
    ),
    reason="tslearn is not installed.",
)
def test_learning_shapelets_matches_tslearn():
    """Test multivariate data is passed to tslearn in the layout it expects."""
    from tslearn.shapelets import LearningShapelets

    X, y = generate_3d_test_data(
        n_samples=20, n_channels=2, series_length=20, random_state=0
    )
    X_tslearn = np.transpose(X, (0, 2, 1))
    params = {"max_iter": 10, "batch_size": 10, "total_lengths": 1, "random_state": 0}

    clf = LearningShapeletClassifier(**params).fit(X, y)
    expected = LearningShapelets(**params).fit(X_tslearn, y)

    # shapelets are (n_shapelets, shapelet_length, n_channels), with a length of
    # 0.15 * 20 time points
    assert clf.clf_.shapelets_as_time_series_.shape[1:] == (3, 2)
    assert_allclose(clf.predict_proba(X), expected.predict_proba(X_tslearn))
