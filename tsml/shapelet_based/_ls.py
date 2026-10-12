"""Learning Shapelets classifier wrapper."""

__author__ = ["MatthewMiddlehurst"]
__all__ = ["LearningShapeletClassifier"]

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted

from tsml.base import BaseTimeSeriesEstimator
from tsml.utils.validation import _check_optional_dependency


class LearningShapeletClassifier(ClassifierMixin, BaseTimeSeriesEstimator):
    """Learning Shapelets classifier.

    This is a wrapper for the LearningShapelets estimator from the `tslearn`
    package [2]_. As well as `tslearn`, it requires `keras` 3 and a `keras` backend,
    for which `tslearn` expects `torch`.

    Learning Shapelets [1]_ learns a set of shapelets jointly with a logistic
    regression on the shapelet transform of the data, where the transform is the
    minimum distance between each shapelet and the subsequences of a series. The
    shapelets and regression weights are optimised by gradient descent.

    Parameters
    ----------
    n_shapelets_per_size : dict or None, default=None
        Dictionary giving, for each shapelet size (key), the number of such
        shapelets to be trained (value). If None, the heuristic from [1]_ is used
        with ``shapelet_length`` and ``total_lengths``.
    max_iter : int, default=10000
        Number of training epochs.
    batch_size : int, default=256
        Batch size to be used.
    verbose : {0, 1, 2}, default=0
        `keras` verbose level.
    optimizer : str or keras.optimizers.Optimizer, default="sgd"
        `keras` optimizer to use for training.
    weight_regularizer : float, default=0.0
        Strength of the L2 regularizer to use for training the classification
        (softmax) layer. If 0, no regularization is performed.
    shapelet_length : float, default=0.15
        The length of the shapelets, expressed as a fraction of the series length.
        Only used if ``n_shapelets_per_size`` is None.
    total_lengths : int, default=3
        The number of different shapelet lengths. Shapelets of length
        ``i * shapelet_length`` for ``i`` in ``[1, total_lengths]`` are extracted.
        Only used if ``n_shapelets_per_size`` is None.
    max_size : int or None, default=None
        Maximum size for time series to be fed to the model. If None, it is set to
        the series length of the training data.
    scale : bool, default=False
        Whether input data should be scaled for each channel of each series to lie
        in the [0, 1] interval.
    random_state : int or None, default=None
        Seed for the random number generation. An int is passed to `keras`, which
        seeds the global Python, NumPy and backend random number generators.

    Attributes
    ----------
    classes_ : np.ndarray
        The class labels.
    n_classes_ : int
        The number of classes.
    clf_ : tslearn.shapelets.LearningShapelets
        The fitted `tslearn` estimator. It takes input of shape
        (n_instances, n_timepoints, n_channels).

    Notes
    -----
    The `keras` backend is fixed the first time `keras` is imported in a process.
    `tslearn` selects `torch` when it is installed, unless a backend is set through
    the ``KERAS_BACKEND`` environment variable or `keras` has already been imported.

    References
    ----------
    .. [1] Grabocka, J., Schilling, N., Wistuba, M. and Schmidt-Thieme, L., 2014.
       Learning time-series shapelets. In Proceedings of the 20th ACM SIGKDD
       international conference on Knowledge discovery and data mining (pp. 392-401).
    .. [2] Tavenard, R. et al., 2020. Tslearn, a machine learning toolkit for time
       series data. Journal of Machine Learning Research, 21(118), pp.1-6.
       https://github.com/tslearn-team/tslearn
    """

    def __init__(
        self,
        n_shapelets_per_size=None,
        max_iter=10000,
        batch_size=256,
        verbose=0,
        optimizer="sgd",
        weight_regularizer=0.0,
        shapelet_length=0.15,
        total_lengths=3,
        max_size=None,
        scale=False,
        random_state=None,
    ):
        self.n_shapelets_per_size = n_shapelets_per_size
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.verbose = verbose
        self.optimizer = optimizer
        self.weight_regularizer = weight_regularizer
        self.shapelet_length = shapelet_length
        self.total_lengths = total_lengths
        self.max_size = max_size
        self.scale = scale
        self.random_state = random_state

        # keras is not checked here, importing it would fix the keras backend before
        # tslearn has selected one
        _check_optional_dependency("tslearn", "tslearn", self)

        super().__init__()

    def fit(self, X: np.ndarray | list[np.ndarray], y: np.ndarray) -> object:
        """Fit the estimator to training data.

        Parameters
        ----------
        X : 3D np.ndarray of shape (n_instances, n_channels, n_timepoints)
            The training data.
        y : 1D np.ndarray of shape (n_instances)
            The class labels for fitting, indices correspond to instance indices in X

        Returns
        -------
        self :
            Reference to self.
        """
        X, y = self._validate_data(
            X=X, y=y, ensure_min_samples=2, ensure_equal_length=True
        )
        X = self._convert_X(X)

        check_classification_targets(y)

        self.n_instances_, self.n_channels_, self.n_timepoints_ = X.shape
        self.classes_ = np.unique(y)
        self.n_classes_ = self.classes_.shape[0]
        self.class_dictionary_ = {}
        for index, class_val in enumerate(self.classes_):
            self.class_dictionary_[class_val] = index

        if self.n_classes_ == 1:
            return self

        from tslearn.shapelets import LearningShapelets

        self.clf_ = LearningShapelets(
            n_shapelets_per_size=self.n_shapelets_per_size,
            max_iter=self.max_iter,
            batch_size=self.batch_size,
            verbose=self.verbose,
            optimizer=self.optimizer,
            weight_regularizer=self.weight_regularizer,
            shapelet_length=self.shapelet_length,
            total_lengths=self.total_lengths,
            max_size=self.max_size,
            scale=self.scale,
            random_state=self.random_state,
        )
        self.clf_.fit(_convert_data(X), y)

        return self

    def predict(self, X: np.ndarray | list[np.ndarray]) -> np.ndarray:
        """Predicts labels for sequences in X.

        Parameters
        ----------
        X : 3D np.array of shape (n_instances, n_channels, n_timepoints)
            The testing data.

        Returns
        -------
        y : array-like of shape (n_instances)
            Predicted class labels.
        """
        check_is_fitted(self)

        # treat case of single class seen in fit
        if self.n_classes_ == 1:
            return np.repeat(list(self.class_dictionary_.keys()), X.shape[0], axis=0)

        X = self._validate_data(X=X, reset=False, ensure_equal_length=True)
        X = self._convert_X(X)

        return self.clf_.predict(_convert_data(X))

    def predict_proba(self, X: np.ndarray | list[np.ndarray]) -> np.ndarray:
        """Predicts labels probabilities for sequences in X.

        Parameters
        ----------
        X : 3D np.array of shape (n_instances, n_channels, n_timepoints)
            The testing data.

        Returns
        -------
        y : array-like of shape (n_instances, n_classes_)
            Predicted probabilities using the ordering in classes_.
        """
        check_is_fitted(self)

        # treat case of single class seen in fit
        if self.n_classes_ == 1:
            return np.repeat([[1]], X.shape[0], axis=0)

        X = self._validate_data(X=X, reset=False, ensure_equal_length=True)
        X = self._convert_X(X)

        return self.clf_.predict_proba(_convert_data(X))

    def _more_tags(self) -> dict:
        return {"optional_dependency": True}

    @classmethod
    def get_test_params(cls, parameter_set: str | None = None) -> dict | list[dict]:
        """Return unit test parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : None or str, default=None
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict
            Parameters to create testing instances of the class.
        """
        return {
            "max_iter": 10,
            "batch_size": 10,
            "total_lengths": 1,
        }


def _convert_data(X):
    # tslearn uses (n_instances, n_timepoints, n_channels)
    return np.transpose(X, (0, 2, 1))
