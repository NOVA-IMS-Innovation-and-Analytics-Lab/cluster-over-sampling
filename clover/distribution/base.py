"""
Base class for distributors.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections import Counter

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils import check_X_y, check_array


class BaseDistributor(BaseEstimator):
    """The base class for distributors. A distributor sets the proportion of
    samples to be generated inside each cluster and between clusters.

    Warning: This class should not be used directly. Use the derive classes
    instead.
    """

    def _intra_distribute(self, X, y, labels, neighbors):
        """Distribute the generated samples in each cluster based on their density."""
        return self

    def _inter_distribute(self, X, y, labels, neighbors):
        """Distribute the generated samples between clusters based on their density."""
        return self

    def _validate_fitting(self):
        """Validate consistency of fitting procedure."""

        # Check labels
        if len(self.labels_) != self.n_samples_:
            raise ValueError(
                f'Number of labels should be equal to the number of samples. '
                f'Got {len(self.labels_)} and {self.n_samples_} instead.'
            )

        # Check neighbors
        if not set(self.labels_).issuperset(self.neighbors_.flatten()):
            raise ValueError('Attribute `neighbors_` contains unknown labels.')
        unique_neighbors = set([tuple(set(pair)) for pair in self.neighbors_])
        if len(unique_neighbors) < len(self.neighbors_):
            raise ValueError('Elements of `neighbors_` attribute are not unique.')

        # Check distribution
        proportions = {
            class_label: 0.0
            for class_label in self.unique_class_labels_
            if class_label != self.majority_class_label_
        }
        for (_, class_label), proportion in self.intra_distribution_.items():
            proportions[class_label] += proportion
        for (
            ((cluster_label1, class_label1), (cluster_label2, class_label2)),
            proportion,
        ) in self.inter_distribution_.items():
            if class_label1 != class_label2:
                multi_label = (
                    (cluster_label1, class_label1),
                    (cluster_label2, class_label2),
                )
                raise ValueError(
                    f'Multi-labels for neighboring cluster pairs should '
                    f'have a common class label. Got {multi_label} instead.'
                )
            proportions[class_label1] += proportion
        if not all(
            [
                np.isclose(val, 0.0) or np.isclose(val, 1.0)
                for val in proportions.values()
            ]
        ):
            raise ValueError(
                f'Intra-distribution and inter-distribution sum of proportions '
                f'for each class label should be either equal to 0.0 or 1.0. '
                f'Got {proportions} instead.'
            )

        return self

    def _fit(self, X, y, labels, neighbors):
        if labels is not None:
            self._intra_distribute(X, y, labels, neighbors)
        if neighbors is not None:
            self._inter_distribute(X, y, labels, neighbors)
        return self

    def fit(self, X, y, labels=None, neighbors=None):
        """Generate the intra-label and inter-label distribution.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Matrix containing the data which have to be sampled.

        y : array-like, shape (n_samples,)
            Corresponding label for each sample in X.

        labels : array-like, shape (n_samples,)
            Labels of each sample.

        neighbors : array-like, (n_neighboring_pairs, 2)
            An array that contains all neighboring pairs. Each row is
            a unique neighboring pair.

        Returns
        -------
        self : object,
            Return self.

        """

        # Check data
        X, y = check_X_y(X, y, dtype=None)

        # Set statistics
        self.majority_class_label_ = Counter(y).most_common()[0][0]
        self.unique_cluster_labels_ = (
            np.unique(labels) if labels is not None else np.array(0, dtype=int)
        )
        self.unique_class_labels_ = np.unique(y)
        self.n_samples_ = len(X)

        # Set default attributes
        self.labels_ = (
            np.repeat(0, len(X))
            if labels is None
            else check_array(labels, ensure_2d=False)
        )
        self.neighbors_ = (
            np.empty((0, 2), dtype=int)
            if neighbors is None
            else check_array(neighbors, ensure_2d=False)
        )
        self.intra_distribution_ = {
            (0, class_label): 1.0
            for class_label in np.unique(y)
            if class_label != self.majority_class_label_
        }
        self.inter_distribution_ = {}

        # Fit distributor
        self._fit(X, y, labels, neighbors)

        # Validate fitting procedure
        self._validate_fitting()

        return self

    def fit_distribute(self, X, y, labels=None, neighbors=None):
        """Return the intra-label and inter-label distribution.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Matrix containing the data which have to be sampled.

        y : array-like, shape (n_samples,)
            Corresponding label for each sample in X.

        labels : array-like, shape (n_samples,)
            Labels of each sample.

        neighbors : array-like, shape (n_neighboring_pairs, 2)
            An array that contains all neighboring pairs. Each row is
            a unique neighboring pair.

        Returns
        -------
        distributions : tuple of (intra_distribution, inter_distribution) arrays
            A tuple with the two distributions.

        """
        self.fit(X, y, labels, neighbors)
        return self.intra_distribution_, self.inter_distribution_
