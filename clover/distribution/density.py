"""
Includes the DensityDistributor class.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: BSD 3 clause

from collections import Counter
from itertools import product

import numpy as np
from sklearn.utils import check_scalar
from sklearn.metrics.pairwise import euclidean_distances

from .base import BaseDistributor


class DensityDistributor(BaseDistributor):
    """Class to perform density based distribution.

    Samples are distributed based on the density of clusters.

    Parameters
    ----------
    filtering_threshold : float or 'auto', optional (default=1.0)
        The threshold of a cluster. It can be any non-negative number. If
        ``'auto'``, the filtering threshold is calculated from the imbalance
        ratio of the target for the binary case or the maximum imbalance ratio
        of the target for the multiclass case. Any cluster that has an imbalance
        ratio smaller than the filtering threshold is identified as a filtered
        cluster and can be potentially used to generate minority class
        instances. Higher values increase the number of filtered clusters.

    distances_exponent : float or 'auto', optional (default=0.0)
        The exponent of the mean distance in the density calculation. It can be
        any non-negative number. If ``'auto'`` then it is set equal to the number of
        features. Higher values make the calculation of density more sensitive
        to the cluster's size i.e. clusters with large mean euclidean distance
        between samples are penalized.

    sparsity_based : bool, optional (default=True)
        When ``True`` clusters receive generated samples that are inversly
        proportional to their density. When ``False`` clusters receive
        generated samples that are proportional to their density.

    distribution_ratio : float, optional (default=1.0)
        The ratio of intra-cluster to inter-cluster generated samples. It is a
        number in the :math:`[0.0, 1.0]` range. As the number increases more
        intra-cluster samples are generated. Inter-cluster generation, i.e. when
        ``distribution_ratio`` is less than ``1.0``, requires a neighborhood structure
        for the clusters and it will raise an error when it is not found.

    Attributes
    ----------

    majority_class_label_ : int
        The majority class label.

    class_labels_ : array, shape (n_classes, )
        An array of unique class labels.

    n_samples_ : int
        The number of samples.

    labels_ : array, shape (n_samples,)
        Labels of each sample.

    neighbors_ : array, (n_neighboring_pairs, 2)
        An array that contains all neighboring pairs. Each row is
        a unique neighboring pair.

    intra_distribution_ : dict
        Each dict key is a multi-label tuple of shape ``(cluster_label, class_label)``

    inter_distribution_ : dict
        Each dict key is a multi-label tuple of
        shape ``((cluster_label1, cluster_label2), class_label)``

    """

    def __init__(
        self,
        filtering_threshold=1.0,
        distances_exponent=0.0,
        sparsity_based=True,
        distribution_ratio=1.0,
    ):
        self.filtering_threshold = filtering_threshold
        self.distances_exponent = distances_exponent
        self.sparsity_based = sparsity_based
        self.distribution_ratio = distribution_ratio

    def _check_parameters(self, X, y, neighbors):
        """Check distributor parameters."""

        # Filtering threshold
        if self.filtering_threshold == 'auto':
            counts_vals = Counter(y).values()
            self.filtering_threshold_ = max(counts_vals) / min(counts_vals)
        else:
            check_scalar(
                self.filtering_threshold, 'filtering_threshold', (int, float), 0
            )
            self.filtering_threshold_ = self.filtering_threshold

        # Distances exponent
        if self.distances_exponent == 'auto':
            self.distances_exponent_ = X.shape[1]
        else:
            check_scalar(self.distances_exponent, 'distances_exponent', (int, float), 0)
            self.distances_exponent_ = self.distances_exponent

        # Sparsity based
        check_scalar(self.sparsity_based, 'sparsity_based', bool)
        self.sparsity_based_ = self.sparsity_based

        # distribution ratio
        check_scalar(self.distribution_ratio, 'distribution_ratio', float, 0.0, 1.0)
        if self.distribution_ratio < 1.0 and neighbors is None:
            raise ValueError(
                'Parameter `distribution_ratio` should be equal to 1.0, '
                'when `neighbors` parameter is `None`.'
            )
        self.distribution_ratio_ = self.distribution_ratio

    def _identify_filtered_clusters(self, y):
        """Identify the filtered clusters."""

        # Generate multi-label
        multi_labels = list(zip(self.labels_, y))

        # Count multi-label
        multi_labels_counts = Counter(multi_labels)

        # Extract unique cluster and class labels
        unique_multi_labels = [
            multi_label
            for multi_label in multi_labels_counts.keys()
            if multi_label[1] != self.majority_class_label_
        ]

        # Identify filtered clusters
        self.filtered_clusters_ = []
        for multi_label in unique_multi_labels:
            n_minority_samples = multi_labels_counts[multi_label]
            n_majority_samples = multi_labels_counts[
                (multi_label[0], self.majority_class_label_)
            ]
            if n_majority_samples <= n_minority_samples * self.filtering_threshold_:
                self.filtered_clusters_.append(multi_label)

    def _calculate_clusters_density(self, X, y):
        """Calculate the density of the filtered clusters."""

        self.clusters_density_ = dict()

        # Calculate density
        for cluster_label, class_label in self.filtered_clusters_:

            # Calculate number of majority and minority samples in each cluster
            mask = (self.labels_ == cluster_label) & (y == class_label)
            n_minority_samples = mask.sum()

            # Identify filtered clusters
            n_minority_pairs = (
                (n_minority_samples - 1) * n_minority_samples
                if n_minority_samples > 1
                else 1
            )
            mean_distances = euclidean_distances(X[mask]).sum() / n_minority_pairs
            self.clusters_density_[(cluster_label, class_label)] = (
                n_minority_samples / (mean_distances ** self.distances_exponent_)
                if mean_distances > 0
                else np.inf
            )

        # Convert infinite densities to finite
        class_labels = set(
            [class_label for _, class_label in self.clusters_density_.keys()]
        )
        max_densities = {}
        for class_label in class_labels:
            densities = [
                density
                for label, density in self.clusters_density_.items()
                if label[1] == class_label
            ]
            finite_densities = set(densities).difference([np.inf])
            max_densities[class_label] = (
                max(finite_densities) if len(finite_densities) > 0 else 1.0
            )
        self.clusters_density_ = {
            label: float(max_densities[label[1]] if np.isinf(density) else density)
            for label, density in self.clusters_density_.items()
        }

    def _intra_distribute(self, X, y, labels, neighbors):
        """Distribute the generated samples in each cluster based on their density."""

        # Calculate weights based on density
        weights = {
            multi_label: (1 / density if self.sparsity_based_ else density)
            for multi_label, density in self.clusters_density_.items()
        }

        # Calculate normalization factors
        class_labels = set([class_label for _, class_label in self.filtered_clusters_])
        normalization_factors = {class_label: 0.0 for class_label in class_labels}
        for (_, class_label), weight in weights.items():
            normalization_factors[class_label] += weight

        # Intra distribution
        self.intra_distribution_ = {
            multi_label: (
                self.distribution_ratio_
                * weight
                / normalization_factors[multi_label[1]]
            )
            for multi_label, weight in weights.items()
        }

        return self

    def _inter_distribute(self, X, y, labels, neighbors):
        """Distribute the generated samples between clusters based on their density."""

        # Identify filtered neighboring clusters
        filtered_neighbors = []
        class_labels = set([class_label for _, class_label in self.filtered_clusters_])
        for pair, class_label in product(self.neighbors_, class_labels):
            multi_label0 = (pair[0], class_label)
            multi_label1 = (pair[1], class_label)
            if (
                multi_label0 in self.filtered_clusters_
                and multi_label1 in self.filtered_clusters_
            ):
                filtered_neighbors.append((multi_label0, multi_label1))

        # Calculate inter-cluster density
        inter_clusters_density = {
            multi_labels: (
                self.clusters_density_[multi_labels[0]]
                + self.clusters_density_[multi_labels[1]]
            )
            for multi_labels in filtered_neighbors
        }

        # Calculate weights based on density
        weights = {
            multi_labels: (1 / density if self.sparsity_based_ else density)
            for multi_labels, density in inter_clusters_density.items()
        }

        # Calculate normalization factors
        normalization_factors = {class_label: 0.0 for class_label in class_labels}
        for multi_labels, weight in weights.items():
            normalization_factors[multi_labels[0][1]] += weight

        # Intra distribution
        self.inter_distribution_ = {
            multi_labels: (
                (1 - self.distribution_ratio_)
                * weight
                / normalization_factors[multi_labels[0][1]]
            )
            for multi_labels, weight in weights.items()
        }

        return self

    def _fit(self, X, y, labels, neighbors):

        # Check distributor parameters
        self._check_parameters(X, y, neighbors)

        # Identify filtered clusters
        self._identify_filtered_clusters(y)

        # Calculate density of filtered clusters
        self._calculate_clusters_density(X, y)

        super(DensityDistributor, self)._fit(X, y, labels, neighbors)

        return self
