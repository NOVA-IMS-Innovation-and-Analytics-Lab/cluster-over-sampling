"""
Includes the clustering base class for oversampling.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: BSD 3 clause

from abc import abstractmethod
from collections import Counter, OrderedDict

import numpy as np
from sklearn.base import clone, BaseEstimator, ClusterMixin
from sklearn.utils import check_random_state
from sklearn.neighbors import NearestNeighbors
from imblearn.over_sampling.base import BaseOverSampler
from imblearn.over_sampling import RandomOverSampler as _RandomOverSampler
from joblib import Parallel, delayed

from ..distribution.base import BaseDistributor
from ..distribution import DensityDistributor


def modify_nn(n_neighbors, n_samples):
    """Modify nearest neighbors object or integer."""
    if isinstance(n_neighbors, NearestNeighbors):
        n_neighbors = (
            clone(n_neighbors).set_params(n_neighbors=n_samples - 1)
            if n_neighbors.n_neighbors >= n_samples
            else clone(n_neighbors)
        )
    elif isinstance(n_neighbors, int) and n_neighbors >= n_samples:
        n_neighbors = n_samples - 1
    return n_neighbors


def clone_modify(oversampler, class_label, y_in_cluster):
    """Clone and modify attributes of oversampler for corner cases."""

    # Clone oversampler
    oversampler = clone(oversampler)

    # Not modify attributes case
    if isinstance(oversampler, _RandomOverSampler):
        return oversampler

    # Select and modify oversampler
    n_minority_samples = Counter(y_in_cluster)[class_label]
    if n_minority_samples == 1:
        oversampler = _RandomOverSampler()

        def _fit_resample_cluster(X, y):
            X_res, y_res = oversampler._fit_resample(X, y)
            X_new, y_new = X_res[len(X) :], y_res[len(X) :]
            return X_new, y_new

        oversampler._fit_resample_cluster = _fit_resample_cluster
    else:
        if hasattr(oversampler, 'k_neighbors'):
            oversampler.k_neighbors = modify_nn(
                oversampler.k_neighbors, n_minority_samples
            )
        if hasattr(oversampler, 'm_neighbors'):
            oversampler.m_neighbors = modify_nn(
                oversampler.m_neighbors, y_in_cluster.size
            )
        if hasattr(oversampler, 'n_neighbors'):
            oversampler.n_neighbors = modify_nn(
                oversampler.n_neighbors, n_minority_samples
            )
    return oversampler


def extract_intra_data(X, y, cluster_labels, intra_distribution, sampling_strategy):
    """Extract data for each filtered cluster."""
    majority_class_label = Counter(y).most_common()[0][0]
    clusters_data = []
    for (cluster_label, class_label), proportion in intra_distribution.items():
        mask = (cluster_labels == cluster_label) & (
            np.isin(y, [majority_class_label, class_label])
        )
        n_minority_samples = int(round(sampling_strategy[class_label] * proportion))
        X_in_cluster, y_in_cluster = X[mask], y[mask]
        cluster_sampling_strategy = {class_label: n_minority_samples}
        if n_minority_samples > 0:
            clusters_data.append(
                (cluster_sampling_strategy, X_in_cluster, y_in_cluster)
            )
    return clusters_data


def extract_inter_data(
    X, y, cluster_labels, inter_distribution, sampling_strategy, random_state
):
    """Extract data between filtered clusters."""
    majority_class_label = Counter(y).most_common()[0][0]
    clusters_data = []
    for (
        ((cluster_label1, class_label1), (cluster_label2, class_label2)),
        proportion,
    ) in inter_distribution.items():
        mask1 = (cluster_labels == cluster_label1) & (
            np.isin(y, [majority_class_label, class_label1])
        )
        mask2 = (cluster_labels == cluster_label2) & (
            np.isin(y, [majority_class_label, class_label2])
        )
        X1, X2, y1, y2 = X[mask1], X[mask2], y[mask1], y[mask2]
        majority_mask1, majority_mask2 = (
            (y1 == majority_class_label),
            (y2 == majority_class_label),
        )
        n_minority_samples = int(round(sampling_strategy[class_label1] * proportion))
        for _ in range(n_minority_samples):
            ind1, ind2 = (
                random_state.randint(0, (~majority_mask1).sum()),
                random_state.randint(0, (~majority_mask2).sum()),
            )
            X_in_clusters = np.vstack(
                (
                    X1[~majority_mask1][ind1].reshape(1, -1),
                    X2[~majority_mask2][ind2].reshape(1, -1),
                    X1[majority_mask1],
                    X2[majority_mask2],
                )
            )
            y_in_clusters = np.hstack(
                (
                    y1[~majority_mask1][ind1],
                    y2[~majority_mask2][ind2],
                    y1[majority_mask1],
                    y2[majority_mask2],
                )
            )
            clusters_sampling_strategy = {class_label1: 1}
            clusters_data.append(
                (clusters_sampling_strategy, X_in_clusters, y_in_clusters)
            )
    return clusters_data


def generate_in_cluster(
    oversampler, cluster_sampling_strategy, X_in_cluster, y_in_cluster
):
    """Generate intra-cluster or inter-cluster new samples."""

    # Create oversampler for specific cluster and class
    oversampler = clone_modify(oversampler, *cluster_sampling_strategy, y_in_cluster)
    oversampler.sampling_strategy_ = cluster_sampling_strategy

    # Resample cluster and class data
    X_new, y_new, *indices_new = oversampler._fit_resample_cluster(
        X_in_cluster, y_in_cluster
    )

    return X_new, y_new, indices_new


class _SingleLabelClusterer(BaseEstimator, ClusterMixin):
    """Clusterer that predicts a single label."""

    def fit(self, X, y=None, **fit_params):
        """

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Input data.

        y : Ignored

        """
        self.labels_ = np.zeros(len(X), dtype=int)
        return self

    def predict(self, X):
        """

        Parameters
        ----------

        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Input data.

        Returns
        -------

        labels : ndarray, shape(n_samples)
            Labelled data.

        """
        labels = np.zeros(len(X), dtype=int)
        return labels


class BaseClusterOverSampler(BaseOverSampler):
    """An extension of the base class for over-sampling algorithms to
    handle clustering based oversampling.

    Warning: This class should not be used directly. Use the derive classes
    instead.
    """

    _clusterer_docstring = """clusterer : clusterer estimator, (default=None)
        Clusterer to apply to input space before over-sampling.

        - When ``None``, it corresponds to a clusterer that assigns
          a single cluster to all the samples.

        - When clusterer, it applies clustering to the input space. Then
          over-sampling is applied inside each cluster and between clusters.
        """.strip()

    _distributor_docstring = """\n    \n    distributor : distributor estimator, (default=None)
        Distributor to distribute the generated samples per cluster label.

        - When ``None``, it corresponds to the density distributor. If clusterer
          is also ``None`` than the distributor does not affect the over-sampling
          procedure. If a clusterer is used than the distributor is the default
          density distributor.

        - When distributor, the generated samples are distributed to the clusters
          based on it.
        """

    def __init__(
        self,
        clusterer=None,
        distributor=None,
        sampling_strategy='auto',
        random_state=None,
        n_jobs=1,
        ratio=None,
    ):
        super(BaseClusterOverSampler, self).__init__(sampling_strategy, ratio)
        self.clusterer = clusterer
        self.distributor = distributor
        self.random_state = random_state
        self.n_jobs = n_jobs

    @abstractmethod
    def _fit_resample_cluster(self, X, y):
        """Basic resample of the dataset for each cluster.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Matrix containing the data which have to be sampled.

        y : array-like, shape (n_samples,)
            Corresponding label for each sample in X.

        Returns
        -------
        X_resampled : {ndarray, sparse matrix}, shape (n_samples_new, n_features)
            The array containing the resampled data.

        y_resampled : ndarray, shape (n_samples_new,)
            The corresponding label of `X_resampled`
        """
        pass

    def _cluster_sample(self, clusters_data, X, y):
        """Generate artificial data inside clusters or between clusters.
        """
        generated_data = Parallel(n_jobs=self.n_jobs)(
            delayed(generate_in_cluster)(self, *data) for data in clusters_data
        )
        if not generated_data:
            X_new, y_new, indices_new = (
                np.empty(shape=(0, X.shape[1]), dtype=X.dtype),
                np.empty(shape=(0,), dtype=y.dtype),
                np.empty(shape=(0,), dtype=int),
            )
        else:
            X_new, y_new, indices_new = [
                np.concatenate(data) for data in zip(*generated_data)
            ]

        # Random oversampling case
        if indices_new.tolist():
            indices_new = indices_new.reshape(-1)
            self.sample_indices_ = indices_new.copy()
            return X_new, y_new, indices_new

        return X_new, y_new

    def _intra_sample(self, X, y):
        """Intracluster resampling."""
        clusters_data = extract_intra_data(
            X,
            y,
            self.clusterer_.labels_,
            self.distributor_.intra_distribution_,
            self.sampling_strategy_,
        )
        return self._cluster_sample(clusters_data, X, y)

    def _inter_sample(self, X, y):
        """Intercluster resampling."""
        clusters_data = extract_inter_data(
            X,
            y,
            self.clusterer_.labels_,
            self.distributor_.inter_distribution_,
            self.sampling_strategy_,
            self.random_state_,
        )
        return self._cluster_sample(clusters_data, X, y)

    def _initialize(self, X, y, **fit_params):
        """Initialize fitting process."""

        super(BaseClusterOverSampler, self).fit(X, y)

        # Check clusterer and distributor
        if self.clusterer is None and self.distributor is not None:
            raise ValueError(
                'Distributor was found but clusterer is set to `None`. Set parameter `distributor` to `None` or use a clusterer.'
            )
        elif self.clusterer is None and self.distributor is None:
            self.clusterer_ = _SingleLabelClusterer()
            self.distributor_ = BaseDistributor()
        else:
            self.clusterer_ = clone(self.clusterer)
            self.distributor_ = (
                DensityDistributor(1.0)
                if self.distributor is None
                else clone(self.distributor)
            )

        # Fit clusterer
        self.clusterer_.fit(X, y, **fit_params)

        # Extract labels and neighbors
        labels = self.clusterer_.labels_
        neighbors = getattr(self.clusterer_, 'neighbors_', None)

        # fit distributor
        self.distributor_.fit(X, y, labels=labels, neighbors=neighbors)

        # Check random state
        self.random_state_ = check_random_state(self.random_state)

    def _fit_resample(self, X, y, **fit_params):
        """Resample the dataset.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Matrix containing the data which have to be sampled.

        y : array-like, shape (n_samples,)
            Corresponding label for each sample in X.

        Returns
        -------
        X_resampled : {ndarray, sparse matrix}, shape (n_samples_new, n_features)
            The array containing the resampled data.

        y_resampled : ndarray, shape (n_samples_new,)
            The corresponding label of `X_resampled`
        """

        # Initialization
        self._initialize(X, y, **fit_params)

        # Intracluster oversampling
        X_intra_new, y_intra_new, *indices_new = self._intra_sample(X, y)

        # Intercluster oversampling
        X_inter_new, y_inter_new = self._inter_sample(X, y)

        # Set sampling strategy
        intra_count, inter_count = Counter(y_intra_new), Counter(y_inter_new)
        self.sampling_strategy_ = OrderedDict({})
        for class_label in set(intra_count.keys()).union(inter_count.keys()):
            self.sampling_strategy_[class_label] = intra_count.get(
                class_label, 0
            ) + inter_count.get(class_label, 0)

        # Stack resampled data
        X_resampled, y_resampled = (
            np.vstack((X, X_intra_new, X_inter_new)),
            np.hstack((y, y_intra_new, y_inter_new)),
        )

        if indices_new:
            indices = np.hstack((np.arange(len(X)), indices_new[0]))
            return X_resampled, y_resampled, indices

        return X_resampled, y_resampled
