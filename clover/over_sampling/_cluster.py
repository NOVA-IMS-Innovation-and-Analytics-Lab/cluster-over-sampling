"""
Implementation of the main class for
clustering-based over-sampling.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import warnings
from collections import Counter, OrderedDict

import numpy as np
from sklearn.base import clone
from sklearn.utils import check_random_state
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import label_binarize
from sklearn.utils.multiclass import check_classification_targets
from sklearn.exceptions import FitFailedWarning
from imblearn.over_sampling.base import BaseOverSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from imblearn.utils import Substitution, check_sampling_strategy
from imblearn.utils._docstring import _random_state_docstring, _n_jobs_docstring
from imblearn.utils._validation import ArraysTransformer
from joblib import Parallel, delayed

from ..distribution.base import BaseDistributor
from ..distribution import DensityDistributor


def _modify_nn(n_neighbors, n_samples):
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


def _clone_modify(oversampler, class_label, y_in_cluster):
    """Clone and modify attributes of oversampler for corner cases."""

    # Clone oversampler
    oversampler = clone(oversampler)

    # Not modify attributes case
    if isinstance(oversampler, RandomOverSampler):
        return oversampler

    # Select and modify oversampler
    n_minority_samples = Counter(y_in_cluster)[class_label]
    if n_minority_samples == 1:
        oversampler = RandomOverSampler()
    else:
        if hasattr(oversampler, 'k_neighbors'):
            oversampler.k_neighbors = _modify_nn(
                oversampler.k_neighbors, n_minority_samples
            )
        if hasattr(oversampler, 'm_neighbors'):
            oversampler.m_neighbors = _modify_nn(
                oversampler.m_neighbors, y_in_cluster.size
            )
        if hasattr(oversampler, 'n_neighbors'):
            oversampler.n_neighbors = _modify_nn(
                oversampler.n_neighbors, n_minority_samples
            )
    return oversampler


def _extract_intra_data(X, y, cluster_labels, intra_distribution, sampling_strategy):
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


def _extract_inter_data(
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


def _generate_in_cluster(
    oversampler, cluster_sampling_strategy, X_in_cluster, y_in_cluster
):
    """Generate intra-cluster or inter-cluster new samples."""

    # Pipeline case
    if isinstance(oversampler, Pipeline):
        transformer, oversampler = oversampler.steps[:-1], oversampler.steps[-1][-1]
        X_in_cluster = Pipeline(transformer).fit_transform(X_in_cluster, y_in_cluster)

    # Create oversampler for specific cluster and class
    oversampler = _clone_modify(oversampler, *cluster_sampling_strategy, y_in_cluster)
    oversampler.sampling_strategy_ = cluster_sampling_strategy

    # Resample cluster and class data
    X_res, y_res = oversampler._fit_resample(X_in_cluster, y_in_cluster)

    # Filter only new data
    X_new, y_new = X_res[len(X_in_cluster):], y_res[len(y_in_cluster):]

    return X_new, y_new


@Substitution(random_state=_random_state_docstring, n_jobs=_n_jobs_docstring)
class ClusterOverSampler(BaseOverSampler):
    """A class that handles clustering-based over-sampling.

    Any combination of over-sampler, clusterer and distributor can
    be used.

    Read more in the :ref:`user guide <user_guide>`.

    Parameters
    ----------
    oversampler : oversampler estimator, default=None
        Over-sampler to apply to each selected cluster.

    clusterer : clusterer estimator, default=None
        Clusterer to apply to input space before over-sampling.

        - When ``None``, it corresponds to a clusterer that assigns
          a single cluster to all the samples i.e. no clustering is applied.

        - When clusterer, it applies clustering to the input space. Then
          over-sampling is applied inside each cluster and between clusters.

    distributor : distributor estimator, default=None
        Distributor to distribute the generated samples per cluster label.

        - When ``None`` and a clusterer is provided then it corresponds to the
          density distributor. If clusterer is also ``None`` than the distributor
          does not affect the over-sampling procedure.

        - When distributor object is provided, it is used to distribute the
          generated samples to the clusters.

    raise_error : bool, default=True
        Raise an error when no samples are generated.

        - If ``True``, it raises an error when no filtered clusters are
          identified and therefore no samples are generated.

        - If ``False``, it displays a warning.

    {random_state}

    {n_jobs}

    Attributes
    ----------
    clusterer_ : object
        A fitted clone of the ``clusterer`` parameter or ``None`` when a
        clusterer is not given.

    distributor_ : object
        A fitted clone of the ``clusterer`` parameter or a fitted instance of
        the ``BaseDistributor`` when a distributor is not given.

    labels_ : array, shape (n_samples,)
        Labels of each sample.

    neighbors_ : array, (n_neighboring_pairs, 2) or None
        An array that contains all neighboring pairs with each row being
        a unique neighboring pair. It is ``None`` when the clusterer does not
        support this attribute.

    oversampler_ : object
        A fitted clone of the ``oversampler`` parameter.

    random_state_ : object
        An instance of ``RandomState`` class.

    sampling_strategy_ : dict
        Actual sampling strategy.

    Examples
    --------
    >>> from collections import Counter
    >>> from clover.over_sampling import ClusterOverSampler
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.cluster import KMeans
    >>> from imblearn.over_sampling import SMOTE
    >>> X, y = make_classification(random_state=0, n_classes=2, weights=[0.9, 0.1])
    >>> print('Original dataset shape %s' % Counter(y))
    Original dataset shape Counter({{0: 90, 1: 10}})
    >>> cluster_oversampler = ClusterOverSampler(
    ... oversampler=SMOTE(random_state=5),
    ... clusterer=KMeans(random_state=10))
    >>> X_res, y_res = cluster_oversampler.fit_resample(X, y)
    >>> print('Resampled dataset shape %s' % Counter(y_res))
    Resampled dataset shape Counter({{0: 90, 1: 90}})
    """

    def __init__(
        self,
        oversampler,
        clusterer=None,
        distributor=None,
        raise_error=True,
        random_state=None,
        n_jobs=None,
    ):
        self.oversampler = oversampler
        self.clusterer = clusterer
        self.distributor = distributor
        self.raise_error = raise_error
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y):
        """Check inputs and statistics of the sampler.

        You should use ``fit_resample`` in all cases.

        Parameters
        ----------
        X : {array-like, dataframe, sparse matrix} of shape \
                (n_samples, n_features)
            Data array.
        y : array-like of shape (n_samples,)
            Target array.

        Returns
        -------
        self : object
            Return the instance itself.
        """
        X, y, _ = self._check_X_y(X, y)
        self._initialize_fitting(X, y)
        return self

    def fit_resample(self, X, y, **fit_params):
        """Resample the dataset.

        Parameters
        ----------
        X : {array-like, dataframe, sparse matrix} of shape \
                (n_samples, n_features)
            Matrix containing the data which have to be sampled.
        y : array-like of shape (n_samples,)
            Corresponding label for each sample in X.

        Returns
        -------
        X_resampled : {array-like, dataframe, sparse matrix} of shape \
                (n_samples_new, n_features)
            The array containing the resampled data.
        y_resampled : array-like of shape (n_samples_new,)
            The corresponding label of `X_resampled`.
        """
        check_classification_targets(y)
        arrays_transformer = ArraysTransformer(X, y)
        X, y, binarize_y = self._check_X_y(X, y)

        self._initialize_fitting(X, y)._fit(X, y, **fit_params)

        output = self._fit_resample(X, y)

        y_ = (
            label_binarize(y=output[1], classes=np.unique(y))
            if binarize_y
            else output[1]
        )

        X_, y_ = arrays_transformer.transform(output[0], y_)
        return (X_, y_) if len(output) == 2 else (X_, y_, output[2])

    def _cluster_sample(self, clusters_data, X, y):
        """Generate artificial data inside clusters or between clusters.
        """
        generated_data = Parallel(n_jobs=self.n_jobs)(
            delayed(_generate_in_cluster)(self.oversampler_, *data)
            for data in clusters_data
        )
        if not generated_data:
            X_new, y_new = (
                np.empty(shape=(0, X.shape[1]), dtype=X.dtype),
                np.empty(shape=(0,), dtype=y.dtype),
            )
        else:
            X_new, y_new = [np.concatenate(data) for data in zip(*generated_data)]

        return X_new, y_new

    def _intra_sample(self, X, y):
        """Intracluster resampling."""
        clusters_data = _extract_intra_data(
            X,
            y,
            self.labels_,
            self.distributor_.intra_distribution_,
            self.sampling_strategy_,
        )
        return self._cluster_sample(clusters_data, X, y)

    def _inter_sample(self, X, y):
        """Intercluster resampling."""
        clusters_data = _extract_inter_data(
            X,
            y,
            self.labels_,
            self.distributor_.inter_distribution_,
            self.sampling_strategy_,
            self.random_state_,
        )
        return self._cluster_sample(clusters_data, X, y)

    def _initialize_fitting(self, X, y):
        """Initialize fitting process."""

        # Check random state
        self.random_state_ = check_random_state(self.random_state)

        # Check oversampler
        self.oversampler_ = clone(self.oversampler)
        self.sampling_strategy_ = check_sampling_strategy(
            self.oversampler_.sampling_strategy
            if isinstance(self.oversampler_, BaseOverSampler)
            else self.oversampler_.steps[-1][-1].sampling_strategy,
            y,
            self._sampling_type,
        )

        # Check clusterer and distributor
        if self.clusterer is None and self.distributor is not None:
            raise ValueError(
                'Distributor was found but clusterer is set to `None`. '
                'Either set parameter `distributor` to `None` or use a clusterer.'
            )
        elif self.clusterer is None and self.distributor is None:
            self.clusterer_ = None
            self.distributor_ = BaseDistributor()
        else:
            self.clusterer_ = clone(self.clusterer)
            self.distributor_ = (
                DensityDistributor()
                if self.distributor is None
                else clone(self.distributor)
            )

        return self

    def _fit(self, X, y, **fit_params):
        """Fit the clusterer and distributor."""

        # Fit clusterer
        if self.clusterer_ is not None:
            self.clusterer_.fit(X, y, **fit_params)

        # Extract labels and neighbors
        self.labels_ = getattr(self.clusterer_, 'labels_', np.zeros(len(X), dtype=int))
        self.neighbors_ = getattr(self.clusterer_, 'neighbors_', None)

        # fit distributor
        self.distributor_.fit(X, y, labels=self.labels_, neighbors=self.neighbors_)

        # Case when no samples are generated
        if (
            not self.distributor_.intra_distribution_
            and not self.distributor_.inter_distribution_
        ):
            msg = (
                'No samples were generated. Try to modify the parameters '
                'of the clusterer or distributor.'
            )

            # Raise error
            if self.raise_error:
                raise ValueError(msg)

            # Display warning
            else:
                warnings.warn(msg, FitFailedWarning)

        return self

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

        # Intracluster oversampling
        X_intra_new, y_intra_new = self._intra_sample(X, y)

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

        return X_resampled, y_resampled
