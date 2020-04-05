"""
Test the _kmeans_smote module.
"""

from collections import OrderedDict, Counter

import pytest
from sklearn.base import clone
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE

from clover.over_sampling._kmeans_smote import KMeansSMOTE
from clover.distribution._density import DensityDistributor

RANDOM_STATE = 1
X, y = make_classification(
    random_state=RANDOM_STATE,
    n_classes=3,
    n_samples=5000,
    n_features=10,
    n_clusters_per_class=2,
    weights=[0.25, 0.45, 0.3],
    n_informative=5,
)
KMEANS_SMOTE = KMeansSMOTE(random_state=RANDOM_STATE)


@pytest.mark.parametrize(
    'k_neighbors,imbalance_ratio_threshold,distances_exponent',
    [(3, 2.0, 'auto'), (5, 1.5, 8), (8, 'auto', 10)],
)
def test_fit(k_neighbors, imbalance_ratio_threshold, distances_exponent):
    """Test the fit method."""

    # Fit oversampler
    kmeans_smote = clone(KMEANS_SMOTE).fit(X, y)
    y_count = Counter(y)

    # Assert random state
    assert hasattr(kmeans_smote, 'random_state_')

    # Assert oversampler
    assert isinstance(kmeans_smote.oversampler_, SMOTE)
    assert kmeans_smote.oversampler_.k_neighbors == kmeans_smote.k_neighbors

    # Assert clusterer
    assert isinstance(kmeans_smote.clusterer_, MiniBatchKMeans)

    # Assert distributor
    assert isinstance(kmeans_smote.distributor_, DensityDistributor)
    assert (
        kmeans_smote.distributor_.filtering_threshold
        == kmeans_smote.imbalance_ratio_threshold
    )
    assert (
        kmeans_smote.distributor_.distances_exponent == kmeans_smote.distances_exponent
    )

    # Assert sampling strategy
    assert kmeans_smote.oversampler_.sampling_strategy == kmeans_smote.sampling_strategy
    assert kmeans_smote.sampling_strategy_ == OrderedDict(
        {0: y_count[1] - y_count[0], 2: y_count[1] - y_count[2]}
    )


def test_fit_default():
    """Test the fit method for default initialization of kmeans estimator."""

    # Fit oversampler
    kmeans_smote = clone(KMEANS_SMOTE).fit(X, y)

    # Assert clusterer
    assert isinstance(kmeans_smote.clusterer_, MiniBatchKMeans)
    assert kmeans_smote.clusterer_.n_clusters == MiniBatchKMeans().n_clusters


@pytest.mark.parametrize('n_clusters', [5, 6, 12])
def test_fit_number_of_clusters(n_clusters):
    """Test the fit method for initialization of kmeans estimator with
    a number of clusters."""

    # Fit oversampler
    kmeans_smote = clone(KMEANS_SMOTE).set_params(kmeans_estimator=n_clusters).fit(X, y)

    # Assert clusterer
    assert isinstance(kmeans_smote.clusterer_, MiniBatchKMeans)
    assert kmeans_smote.clusterer_.n_clusters == n_clusters


@pytest.mark.parametrize('proportion', [0.0, 0.5, 1.0])
def test_fit_proportion_of_samples(proportion):
    """Test the fit method for initialization of kmeans estimator with
    the proportion of the number of samples."""

    # Fit oversampler
    kmeans_smote = clone(KMEANS_SMOTE).set_params(kmeans_estimator=proportion).fit(X, y)

    # Assert clusterer
    assert isinstance(kmeans_smote.clusterer_, MiniBatchKMeans)
    assert kmeans_smote.clusterer_.n_clusters == round((len(X) - 1) * proportion + 1)


@pytest.mark.parametrize('kmeans_estimator', [KMeans(), MiniBatchKMeans()])
def test_fit_kmeans_estimator(kmeans_estimator):
    """Test the fit method for initialization of kmeans estimator with
    a KMeans or MiniBatchKMeans clusterer."""

    # Fit oversampler
    kmeans_smote = (
        clone(KMEANS_SMOTE).set_params(kmeans_estimator=kmeans_estimator).fit(X, y)
    )

    # Assert clusterer
    assert isinstance(kmeans_smote.clusterer_, type(kmeans_estimator))
    assert kmeans_smote.clusterer_.n_clusters == kmeans_estimator.n_clusters


@pytest.mark.parametrize('kmeans_estimator', [-3, 0, -1.5, 2.0])
def test_fit_wrong_value(kmeans_estimator):
    """Test the raise of value error when fit method is invoked."""
    with pytest.raises(ValueError):
        clone(KMEANS_SMOTE).set_params(kmeans_estimator=kmeans_estimator).fit(X, y)


@pytest.mark.parametrize('kmeans_estimator', [AgglomerativeClustering, [3, 5]])
def test_fit_wrong_type(kmeans_estimator):
    """Test the raise of type error when fit method is invoked."""
    with pytest.raises(TypeError):
        clone(KMEANS_SMOTE).set_params(kmeans_estimator=kmeans_estimator).fit(X, y)


def test_fit_resample():
    """Test the fit and resample method."""

    # Fit oversampler
    kmeans_smote = clone(KMEANS_SMOTE)
    _, y_res = kmeans_smote.fit_resample(X, y)

    # Assert clusterer is fitted
    assert hasattr(kmeans_smote.clusterer_, 'labels_')
    assert not hasattr(kmeans_smote.clusterer_, 'neighbors_')

    # Assert distributor is fitted
    assert hasattr(kmeans_smote.distributor_, 'intra_distribution_')
    assert hasattr(kmeans_smote.distributor_, 'inter_distribution_')

    # Assert balanced resampled target
    assert len(set(Counter(y_res).values())) == 1
