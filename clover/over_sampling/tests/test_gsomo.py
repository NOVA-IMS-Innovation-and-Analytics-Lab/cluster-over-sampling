"""
Test the _gsomo module.
"""

from math import sqrt
from collections import OrderedDict, Counter

import pytest
from sklearn.base import clone
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_classification

from clover.over_sampling._gsomo import GeometricSOMO
from clover.distribution._density import DensityDistributor

GeometricSMOTE = pytest.importorskip('gsmote.GeometricSMOTE')
SOM = pytest.importorskip('somlearn.SOM')

RANDOM_STATE = 11
X, y = make_classification(
    random_state=RANDOM_STATE,
    n_classes=3,
    n_samples=5000,
    n_features=10,
    n_clusters_per_class=2,
    weights=[0.3, 0.45, 0.25],
    n_informative=5,
)
GSOMO = GeometricSOMO(random_state=RANDOM_STATE)


@pytest.mark.parametrize(
    'k_neighbors,imbalance_ratio_threshold,distances_exponent',
    [(3, 2.0, 'auto'), (5, 1.5, 8), (8, 'auto', 10)],
)
def test_fit(k_neighbors, imbalance_ratio_threshold, distances_exponent):
    """Test the fit method."""

    # Fit oversampler
    gsomo = clone(GSOMO).fit(X, y)
    y_count = Counter(y)

    # Assert random state
    assert hasattr(gsomo, 'random_state_')

    # Assert oversampler
    assert isinstance(gsomo.oversampler_, GeometricSMOTE)
    assert gsomo.oversampler_.k_neighbors == gsomo.k_neighbors
    assert gsomo.oversampler_.truncation_factor == gsomo.truncation_factor
    assert gsomo.oversampler_.deformation_factor == gsomo.deformation_factor
    assert gsomo.oversampler_.selection_strategy == gsomo.selection_strategy

    # Assert clusterer
    assert isinstance(gsomo.clusterer_, SOM)

    # Assert distributor
    assert isinstance(gsomo.distributor_, DensityDistributor)
    assert gsomo.distributor_.filtering_threshold == gsomo.imbalance_ratio_threshold
    assert gsomo.distributor_.distances_exponent == gsomo.distances_exponent
    assert gsomo.distributor_.distribution_ratio == gsomo.distribution_ratio

    # Assert sampling strategy
    assert gsomo.oversampler_.sampling_strategy == gsomo.sampling_strategy
    assert gsomo.sampling_strategy_ == OrderedDict(
        {0: y_count[1] - y_count[0], 2: y_count[1] - y_count[2]}
    )


def test_fit_default():
    """Test the fit method for default initialization of gsomo estimator."""

    # Fit oversampler
    gsomo = clone(GSOMO).fit(X, y)

    # Create SOM instance with default parameters
    som = SOM()

    # Assert clusterer
    assert isinstance(gsomo.clusterer_, SOM)
    assert gsomo.clusterer_.n_rows == som.n_rows
    assert gsomo.clusterer_.n_columns == som.n_columns


@pytest.mark.parametrize('n_clusters', [5, 6, 12])
def test_fit_number_of_clusters(n_clusters):
    """Test the fit method for initialization of kmeans estimator with
    a number of clusters."""

    # Fit oversampler
    gsomo = clone(GSOMO).set_params(som_estimator=n_clusters).fit(X, y)

    # Assert clusterer
    assert isinstance(gsomo.clusterer_, SOM)
    assert gsomo.clusterer_.n_rows == round(sqrt(gsomo.som_estimator))
    assert gsomo.clusterer_.n_columns == round(sqrt(gsomo.som_estimator))


@pytest.mark.parametrize('proportion', [0.0, 0.5, 1.0])
def test_fit_proportion_of_samples(proportion):
    """Test the fit method for initialization of kmeans estimator with
    the proportion of the number of samples."""

    # Fit oversampler
    gsomo = clone(GSOMO).set_params(som_estimator=proportion).fit(X, y)

    # Assert clusterer
    assert isinstance(gsomo.clusterer_, SOM)
    assert gsomo.clusterer_.n_rows == round(
        sqrt((X.shape[0] - 1) * gsomo.som_estimator + 1)
    )
    assert gsomo.clusterer_.n_columns == round(
        sqrt((X.shape[0] - 1) * gsomo.som_estimator + 1)
    )


def test_som_estimator():
    """Test the fit method for initialization of som estimator with
    a SOM clusterer."""

    # Fit oversampler
    gsomo = clone(GSOMO).set_params(som_estimator=SOM()).fit(X, y)

    # Define som estimator
    som = SOM()

    # Assert clusterer
    assert isinstance(gsomo.clusterer_, type(som))
    assert gsomo.clusterer_.n_rows == som.n_rows
    assert gsomo.clusterer_.n_columns == som.n_columns


@pytest.mark.parametrize('som_estimator', [-3, 0, -1.5, 2.0])
def test_fit_wrong_value(som_estimator):
    """Test the raise of value error when fit method is invoked."""
    with pytest.raises(ValueError):
        clone(GSOMO).set_params(som_estimator=som_estimator).fit(X, y)


@pytest.mark.parametrize('som_estimator', [AgglomerativeClustering, [3, 5]])
def test_fit_wrong_type(som_estimator):
    """Test the raise of type error when fit method is invoked."""
    with pytest.raises(TypeError):
        clone(GSOMO).set_params(som_estimator=som_estimator).fit(X, y)


def test_fit_resample():
    """Test the fit and resample method."""

    # Fit oversampler
    gsomo = clone(GSOMO)
    _, y_res = gsomo.fit_resample(X, y)

    # Assert clusterer is fitted
    assert hasattr(gsomo.clusterer_, 'labels_')
    assert hasattr(gsomo.clusterer_, 'neighbors_')

    # Assert distributor is fitted
    assert hasattr(gsomo.distributor_, 'intra_distribution_')
    assert hasattr(gsomo.distributor_, 'inter_distribution_')

    # Assert almost balanced resampled target
    count = Counter(y_res).values()
    assert max(count) - min(count) < 3
