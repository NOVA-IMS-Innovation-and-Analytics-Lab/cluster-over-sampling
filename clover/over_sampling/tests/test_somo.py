"""
Test the _somo module.
"""

from math import sqrt
from collections import OrderedDict, Counter

import pytest
from sklearn.base import clone
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE

from clover.over_sampling._somo import SOMO
from clover.distribution._density import DensityDistributor

SOM = pytest.importorskip('somlearn.SOM')

RANDOM_STATE = 2
X, y = make_classification(
    random_state=RANDOM_STATE,
    n_classes=3,
    n_samples=5000,
    n_features=10,
    n_clusters_per_class=2,
    weights=[0.25, 0.45, 0.3],
    n_informative=5,
)
SM = SOMO(random_state=RANDOM_STATE)


@pytest.mark.parametrize(
    'k_neighbors,imbalance_ratio_threshold,distances_exponent',
    [(3, 2.0, 'auto'), (5, 1.5, 8), (8, 'auto', 10)],
)
def test_fit(k_neighbors, imbalance_ratio_threshold, distances_exponent):
    """Test the fit method."""

    # Fit oversampler
    somo = clone(SM).fit(X, y)
    y_count = Counter(y)

    # Assert random state
    assert hasattr(somo, 'random_state_')

    # Assert oversampler
    assert isinstance(somo.oversampler_, SMOTE)
    assert somo.oversampler_.k_neighbors == somo.k_neighbors

    # Assert clusterer
    assert isinstance(somo.clusterer_, SOM)

    # Assert distributor
    assert isinstance(somo.distributor_, DensityDistributor)
    assert somo.distributor_.filtering_threshold == 1.0
    assert somo.distributor_.distances_exponent == 2.0
    assert somo.distributor_.distribution_ratio == somo.distribution_ratio

    # Assert sampling strategy
    assert somo.oversampler_.sampling_strategy == somo.sampling_strategy
    assert somo.sampling_strategy_ == OrderedDict(
        {0: y_count[1] - y_count[0], 2: y_count[1] - y_count[2]}
    )


def test_fit_default():
    """Test the fit method for default initialization of somo estimator."""

    # Fit oversampler
    somo = clone(SM).fit(X, y)

    # Create SOM instance with default parameters
    som = SOM()

    # Assert clusterer
    assert isinstance(somo.clusterer_, SOM)
    assert somo.clusterer_.n_rows == som.n_rows
    assert somo.clusterer_.n_columns == som.n_columns


@pytest.mark.parametrize('n_clusters', [5, 6, 12])
def test_fit_number_of_clusters(n_clusters):
    """Test the fit method for initialization of kmeans estimator with
    a number of clusters."""

    # Fit oversampler
    somo = clone(SM).set_params(som_estimator=n_clusters).fit(X, y)

    # Assert clusterer
    assert isinstance(somo.clusterer_, SOM)
    assert somo.clusterer_.n_rows == round(sqrt(somo.som_estimator))
    assert somo.clusterer_.n_columns == round(sqrt(somo.som_estimator))


@pytest.mark.parametrize('proportion', [0.0, 0.5, 1.0])
def test_fit_proportion_of_samples(proportion):
    """Test the fit method for initialization of kmeans estimator with
    the proportion of the number of samples."""

    # Fit oversampler
    somo = clone(SM).set_params(som_estimator=proportion).fit(X, y)

    # Assert clusterer
    assert isinstance(somo.clusterer_, SOM)
    assert somo.clusterer_.n_rows == round(
        sqrt((X.shape[0] - 1) * somo.som_estimator + 1)
    )
    assert somo.clusterer_.n_columns == round(
        sqrt((X.shape[0] - 1) * somo.som_estimator + 1)
    )


def test_fit_som_estimator():
    """Test the fit method for initialization of som estimator with
    a SOM clusterer."""

    # Fit oversampler
    somo = clone(SM).set_params(som_estimator=SOM()).fit(X, y)

    # Define som estimator
    som = SOM()

    # Assert clusterer
    assert isinstance(somo.clusterer_, type(som))
    assert somo.clusterer_.n_rows == som.n_rows
    assert somo.clusterer_.n_columns == som.n_columns


@pytest.mark.parametrize('som_estimator', [-3, 0, -1.5, 2.0])
def test_fit_wrong_value(som_estimator):
    """Test the raise of value error when fit method is invoked."""
    with pytest.raises(ValueError):
        clone(SM).set_params(som_estimator=som_estimator).fit(X, y)


@pytest.mark.parametrize('som_estimator', [AgglomerativeClustering, [3, 5]])
def test_fit_wrong_type(som_estimator):
    """Test the raise of type error when fit method is invoked."""
    with pytest.raises(TypeError):
        clone(SM).set_params(som_estimator=som_estimator).fit(X, y)


def test_fit_resample():
    """Test the fit and resample method."""

    # Fit oversampler
    somo = clone(SM)
    _, y_res = somo.fit_resample(X, y)

    # Assert clusterer is fitted
    assert hasattr(somo.clusterer_, 'labels_')
    assert hasattr(somo.clusterer_, 'neighbors_')

    # Assert distributor is fitted
    assert hasattr(somo.distributor_, 'intra_distribution_')
    assert hasattr(somo.distributor_, 'inter_distribution_')

    # Assert balanced resampled target
    assert len(set(Counter(y_res).values())) == 1
