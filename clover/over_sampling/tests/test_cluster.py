"""
Test the cluster module.
"""

from collections import OrderedDict, Counter

import pytest
import numpy as np
from sklearn.utils import check_random_state
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.datasets import make_classification
from sklearn.neighbors import NearestNeighbors
from sklearn.exceptions import FitFailedWarning
from imblearn.over_sampling import (
    RandomOverSampler,
    SMOTE,
    BorderlineSMOTE,
    SVMSMOTE,
)

from clover.over_sampling._cluster import (
    _modify_nn,
    _clone_modify,
    _generate_in_cluster,
    _extract_intra_data,
    _extract_inter_data,
    ClusterOverSampler,
)
from clover.distribution._density import DensityDistributor

RANDOM_STATE = 1
X, y = make_classification(
    random_state=RANDOM_STATE,
    n_classes=3,
    n_samples=5000,
    n_features=10,
    n_clusters_per_class=2,
    weights=[0.2, 0.5, 0.3],
    n_informative=5,
)
CLUSTERER = KMeans(n_clusters=5, n_init=1, random_state=RANDOM_STATE)
OVERSAMPLERS = [
    RandomOverSampler(random_state=RANDOM_STATE),
    SMOTE(random_state=RANDOM_STATE),
    BorderlineSMOTE(random_state=RANDOM_STATE),
    SVMSMOTE(random_state=RANDOM_STATE),
]
CLUSTER_OVERSAMPLERS = [
    ClusterOverSampler(
        RandomOverSampler(random_state=RANDOM_STATE), clusterer=CLUSTERER
    ),
    ClusterOverSampler(SMOTE(random_state=RANDOM_STATE), clusterer=CLUSTERER),
    ClusterOverSampler(BorderlineSMOTE(random_state=RANDOM_STATE), clusterer=CLUSTERER),
    ClusterOverSampler(SVMSMOTE(random_state=RANDOM_STATE), clusterer=CLUSTERER),
]


def test_modify_nn_object():
    """Test the modification of nn object."""
    assert _modify_nn(NearestNeighbors(n_neighbors=5), 3).n_neighbors == 2
    assert _modify_nn(NearestNeighbors(n_neighbors=3), 3).n_neighbors == 2
    assert _modify_nn(NearestNeighbors(n_neighbors=2), 5).n_neighbors == 2


def test_modify_nn_int():
    """Test the modification of integer nn."""
    assert _modify_nn(5, 3) == 2
    assert _modify_nn(3, 3) == 2
    assert _modify_nn(2, 5) == 2


def test_clone_modify_ros():
    """Test the cloning and modification of random oversampler."""
    cloned_oversampler = _clone_modify(OVERSAMPLERS[0], None, None)
    assert isinstance(cloned_oversampler, RandomOverSampler)


@pytest.mark.parametrize(
    'oversampler',
    [ovs for ovs in OVERSAMPLERS if not isinstance(ovs, RandomOverSampler)],
)
def test_clone_modify_single_min_sample(oversampler):
    """Test the cloning and modification for one minority class sample."""
    class_label = 1
    y_in_cluster = np.array([0, 0, 0, 0, 1, 2, 2, 2])
    cloned_oversampler = _clone_modify(oversampler, class_label, y_in_cluster)
    assert isinstance(cloned_oversampler, RandomOverSampler)


@pytest.mark.parametrize(
    'oversampler',
    [ovs for ovs in OVERSAMPLERS if not isinstance(ovs, RandomOverSampler)],
)
def test_clone_modify_neighbors(oversampler):
    """Test the cloning and modification of neighbors based oversamplers."""
    class_label = 2
    y_in_cluster = np.array([0, 0, 0, 0, 1, 2, 2, 2])
    n_minority_samples = Counter(y_in_cluster)[class_label]
    cloned_oversampler = _clone_modify(oversampler, class_label, y_in_cluster)
    assert isinstance(cloned_oversampler, oversampler.__class__)
    if hasattr(cloned_oversampler, 'k_neighbors'):
        assert cloned_oversampler.k_neighbors == n_minority_samples - 1
    if hasattr(cloned_oversampler, 'm_neighbors'):
        assert (cloned_oversampler.m_neighbors == y_in_cluster.size - 1) or (
            cloned_oversampler.m_neighbors == 'deprecated'
        )
    if hasattr(cloned_oversampler, 'n_neighbors'):
        assert (cloned_oversampler.n_neighbors == n_minority_samples - 1) or (
            cloned_oversampler.n_neighbors == 'deprecated'
        )


def test_extract_intra_data():
    """Test the extraction of data for each filtered cluster."""
    X = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).reshape(-1, 1)
    y = np.array([0, 0, 0, 0, 1, 2, 2, 2, 0])
    cluster_labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    intra_distribution = {(1, 1): 1.0, (1, 2): 0.8, (2, 2): 0.2}
    sampling_strategy = OrderedDict({1: 4, 2: 2})
    clusters_data = _extract_intra_data(
        X, y, cluster_labels, intra_distribution, sampling_strategy
    )
    cluster_sampling_strategies, Xs, ys = zip(*clusters_data)
    assert cluster_sampling_strategies == ({1: 4}, {2: 2})
    assert [X.tolist() for X in Xs] == [[[4.0], [5.0]], [[4.0], [6.0]]]
    assert [y.tolist() for y in ys] == [[0, 1], [0, 2]]


def test_extract_inter_data():
    """Test the extraction of data between filtered clusters."""
    X = np.array(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    ).reshape(-1, 1)
    y = np.array([1, 0, 0, 0, 1, 2, 2, 2, 0, 0, 1, 0])
    cluster_labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2])
    inter_distribution = {
        ((0, 1), (1, 1)): 0.5,
        ((1, 1), (2, 1)): 0.5,
        ((1, 2), (2, 2)): 1.0,
    }
    sampling_strategy = OrderedDict({1: 3, 2: 3})
    random_state = check_random_state(RANDOM_STATE)
    clusters_data = _extract_inter_data(
        X, y, cluster_labels, inter_distribution, sampling_strategy, random_state
    )
    cluster_sampling_strategies, Xs, ys = zip(*clusters_data)
    assert cluster_sampling_strategies == (
        {1: 1},
        {1: 1},
        {1: 1},
        {1: 1},
        {2: 1},
        {2: 1},
        {2: 1},
    )
    assert [X.tolist() for X in Xs] == 2 * [[[1.0], [5.0], [2.0], [3.0], [4.0]]] + 2 * [
        [[5.0], [11.0], [4.0], [9.0], [10.0], [12.0]]
    ] + 2 * [[[6.0], [8.0], [4.0], [9.0], [10.0], [12.0]]] + [
        [[6.0], [7.0], [4.0], [9.0], [10.0], [12.0]]
    ]
    assert [y.tolist() for y in ys] == 2 * [[1, 1, 0, 0, 0]] + 2 * [
        [1, 1, 0, 0, 0, 0]
    ] + 3 * [[2, 2, 0, 0, 0, 0]]


@pytest.mark.parametrize('oversampler', OVERSAMPLERS)
def test_generate_in_cluster(oversampler):
    """Test the generation of new samples."""
    oversampler = clone(oversampler)

    X_in_cluster = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).reshape(-1, 1)
    y_in_cluster = np.array([0, 0, 0, 0, 1, 2, 2, 2])

    # First class
    cluster_sampling_strategy = {1: 5}
    (class_label,) = cluster_sampling_strategy
    X_new, y_new = _generate_in_cluster(
        oversampler, cluster_sampling_strategy, X_in_cluster, y_in_cluster
    )
    assert len(X_new) == len(y_new) <= cluster_sampling_strategy[1]
    np.testing.assert_array_equal(np.unique(X_new), np.array([5.0]))
    assert Counter(y_new)[class_label] == cluster_sampling_strategy[1]

    # Second class
    cluster_sampling_strategy = {2: 3}
    (class_label,) = cluster_sampling_strategy
    X_new, y_new = _generate_in_cluster(
        oversampler, cluster_sampling_strategy, X_in_cluster, y_in_cluster
    )
    assert len(X_new) == len(y_new) <= cluster_sampling_strategy[2]
    assert Counter(y_new)[class_label] <= cluster_sampling_strategy[2]


@pytest.mark.parametrize('oversampler', CLUSTER_OVERSAMPLERS)
def test_fit(oversampler):
    """Test the fit method."""
    oversampler = clone(oversampler).fit(X, y)
    y_count = Counter(y)
    assert hasattr(oversampler, 'sampling_strategy_')
    assert hasattr(oversampler, 'oversampler_')
    assert hasattr(oversampler, 'clusterer_')
    assert hasattr(oversampler, 'distributor_')
    assert hasattr(oversampler, 'random_state_')
    assert oversampler.sampling_strategy_ == OrderedDict(
        {0: y_count[1] - y_count[0], 2: y_count[1] - y_count[2]}
    )


@pytest.mark.parametrize('oversampler', CLUSTER_OVERSAMPLERS)
def test_fit_resample(oversampler):
    """Test the fit and resample method."""
    oversampler = clone(oversampler)
    oversampler.fit_resample(X, y)
    assert hasattr(oversampler, 'sampling_strategy_')
    assert hasattr(oversampler, 'oversampler_')
    assert hasattr(oversampler, 'clusterer_')
    assert hasattr(oversampler, 'distributor_')
    assert hasattr(oversampler, 'random_state_')
    assert hasattr(oversampler.distributor_, 'intra_distribution_')
    assert hasattr(oversampler.distributor_, 'inter_distribution_')


@pytest.mark.parametrize(
    'X,y,oversampler',
    [
        (
            np.array([(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0)]),
            np.array([0, 0, 1, 1, 1]),
            ClusterOverSampler(
                oversampler=SMOTE(k_neighbors=5, random_state=RANDOM_STATE)
            ),
        ),
        (
            np.array([(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0)]),
            np.array([0, 0, 1, 1, 1]),
            ClusterOverSampler(
                oversampler=SMOTE(k_neighbors=5, random_state=RANDOM_STATE),
                clusterer=CLUSTERER.set_params(n_clusters=3),
                random_state=RANDOM_STATE,
            ),
        ),
    ],
)
def test_fit_resample_intra_corner_cases(X, y, oversampler):
    """Test the fit_resample method for various
    corner cases and oversamplers."""
    X_res, y_res = oversampler.fit_resample(X, y)
    y_count = Counter(y_res)
    assert y_count[0] == y_count[1]
    assert X.item(0, 0) <= X_res.item(-1, 0) <= X.item(1, 0)
    assert X.item(0, 1) <= X_res.item(-1, 1) <= X.item(1, 1)


@pytest.mark.parametrize('oversampler', CLUSTER_OVERSAMPLERS)
def test_raise_error(oversampler):
    """Test the raise of an error when no samples
    are generated."""
    oversampler = clone(oversampler)
    oversampler.set_params(
        clusterer=CLUSTERER.set_params(n_clusters=2),
        distributor=DensityDistributor(filtering_threshold=0.1),
    )
    with pytest.raises(ValueError):
        oversampler.fit_resample(X, y)


@pytest.mark.parametrize('oversampler', CLUSTER_OVERSAMPLERS)
def test_display_warning(oversampler):
    """Test the display of a warning when no samples
    are generated."""
    oversampler = clone(oversampler)
    oversampler.set_params(
        clusterer=CLUSTERER.set_params(n_clusters=2),
        distributor=DensityDistributor(filtering_threshold=0.1),
        raise_error=False,
    )
    with pytest.warns(FitFailedWarning):
        oversampler.fit_resample(X, y)
