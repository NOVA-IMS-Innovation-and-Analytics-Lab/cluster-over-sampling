"""
Test the density module.
"""

import pytest
import numpy as np

from sklearn.base import clone

from clover.distribution._density import DensityDistributor

X = np.array(
    [
        [1.0, 1.0],
        [1.0, 2.0],
        [1.5, 1.5],
        [-1.0, 1.0],
        [-1.0, 1.5],
        [-1.0, -1.0],
        [2.0, -1.0],
        [2.5, -1.0],
        [2.5, -1.5],
        [2.0, -1.5],
        [2.0, -2.0],
        [2.0, -2.5],
        [3.0, -1.0],
        [2.0, -1.0],
        [4.0, -1.0],
        [4.0, -1.0],
    ]
)
y_bin = np.array([1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0])
y_multi = np.array([0, 1, 2, 1, 2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 0])
LABELS = np.array([0, 0, 0, 1, 1, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4])
NEIGHBORS_BIN = np.array([(0, 1), (0, 2), (0, 3), (4, 2), (2, 3)])
NEIGHBORS_MULTI = np.array([(0, 1), (1, 4), (2, 3)])
DISTRIBUTOR = DensityDistributor(filtering_threshold=0.6, distances_exponent=1)


def test_filtered_clusters_binary():
    """Test the identification of filtered clusters for the binary case."""
    distributor = clone(DISTRIBUTOR).fit(X, y_bin, LABELS)
    assert distributor.filtered_clusters_ == [(0, 1), (2, 1), (4, 1)]


def test_filtered_clusters_multiclass():
    """Test the identification of filtered clusters for the multiclass case."""
    distributor = (
        clone(DISTRIBUTOR).set_params(filtering_threshold=1.0).fit(X, y_multi, LABELS)
    )
    assert distributor.filtered_clusters_ == [
        (0, 1),
        (0, 2),
        (1, 1),
        (1, 2),
        (4, 1),
        (4, 2),
    ]


def test_clusters_density_binary():
    """Test the calculation of filtered clusters density for the binary case."""
    distributor = clone(DISTRIBUTOR).fit(X, y_bin, LABELS)
    assert distributor.clusters_density_ == {(0, 1): 2.0, (2, 1): 2.25, (4, 1): 2.25}


def test_clusters_density_multiclass():
    """Test the calculation of filtered clusters density for the multiclass case."""
    distributor = (
        clone(DISTRIBUTOR).set_params(filtering_threshold=1.0).fit(X, y_multi, LABELS)
    )
    assert distributor.clusters_density_ == {
        (0, 1): 2.0,
        (0, 2): 1.0,
        (1, 1): 2.0,
        (1, 2): 1.0,
        (4, 1): 2.0,
        (4, 2): 1.0,
    }


def test_clusters_density_no_filtered():
    """Test the calculation clusters density when no filtered clusters are found."""
    X = np.arange(0.0, 5.0).reshape(-1, 1)
    y = np.array([0, 0, 0, 1, 1])
    labels = np.array([-1, -1, -1, -1, -1])
    distributor = clone(DISTRIBUTOR).set_params().fit(X, y, labels)
    assert distributor.clusters_density_ == {}


def test_filtering_threshold():
    """Test the filtering threshold parameter."""
    with pytest.raises(ValueError):
        clone(DISTRIBUTOR).set_params(filtering_threshold=-1.0).fit(X, y_bin, LABELS)
    with pytest.raises(TypeError):
        clone(DISTRIBUTOR).set_params(filtering_threshold=None).fit(X, y_bin, LABELS)
    with pytest.raises(TypeError):
        clone(DISTRIBUTOR).set_params(filtering_threshold='value').fit(X, y_bin, LABELS)


def test_distances_exponent():
    """Test the distances exponent parameter."""
    with pytest.raises(ValueError):
        clone(DISTRIBUTOR).set_params(distances_exponent=-1.0).fit(X, y_bin, LABELS)
    with pytest.raises(TypeError):
        clone(DISTRIBUTOR).set_params(distances_exponent=None).fit(X, y_bin, LABELS)
    with pytest.raises(TypeError):
        clone(DISTRIBUTOR).set_params(distances_exponent='value').fit(X, y_bin, LABELS)


def test_sparsity_based():
    """Test the sparsity based parameter."""
    with pytest.raises(TypeError):
        clone(DISTRIBUTOR).set_params(sparsity_based=None).fit(X, y_bin, LABELS)


def test_distribution_ratio():
    """Test the distribution ratio parameter."""
    with pytest.raises(ValueError):
        clone(DISTRIBUTOR).set_params(distribution_ratio=-1.0).fit(X, y_bin, LABELS)
    with pytest.raises(ValueError):
        clone(DISTRIBUTOR).set_params(distribution_ratio=2.0).fit(X, y_bin, LABELS)
    with pytest.raises(TypeError):
        clone(DISTRIBUTOR).set_params(distribution_ratio='value').fit(X, y_bin, LABELS)


def test_distribution_ratio_neighbor():
    """Test the distribution ratio parameter for no neighbors."""
    with pytest.raises(ValueError):
        clone(DISTRIBUTOR).set_params(neighbors=None, distribution_ratio=0.5).fit(
            X, y_bin, LABELS
        )


def test_fit_default():
    """Test the fit method for default initialization."""
    distributor = clone(DISTRIBUTOR).fit(X, y_bin, None, None)
    assert distributor.majority_class_label_ == 0
    assert hasattr(distributor, 'filtered_clusters_')
    assert hasattr(distributor, 'clusters_density_')
    np.testing.assert_array_equal(distributor.labels_, np.repeat(0, len(X)))
    np.testing.assert_array_equal(distributor.neighbors_, np.empty((0, 2)))
    assert distributor.intra_distribution_ == {(0, 1): 1.0}
    assert distributor.inter_distribution_ == {}


def test_fit_binary_intra():
    """Test the fit method for the binary case and intra-cluster generation."""
    distributor = clone(DISTRIBUTOR).fit(X, y_bin, LABELS)
    np.testing.assert_almost_equal(distributor.intra_distribution_[(0, 1)], 9.0 / 25.0)
    np.testing.assert_almost_equal(distributor.intra_distribution_[(2, 1)], 8.0 / 25.0)
    np.testing.assert_almost_equal(distributor.intra_distribution_[(4, 1)], 8.0 / 25.0)


def test_fit_multiclass_intra():
    """Test the fit method for the multiclass case and intra-cluster generation."""
    distributor = (
        clone(DISTRIBUTOR).set_params(filtering_threshold=1.0).fit(X, y_multi, LABELS)
    )
    np.testing.assert_almost_equal(distributor.intra_distribution_[(0, 1)], 1.0 / 3.0)
    np.testing.assert_almost_equal(distributor.intra_distribution_[(1, 1)], 1.0 / 3.0)
    np.testing.assert_almost_equal(distributor.intra_distribution_[(4, 1)], 1.0 / 3.0)
    np.testing.assert_almost_equal(distributor.intra_distribution_[(0, 2)], 1.0 / 3.0)
    np.testing.assert_almost_equal(distributor.intra_distribution_[(1, 2)], 1.0 / 3.0)
    np.testing.assert_almost_equal(distributor.intra_distribution_[(4, 2)], 1.0 / 3.0)


def test_fit_binary_inter():
    """Test the fit method for the binary case and inter-cluster generation."""
    distributor = (
        clone(DISTRIBUTOR)
        .set_params(distribution_ratio=0.0)
        .fit(X, y_bin, LABELS, NEIGHBORS_BIN)
    )
    np.testing.assert_equal(distributor.labels_, LABELS)
    np.testing.assert_equal(distributor.neighbors_, NEIGHBORS_BIN)
    np.testing.assert_almost_equal(
        distributor.inter_distribution_[((0, 1), (2, 1))], 18.0 / 35.0
    )
    np.testing.assert_almost_equal(
        distributor.inter_distribution_[((4, 1), (2, 1))], 17.0 / 35.0
    )


def test_fit_multiclass_inter():
    """Test the fit method for the multiclass case and inter-cluster generation."""
    distributor = (
        clone(DISTRIBUTOR)
        .set_params(distribution_ratio=0.0, filtering_threshold=1.0)
        .fit(X, y_multi, LABELS, NEIGHBORS_MULTI)
    )
    np.testing.assert_equal(distributor.labels_, LABELS)
    np.testing.assert_equal(distributor.neighbors_, NEIGHBORS_MULTI)
    np.testing.assert_almost_equal(
        distributor.inter_distribution_[((0, 1), (1, 1))], 0.5
    )
    np.testing.assert_almost_equal(
        distributor.inter_distribution_[((1, 1), (4, 1))], 0.5
    )
    np.testing.assert_almost_equal(
        distributor.inter_distribution_[((0, 2), (1, 2))], 0.5
    )
    np.testing.assert_almost_equal(
        distributor.inter_distribution_[((1, 2), (4, 2))], 0.5
    )


def test_fit_binary_intra_inter():
    """Test the fit method for the binary case, intra-cluster
    and inter-cluster generation."""
    distributor = (
        clone(DISTRIBUTOR)
        .set_params(distribution_ratio=0.5)
        .fit(X, y_bin, LABELS, NEIGHBORS_BIN)
    )
    np.testing.assert_equal(distributor.labels_, LABELS)
    np.testing.assert_equal(distributor.neighbors_, NEIGHBORS_BIN)
    np.testing.assert_almost_equal(distributor.intra_distribution_[(0, 1)], 9.0 / 50.0)
    np.testing.assert_almost_equal(distributor.intra_distribution_[(2, 1)], 8.0 / 50.0)
    np.testing.assert_almost_equal(distributor.intra_distribution_[(4, 1)], 8.0 / 50.0)
    np.testing.assert_almost_equal(
        distributor.inter_distribution_[((0, 1), (2, 1))], 18.0 / 70.0
    )
    np.testing.assert_almost_equal(
        distributor.inter_distribution_[((4, 1), (2, 1))], 17.0 / 70.0
    )


def test_fit_multiclass_intra_inter():
    """Test the fit method for the multiclass case, intra-cluster
    and inter-cluster generation."""
    distributor = (
        clone(DISTRIBUTOR)
        .set_params(distribution_ratio=0.5, filtering_threshold=1.0)
        .fit(X, y_multi, LABELS, NEIGHBORS_MULTI)
    )
    np.testing.assert_equal(distributor.labels_, LABELS)
    np.testing.assert_equal(distributor.neighbors_, NEIGHBORS_MULTI)
    np.testing.assert_almost_equal(distributor.intra_distribution_[(0, 1)], 1.0 / 6.0)
    np.testing.assert_almost_equal(distributor.intra_distribution_[(1, 1)], 1.0 / 6.0)
    np.testing.assert_almost_equal(distributor.intra_distribution_[(4, 1)], 1.0 / 6.0)
    np.testing.assert_almost_equal(distributor.intra_distribution_[(0, 2)], 1.0 / 6.0)
    np.testing.assert_almost_equal(distributor.intra_distribution_[(1, 2)], 1.0 / 6.0)
    np.testing.assert_almost_equal(distributor.intra_distribution_[(4, 2)], 1.0 / 6.0)
    np.testing.assert_almost_equal(
        distributor.inter_distribution_[((0, 1), (1, 1))], 0.25
    )
    np.testing.assert_almost_equal(
        distributor.inter_distribution_[((1, 1), (4, 1))], 0.25
    )
    np.testing.assert_almost_equal(
        distributor.inter_distribution_[((0, 2), (1, 2))], 0.25
    )
    np.testing.assert_almost_equal(
        distributor.inter_distribution_[((1, 2), (4, 2))], 0.25
    )
