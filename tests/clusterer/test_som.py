"""Test the som module."""

from clover.clusterer import SOM, extract_topological_neighbors, generate_labels_mapping
from sklearn.datasets import make_classification

RANDOM_STATE = 10
X, _ = make_classification(random_state=RANDOM_STATE)


def test_generate_labels_mapping():
    """Test the generation of the labels mapping."""
    grid_labels = [(1, 1), (0, 0), (0, 1), (1, 0), (1, 1), (1, 0), (0, 1)]
    labels_mapping = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3}
    assert generate_labels_mapping(grid_labels) == labels_mapping


def test_return_topological_neighbors_rectangular():
    """Test the topological neighbors of a neuron for rectangular grid type."""
    som = SOM(random_state=RANDOM_STATE).fit(X)
    assert set(
        extract_topological_neighbors(0, 0, som.gridtype, som.n_rows, som.n_columns, som.algorithm_.bmus.tolist()),
    ).issubset({(1, 0), (0, 1)})
    assert set(
        extract_topological_neighbors(1, 1, som.gridtype, som.n_rows, som.n_columns, som.algorithm_.bmus.tolist()),
    ).issubset({(0, 1), (2, 1), (1, 0), (1, 2)})


def test_return_topological_neighbors_hexagonal():
    """Test the topological neighbors of a neuron for hexagonal grid type."""
    som = SOM(random_state=RANDOM_STATE, gridtype='hexagonal').fit(X)
    assert set(
        extract_topological_neighbors(0, 0, som.gridtype, som.n_rows, som.n_columns, som.algorithm_.bmus.tolist()),
    ).issubset({(1, 0), (0, 1)})
    assert set(
        extract_topological_neighbors(1, 1, som.gridtype, som.n_rows, som.n_columns, som.algorithm_.bmus.tolist()),
    ).issubset({(0, 1), (2, 1), (1, 0), (1, 2), (2, 2), (2, 0)})


def test_fit():
    """Test the SOM fitting process."""

    som = SOM(random_state=RANDOM_STATE)
    assert not hasattr(som, 'labels_')
    assert not hasattr(som, 'neighbors_')
    assert not hasattr(som, 'algorithm_')
    assert not hasattr(som, 'n_columns_')
    assert not hasattr(som, 'n_rows_')

    som.fit(X)
    assert hasattr(som, 'labels_')
    assert hasattr(som, 'neighbors_')
    assert hasattr(som, 'algorithm_')
