"""Implementation of the Self-Organizing Map (SOM) clusterer."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: BSD 3 clause

from itertools import product
from typing import Any, ClassVar, cast

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.preprocessing import minmax_scale
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array
from somoclu import Somoclu
from typing_extensions import Self


def generate_labels_mapping(grid_labels: list[tuple[int, int]]) -> dict[tuple[int, int], int]:
    """Generate a mapping between grid labels and cluster labels."""

    # Identify unique grid labels
    unique_labels = sorted(set(grid_labels))

    # Generate mapping
    labels_mapping = dict(zip(unique_labels, range(len(unique_labels)), strict=True))

    return labels_mapping


def extract_topological_neighbors(
    col: int,
    row: int,
    gridtype: str,
    n_rows: int,
    n_columns: int,
    bmus: list[list[int]],
) -> list[tuple[int, int]]:
    """Return the topological neighbors of a neuron."""

    # Return common topological neighbors for the two grid types
    topological_neighbors = [
        (col - 1, row),
        (col + 1, row),
        (col, row - 1),
        (col, row + 1),
    ]

    # Append extra topological neighbors for hexagonal grid type
    if gridtype == 'hexagonal':
        offset = (-1) ** row
        topological_neighbors += [
            (col - offset, row - offset),
            (col - offset, row + offset),
        ]

    # Apply constraints
    topological_neighbors = [
        (col, row)
        for col, row in topological_neighbors
        if 0 <= col < n_columns and 0 <= row < n_rows and [col, row] in bmus
    ]

    return topological_neighbors


class SOM(BaseEstimator, ClusterMixin):
    """Class to fit and visualize a Self-Organizing Map (SOM).

    The implementation uses SOM from Somoclu. Read more in the
    [user_guide].

    Args:
        n_columns:
            The number of columns in the map.

        n_rows:
            The number of rows in the map.

        initialcodebook:
            Define the codebook to start the training. If `initialcodebook='pca'` then
            the codebook is initialized from the first subspace spanned by the first two
            eigenvectors of the correlation matrix.

        kerneltype:
            Specify which kernel to use. If `kerneltype=0` use dense CPU kernel.
            Else if `kerneltype=1` use dense GPU kernel if compiled with it.

        maptype:
            Specify the map topology. If `maptype='planar'` use planar map.
            Else if `maptype='toroid'` use toroid map.

        gridtype:
            Specify the grid form of the nodes. If `gridtype='rectangular'`
            use rectangular neurons. Else if `gridtype='hexagonal'` use
            hexagonal neurons.

        compactsupport:
            Cut off map updates beyond the training radius with the Gaussian neighborhood.

        neighborhood:
            Specify the neighborhood. If `neighborhood='gaussian'` use
            Gaussian neighborhood. Else if `neighborhood='bubble'` use
            bubble neighborhood function.

        std_coeff:
            Set the coefficient in the Gaussian
            neighborhood :math:`exp(-||x-y||^2/(2*(coeff*radius)^2))`.

        random_state:
            Control the randomization of the algorithm by specifying the
            codebook initalization. It is ignored when `initialcodebook` is
            not `None`.

            - If int, `random_state` is the seed used by the random number
            generator.
            - If `RandomState` instance, random_state is the random number
            generator.
            - If `None`, the random number generator is the `RandomState`
            instance used by `np.random`.

        verbose:
            Specify verbosity level (0, 1, or 2).
    """

    _attributes: ClassVar = ['train', 'codebook', 'bmus']

    def __init__(
        self: Self,
        n_columns: int = 5,
        n_rows: int = 5,
        initialcodebook: npt.ArrayLike | str | None = None,
        kerneltype: int = 0,
        maptype: str = 'planar',
        gridtype: str = 'rectangular',
        compactsupport: bool = True,
        neighborhood: str = 'gaussian',
        std_coeff: float = 0.5,
        random_state: int | np.random.RandomState | None = None,
        verbose: int = 0,
    ) -> None:
        self.n_columns = n_columns
        self.n_rows = n_rows
        self.initialcodebook = initialcodebook
        self.kerneltype = kerneltype
        self.maptype = maptype
        self.gridtype = gridtype
        self.compactsupport = compactsupport
        self.neighborhood = neighborhood
        self.std_coeff = std_coeff
        self.random_state = random_state
        self.verbose = verbose

    def _generate_neighbors(
        self: Self,
        grid_labels_unique: list[tuple[int, int]],
        labels_mapping: dict[tuple[int, int], int],
    ) -> npt.NDArray:
        """Generate pairs of neighboring labels."""

        # Generate grid topological neighbors
        grid_topological_neighbors = [
            product(
                [tuple(grid_label)],
                extract_topological_neighbors(
                    *grid_label,
                    self.gridtype,
                    self.n_rows,
                    self.n_columns,
                    self.algorithm_.bmus.tolist(),
                ),
            )
            for grid_label in grid_labels_unique
        ]

        # Flatten grid topological neighbors
        grid_topological_neighbors_flat = cast(
            list[tuple[tuple[int, int], tuple[int, int]]],
            [pair for pairs in grid_topological_neighbors for pair in pairs],
        )

        # Generate cluster neighbors
        all_neighbors = sorted(
            {(labels_mapping[pair[0]], labels_mapping[pair[1]]) for pair in grid_topological_neighbors_flat},
        )

        # Keep unique unordered pairs
        neighbors = []
        for pair in all_neighbors:
            if pair not in neighbors and pair[::-1] not in neighbors:
                neighbors.append(pair)

        return np.array(neighbors)

    def fit(self: Self, X: npt.ArrayLike, y: npt.ArrayLike | None = None, **fit_params: dict[str, Any]) -> Self:
        """Train the self-organizing map.

        Args:
            X:
                Training instances to cluster.
            y:
                Ignored
            fit_params:
                Parameters to pass to train method of Somoclu object.

        Returns:
            The object itself.
        """

        # Check and normalize input data
        X_scaled = minmax_scale(check_array(X, dtype=np.float32))

        # Check random_state
        self.random_state_ = check_random_state(self.random_state)

        # Initialize codebook
        if self.initialcodebook is None:
            if self.random_state is None:
                initialcodebook = None
                initialization = 'random'
            else:
                codebook_size = self.n_columns * self.n_rows * X_scaled.shape[1]
                initialcodebook = self.random_state_.random_sample(
                    codebook_size,
                ).astype(np.float32)
                initialization = None
        elif self.initialcodebook == 'pca':
            initialcodebook = None
            initialization = 'random'
        else:
            initialcodebook = self.initialcodebook
            initialization = None

        # Create Somoclu object
        self.algorithm_ = Somoclu(
            n_columns=self.n_columns,
            n_rows=self.n_rows,
            initialcodebook=initialcodebook,
            kerneltype=self.kerneltype,
            maptype=self.maptype,
            gridtype=self.gridtype,
            compactsupport=self.compactsupport,
            neighborhood=self.neighborhood,
            std_coeff=self.std_coeff,
            initialization=initialization,
            data=None,
            verbose=self.verbose,
        )

        # Fit Somoclu
        self.algorithm_.train(data=X_scaled, **fit_params)

        # Grid labels
        grid_labels = cast(list[tuple[int, int]], [tuple(grid_label) for grid_label in self.algorithm_.bmus.tolist()])

        # Generate labels mapping
        self.labels_mapping_ = generate_labels_mapping(grid_labels)

        # Generate cluster labels
        self.labels_ = np.array(
            [self.labels_mapping_[grid_label] for grid_label in grid_labels],
        )

        # Generate labels neighbors
        self.neighbors_ = self._generate_neighbors(
            sorted(set(grid_labels)),
            self.labels_mapping_,
        )

        return self

    def fit_predict(
        self: Self,
        X: npt.ArrayLike,
        y: npt.ArrayLike | None = None,
        **fit_params: dict[str, Any],
    ) -> npt.NDArray:
        """Train the self-organizing map and assign cluster labels to samples.

        Args:
            X:
                New data to transform.
            y:
                Ignored.
            fit_params:
                Parameters to pass to train method of Somoclu object.

        Returns:
            labels:
                Index of the cluster each sample belongs to.
        """
        return self.fit(X=X, y=None, **fit_params).labels_
