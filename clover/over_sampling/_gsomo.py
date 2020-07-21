"""
Includes the implementation of SOMO.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from math import sqrt

from sklearn.base import clone
from sklearn.utils import check_random_state
from sklearn.utils import check_scalar
from imblearn.over_sampling.base import BaseOverSampler
from imblearn.utils import Substitution, check_sampling_strategy
from imblearn.utils._docstring import _random_state_docstring, _n_jobs_docstring

from ._cluster import ClusterOverSampler
from ..distribution import DensityDistributor


@Substitution(
    sampling_strategy=BaseOverSampler._sampling_strategy_docstring,
    n_jobs=_n_jobs_docstring,
    random_state=_random_state_docstring,
)
class GeometricSOMO(ClusterOverSampler):
    """Applies the SOM algorithm to the input space before applying Geometric SMOTE.

    Read more in the :ref:`user guide <user_guide>`.

    Parameters
    ----------
    {sampling_strategy}

    {random_state}

    k_neighbors : int or object, default=5
        Defines the number of nearest neighbors to be used by Geometric SMOTE.

        - If ``int``, this number is used to construct synthetic
          samples.

        - If ``object``, an estimator that inherits from
          :class:`sklearn.neighbors.base.KNeighborsMixin` that will be
          used to find the number of nearest neighbors.

    truncation_factor : float, default=1.0
        The type of truncation. The values should be in the [-1.0, 1.0] range.

    deformation_factor : float, default=0.0
        The type of geometry. The values should be in the [0.0, 1.0] range.

    selection_strategy : str, default='combined'
        The type of Geometric SMOTE algorithm with the following options:
        ``'combined'``, ``'majority'``, ``'minority'``.

    som_estimator : None or object or int or float, default=None
        Defines the SOM clusterer applied to the input space.

        - If ``None``, :class:`` is used which
          tends to be better with large number of samples.

        - If KMeans object, then an instance from either
          :class:`sklearn.cluster.KMeans` or :class:`sklearn.cluster.MiniBatchKMeans`.

        - If ``int``, the number of clusters to be used.

        - If ``float``, the proportion of the number of clusters over the number
          of samples to be used.

    imbalance_ratio_threshold : 'auto' or float, default='auto'
        The threshold of a filtered cluster. It can be any non-negative number or
        ``'auto'`` to be calculated automatically.

        - If ``'auto'``, the filtering threshold is calculated from the imbalance
          ratio of the target for the binary case or the maximum of the target's
          imbalance ratios for the multiclass case.

        - If ``float`` then it is manually set to this number.

        Any cluster that has an imbalance ratio smaller than the filtering threshold is
        identified as a filtered cluster and can be potentially used to generate
        minority class instances. Higher values increase the number of filtered
        clusters.

    distances_exponent : 'auto' or float, default='auto'
        The exponent of the mean distance in the density calculation. It can be
        any non-negative number or ``'auto'`` to be calculated automatically.

        - If ``'auto'`` then it is set equal to the number of
          features. Higher values make the calculation of density more sensitive
          to the cluster's size i.e. clusters with large mean euclidean distance
          between samples are penalized.

        - If ``float`` then it is manually set to this number.

    distribution_ratio : float, default=0.8
        The ratio of intra-cluster to inter-cluster generated samples. It is a
        number in the :math:`[0.0, 1.0]` range. The default value is ``0.8``, a
        number equal to the proportion of intra-cluster generated samples over
        the total number of generated samples. As the number decreases, less
        intra-cluster and more inter-cluster samples are generated.

    raise_error : bool, default=True
        Raise an error when no samples are generated.

        - If ``True``, it raises an error when no filtered clusters are
          identified and therefore no samples are generated.

        - If ``False``, it displays a warning.

    {n_jobs}


    Attributes
    ----------
    clusterer_ : object
        A fitted :class:`somlearn.SOM` instance.

    distributor_ : object
        A fitted :class:`clover.distribution.DensityDistributor` instance.

    labels_ : array, shape (n_samples,)
        Labels of each sample.

    neighbors_ : array, (n_neighboring_pairs, 2) or None
        An array that contains all neighboring pairs with each row being
        a unique neighboring pair.

    oversampler_ : object
        A fitted :class:`imblearn.over_sampling.SMOTE` instance.

    random_state_ : object
        An instance of ``RandomState`` class.

    sampling_strategy_ : dict
        Actual sampling strategy.

    Examples
    --------
    >>> import numpy as np
    >>> from clover.over_sampling import GeometricSOMO # doctest: +SKIP
    >>> from sklearn.datasets import make_blobs
    >>> blobs = [100, 800, 100]
    >>> X, y  = make_blobs(blobs, centers=[(-10, 0), (0,0), (10, 0)])
    >>> # Add a single 0 sample in the middle blob
    >>> X = np.concatenate([X, [[0, 0]]])
    >>> y = np.append(y, 0)
    >>> # Make this a binary classification problem
    >>> y = y == 1
    >>> somo = GeometricSOMO(random_state=42) # doctest: +SKIP
    >>> X_res, y_res = somo.fit_resample(X, y) # doctest: +SKIP
    >>> # Find the number of new samples in the middle blob
    >>> right, left = X_res[:, 0] > -5, X_res[:, 0] < 5 # doctest: +SKIP
    >>> n_res_in_middle = (right & left).sum() # doctest: +SKIP
    >>> print("Samples in the middle blob: %s" % n_res_in_middle) # doctest: +SKIP
    Samples in the middle blob: 801
    >>> unchanged = n_res_in_middle == blobs[1] + 1 # doctest: +SKIP
    >>> print("Middle blob unchanged: %s" % unchanged) # doctest: +SKIP
    Middle blob unchanged: True
    >>> more_zero_samples = (y_res == 0).sum() > (y == 0).sum() # doctest: +SKIP
    >>> print("More 0 samples: %s" % more_zero_samples) # doctest: +SKIP
    More 0 samples: True
    """

    def __init__(
        self,
        sampling_strategy='auto',
        random_state=None,
        k_neighbors=5,
        truncation_factor=1.0,
        deformation_factor=0.0,
        selection_strategy='combined',
        som_estimator=None,
        imbalance_ratio_threshold='auto',
        distances_exponent='auto',
        distribution_ratio=0.8,
        raise_error=True,
        n_jobs=None,
    ):
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.k_neighbors = k_neighbors
        self.truncation_factor = truncation_factor
        self.deformation_factor = deformation_factor
        self.selection_strategy = selection_strategy
        self.som_estimator = som_estimator
        self.distribution_ratio = distribution_ratio
        self.imbalance_ratio_threshold = imbalance_ratio_threshold
        self.distances_exponent = distances_exponent
        self.raise_error = raise_error
        self.n_jobs = n_jobs

    def _initialize_fitting(self, X, y):
        """Initialize fitting process."""

        # Import SOM and GeometricSMOTE
        try:
            from somlearn import SOM
        except ImportError:
            raise ImportError(
                'GeometricSOMO class requires the package `som-learn` to be installed.'
            )
        try:
            from gsmote import GeometricSMOTE
        except ImportError:
            raise ImportError(
                'GeometricSOMO class requires the package `geometric-smote` to '
                'be installed.'
            )

        # Check random state
        self.random_state_ = check_random_state(self.random_state)

        # Check oversampler
        self.oversampler_ = GeometricSMOTE(
            sampling_strategy=self.sampling_strategy,
            k_neighbors=self.k_neighbors,
            truncation_factor=self.truncation_factor,
            deformation_factor=self.deformation_factor,
            selection_strategy=self.selection_strategy,
            random_state=self.random_state_,
            n_jobs=self.n_jobs,
        )
        self.sampling_strategy_ = check_sampling_strategy(
            self.oversampler_.sampling_strategy, y, self._sampling_type,
        )

        if self.som_estimator is None:
            self.clusterer_ = SOM(random_state=self.random_state_)
        elif isinstance(self.som_estimator, int):
            check_scalar(self.som_estimator, 'som_estimator', int, min_val=1)
            n = round(sqrt(self.som_estimator))
            self.clusterer_ = SOM(
                n_columns=n, n_rows=n, random_state=self.random_state_
            )
        elif isinstance(self.som_estimator, float):
            check_scalar(
                self.som_estimator, 'som_estimator', float, min_val=0.0, max_val=1.0
            )
            n = round(sqrt((X.shape[0] - 1) * self.som_estimator + 1))
            self.clusterer_ = SOM(
                n_columns=n, n_rows=n, random_state=self.random_state_
            )
        elif isinstance(self.som_estimator, SOM):
            self.clusterer_ = clone(self.som_estimator)
        else:
            raise TypeError(
                'Parameter `som_estimator` should be '
                'either `None` or the number of clusters '
                'or a float in the [0.0, 1.0] range equal to'
                ' the number of clusters over the number of '
                'samples or an instance of the `SOM` class.'
            )

        # Check distributor
        self.distributor_ = DensityDistributor(
            filtering_threshold=self.imbalance_ratio_threshold,
            distances_exponent=self.distances_exponent,
            distribution_ratio=self.distribution_ratio,
        )

        return self
