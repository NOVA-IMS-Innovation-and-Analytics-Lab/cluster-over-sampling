"""
Includes the implementation of KMeans-SMOTE.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from sklearn.base import clone
from sklearn.utils import check_random_state
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.utils import check_scalar
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling.base import BaseOverSampler
from imblearn.utils import Substitution, check_sampling_strategy
from imblearn.utils._docstring import _random_state_docstring, _n_jobs_docstring

from ._cluster import ClusterOverSampler
from ..distribution._density import DensityDistributor


@Substitution(
    sampling_strategy=BaseOverSampler._sampling_strategy_docstring,
    n_jobs=_n_jobs_docstring,
    random_state=_random_state_docstring,
)
class KMeansSMOTE(ClusterOverSampler):
    """Applies KMeans clustering to the input space before applying SMOTE.

    This is an implementation of the algorithm described in [1]_.

    Read more in the :ref:`user guide <user_guide>`.

    Parameters
    ----------
    {sampling_strategy}

    {random_state}

    k_neighbors : int or object, default=5
        Defines the number of nearest neighbors to be used by SMOTE.

        - If ``int``, this number is used to construct synthetic
          samples.

        - If ``object``, an estimator that inherits from
          :class:`sklearn.neighbors.base.KNeighborsMixin` that will be
          used to find the number of nearest neighbors.

    kmeans_estimator : None or object or int or float, default=None
        Defines the KMeans clusterer applied to the input space.

        - If ``None``, :class:`sklearn.cluster.MiniBatchKMeans` is used which
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

    raise_error : boolean, default=True

    {n_jobs}

    Attributes
    ----------
    clusterer_ : object
        A fitted :class:`sklearn.cluster.KMeans` or
        :class:`sklearn.cluster.MiniBatchKMeans` instance.

    distributor_ : object
        A fitted :class:`clover.distribution.DensityDistributor` instance.

    labels_ : array, shape (n_samples,)
        Cluster labels of each sample.

    oversampler_ : object
        A fitted :class:`imblearn.over_sampling.SMOTE` instance.

    random_state_ : object
        An instance of ``RandomState`` class.

    sampling_strategy_ : dict
        Actual sampling strategy.

    References
    ----------
    .. [1] Georgios Douzas, Fernando Bacao, Felix Last, "Improving
       imbalanced learning through a heuristic oversampling method
       based on k-means and SMOTE"
       https://www.sciencedirect.com/science/article/pii/S0020025518304997

    Examples
    --------
    >>> import numpy as np
    >>> from clover.over_sampling import KMeansSMOTE
    >>> from sklearn.datasets import make_blobs
    >>> blobs = [100, 800, 100]
    >>> X, y  = make_blobs(blobs, centers=[(-10, 0), (0,0), (10, 0)])
    >>> # Add a single 0 sample in the middle blob
    >>> X = np.concatenate([X, [[0, 0]]])
    >>> y = np.append(y, 0)
    >>> # Make this a binary classification problem
    >>> y = y == 1
    >>> kmeans_smote = KMeansSMOTE(random_state=42)
    >>> X_res, y_res = kmeans_smote.fit_resample(X, y)
    >>> # Find the number of new samples in the middle blob
    >>> n_res_in_middle = ((X_res[:, 0] > -5) & (X_res[:, 0] < 5)).sum()
    >>> print("Samples in the middle blob: %s" % n_res_in_middle)
    Samples in the middle blob: 801
    >>> print("Middle blob unchanged: %s" % (n_res_in_middle == blobs[1] + 1))
    Middle blob unchanged: True
    >>> print("More 0 samples: %s" % ((y_res == 0).sum() > (y == 0).sum()))
    More 0 samples: True
    """

    def __init__(
        self,
        sampling_strategy="auto",
        random_state=None,
        k_neighbors=5,
        kmeans_estimator=None,
        imbalance_ratio_threshold='auto',
        distances_exponent='auto',
        raise_error=True,
        n_jobs=None,
    ):
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.k_neighbors = k_neighbors
        self.kmeans_estimator = kmeans_estimator
        self.imbalance_ratio_threshold = imbalance_ratio_threshold
        self.distances_exponent = distances_exponent
        self.raise_error = raise_error
        self.n_jobs = n_jobs

    def _initialize_fitting(self, X, y):
        """Initialize fitting process."""

        # Check random state
        self.random_state_ = check_random_state(self.random_state)

        # Check oversampler
        self.oversampler_ = SMOTE(
            sampling_strategy=self.sampling_strategy,
            k_neighbors=self.k_neighbors,
            random_state=self.random_state_,
            n_jobs=self.n_jobs,
        )
        self.sampling_strategy_ = check_sampling_strategy(
            self.oversampler_.sampling_strategy, y, self._sampling_type,
        )

        # Check clusterer
        if self.kmeans_estimator is None:
            self.clusterer_ = MiniBatchKMeans(random_state=self.random_state_)
        elif isinstance(self.kmeans_estimator, int):
            check_scalar(self.kmeans_estimator, 'k_means_estimator', int, min_val=1)
            self.clusterer_ = MiniBatchKMeans(
                n_clusters=self.kmeans_estimator, random_state=self.random_state_
            )
        elif isinstance(self.kmeans_estimator, float):
            check_scalar(
                self.kmeans_estimator,
                'k_means_estimator',
                float,
                min_val=0.0,
                max_val=1.0,
            )
            n_clusters = round((X.shape[0] - 1) * self.kmeans_estimator + 1)
            self.clusterer_ = MiniBatchKMeans(
                n_clusters=n_clusters, random_state=self.random_state
            )
        elif isinstance(self.kmeans_estimator, KMeans) or isinstance(
            self.kmeans_estimator, MiniBatchKMeans
        ):
            self.clusterer_ = clone(self.kmeans_estimator)
        else:
            raise TypeError(
                'Parameter `kmeans_estimator` should be '
                'either `None` or the number of clusters '
                'or a float in the [0.0, 1.0] range equal to'
                ' the number of clusters over the number of '
                'samples or an instance of either `KMeans` '
                'or `MiniBatchKMeans` class.'
            )

        # Check distributor
        self.distributor_ = DensityDistributor(
            filtering_threshold=self.imbalance_ratio_threshold,
            distances_exponent=self.distances_exponent,
        )

        return self
