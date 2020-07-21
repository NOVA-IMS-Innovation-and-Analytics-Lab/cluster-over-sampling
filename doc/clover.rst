.. _imbalanced-learn: https://imbalanced-learn.readthedocs.io/en/stable/

.. _scikit-learn: http://scikit-learn.org/stable/

.. _clover:

==============================
Clustering-based over-sampling
==============================

A practical guide
-----------------

.. currentmodule:: clover.over_sampling

One way to fight the imbalanced learning problem is to generate new samples in
the classes which are under-represented. Many algorithms have been proposed for
this task, tend to generate unnecessary noise and ignore the within class
imbalance problem. The package `cluster-over-sampling` extends the functionality
of imbalanced-learn_'s over-samplers by introducing the 
:class:`ClusterOverSampler` class. An instance of this
class is created by defining the ``oversampler`` parameter as well as the 
``clusterer`` and ``distributor`` parameters::

   >>> from collections import Counter
   >>> from sklearn.datasets import make_classification
   >>> from sklearn.cluster import KMeans
   >>> from imblearn.over_sampling import SMOTE
   >>> from clover.over_sampling import ClusterOverSampler
   >>> from clover.distribution import DensityDistributor
   >>> X, y = make_classification(n_classes=3, weights=[0.10, 0.10, 0.80], random_state=0, n_informative=10)
   >>> print(sorted(Counter(y).items()))
   [(0, 10), (1, 10), (2, 80)]
   >>> clovrs = ClusterOverSampler(oversampler=SMOTE(random_state=5), clusterer=KMeans(random_state=9), distributor=DensityDistributor())
   >>> X_resampled, y_resampled = clovrs.fit_resample(X, y)
   >>> print(sorted(Counter(y_resampled).items()))
   [(0, 80), (1, 80), (2, 80)]

The augmented data set should be used instead of the original data set
to train a classifier::

   >>> from sklearn.tree import DecisionTreeClassifier
   >>> clf = DecisionTreeClassifier()
   >>> clf.fit(X_resampled, y_resampled) # doctest : +ELLIPSIS
   DecisionTreeClassifier(...)

For convenience reasons, `cluster-over-sampling` provides classes that implement the
algorithms SOMO and KMeans-SMOTE as described in [DB2017]_ and [DB2018]_,
respectively::

   >>> from clover.over_sampling import KMeansSMOTE
   >>> kmeans_smote = KMeansSMOTE(random_state=15)
   >>> X_resampled, y_resampled = kmeans_smote.fit_resample(X, y)
   >>> print(sorted(Counter(y_resampled).items()))
   [(0, 80), (1, 80), (2, 80)]

Combining over-sampling and clustering algorithms
-------------------------------------------------

The :class:`ClusterOverSampler` class allows to combine
imbalanced-learn_'s oversamplers with scikit-learn_'s clusterers. This achieved
through the use of the parameters ``oversampler`` and ``clusterer``. For
example, if we select 
`SMOTE
<https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.SMOTE.html>`_
[CBHK2002]_ as the over-sampler and 
`KMeans <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html>`_
as the clustering algorithm than the clustering based over-sampling algorithm 
KMeans-SMOTE as described in [DB2018]_ is created::

   >>> from collections import Counter
   >>> from sklearn.datasets import make_classification
   >>> from sklearn.cluster import KMeans
   >>> from imblearn.over_sampling import SMOTE
   >>> from clover.over_sampling import ClusterOverSampler
   >>> from clover.distribution import DensityDistributor
   >>> X, y = make_classification(n_classes=3, weights=[0.10, 0.10, 0.80], random_state=0, n_informative=10)
   >>> clovrs = ClusterOverSampler(oversampler=SMOTE(random_state=1), clusterer=KMeans(random_state=2), distributor=DensityDistributor(), random_state=3)
   >>> X_resampled, y_resampled = clovrs.fit_resample(X, y)
   >>> print(sorted(Counter(y_resampled).items()))
   [(0, 80), (1, 80), (2, 80)]

Similarly, any other combination of an over-sampler and clusterer can be
selected::

   >>> from imblearn.over_sampling import RandomOverSampler
   >>> from sklearn.cluster import AffinityPropagation
   >>> from clover.over_sampling import ClusterOverSampler
   >>> clovrs = ClusterOverSampler(oversampler=RandomOverSampler(random_state=13), clusterer=AffinityPropagation(random_state=0), distributor=DensityDistributor(), random_state=4)
   >>> X_resampled, y_resampled = clovrs.fit_resample(X, y)
   >>> print(sorted(Counter(y_resampled).items()))
   [(0, 80), (1, 80), (2, 80)]

Additionally, if the clusterer supports a neighboring structure for the
clusters through a ``neighbors_`` attribute, then it can be used to generate
inter-cluster artificial data as suggested in [DB2017]_.

.. currentmodule:: clover.distribution

Additionally, the parameter ``distributor`` is used to define the distribution of the
generated samples to the clusters. The :class:`DensityDistributor` class
implements a density based distribution::

   >>> distributor = DensityDistributor(distances_exponent=0)
   >>> clusterer = KMeans(n_clusters=5, random_state=1).fit(X, y)
   >>> labels = clusterer.labels_
   >>> intra_distribution, inter_distribution = distributor.fit_distribute(X, y, labels, neighbors=None)
   >>> print(distributor.filtered_clusters_)
   [(2, 1), (1, 0), (1, 1), (0, 0)]
   >>> print(distributor.clusters_density_)
   {(2, 1): 3.0, (1, 0): 6.0, (1, 1): 7.0, (0, 0): 2.0}
   >>> print(intra_distribution)
   {(2, 1): 0.7, (1, 0): 0.25, (1, 1): 0.3, (0, 0): 0.75}
   >>> print(inter_distribution)
   {}

Also any other distributor can be defined by extending the 
:class:`BaseDistributor` class.

Basic algorithms
----------------

The basic clustering-based over-samplers SOMO [DB2017]_, KMeans-SMOTE [DB2018]_
and Geometric SOMO are implemented. The corresponding classes :class:`SOMO`,
:class:`KMeansSMOTE` and :class:`GeometricSOMO` include all the appropriate
parameters::

   >>> from clover.over_sampling import KMeansSMOTE
   >>> kmeans_smote = KMeansSMOTE(kmeans_estimator=10, imbalance_ratio_threshold=0.9, random_state=15)
   >>> X_resampled, y_resampled = kmeans_smote.fit_resample(X, y)
   >>> print(sorted(Counter(y_resampled).items()))
   [(0, 80), (1, 80), (2, 80)]

Compatibility
-------------

The API of `cluster-over-sampling` is fully compatible to imbalanced-learn_.
Any over-sampler from cluster-over-sampling that does not use clustering,
i.e. when ``clusterer=None``, is equivalent to the corresponding
imbalanced-learn_'s over-sampler::

   >>> import numpy as np
   >>> from imblearn.over_sampling import SMOTE
   >>> X_res_im, y_res_im = SMOTE(random_state=5).fit_resample(X, y)
   >>> from clover.over_sampling import ClusterOverSampler
   >>> X_res_cl, y_res_cl = ClusterOverSampler(SMOTE(random_state=5), clusterer=None).fit_resample(X, y)
   >>> np.testing.assert_equal(X_res_im, X_res_cl)
   >>> np.testing.assert_equal(y_res_im, y_res_cl)

.. topic:: References

   .. [DB2017] Douzas, G., & Bacao, F. (2019). "Self-Organizing Map
      Oversampling for imbalanced data set learning",
      Expert Systems with Applications, 82, 40-52.
      https://doi.org/10.1016/j.eswa.2017.03.073

   .. [DB2018] Douzas, G., & Bacao, F. (2019). "Improving
      imbalanced learning through a heuristic oversampling
      method based on k-means and SMOTE",
      Information Sciences, 465, 1-20.
      https://doi.org/10.1016/j.ins.2018.06.056

   .. [CBHK2002] N. V. Chawla, K. W. Bowyer, L. O. Hall, W. P. Kegelmeyer, "SMOTE:
      synthetic minority over-sampling technique", Journal of Artificial
      Intelligence Research, vol. 16, pp. 321-357, 2002.
