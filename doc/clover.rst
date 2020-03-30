.. _imbalanced-learn: https://imbalanced-learn.readthedocs.io/en/stable/

.. _scikit-learn: http://scikit-learn.org/stable/

.. _clover:

==============================
Clustering based over-sampling
==============================

A practical guide
-----------------

One way to fight the imbalanced learning problem is to generate new samples in
the classes which are under-represented. Many algorithms have been proposed for
this task, tend to generate unnecessary noise and ignore the within class
imbalance problem. The package cluster-over-sampling extends the functionality
of imbalanced-learn_'s over-samplers by introducing the ``clusterer`` and
``distributor`` parameters::

   >>> from collections import Counter
   >>> from sklearn.datasets import make_classification
   >>> from sklearn.cluster import KMeans
   >>> from clover.over_sampling import SMOTE
   >>> from clover.distribution import DensityDistributor
   >>> X, y = make_classification(n_classes=3, weights=[0.10, 0.10, 0.80], random_state=0, n_informative=10)
   >>> kmeans_smote = SMOTE(clusterer=KMeans(random_state=1), distributor=DensityDistributor())
   >>> X_resampled, y_resampled = kmeans_smote.fit_resample(X, y)
   >>> print(sorted(Counter(y_resampled).items()))
   [(0, 80), (1, 80), (2, 80)]

The augmented data set should be used instead of the original data set
to train a classifier::

   >>> from sklearn.tree import DecisionTreeClassifier
   >>> clf = DecisionTreeClassifier()
   >>> clf.fit(X_resampled, y_resampled) # doctest : +ELLIPSIS
   DecisionTreeClassifier(...)

.. currentmodule:: clover.over_sampling

Parameter ``clusterer``
-----------------------

The parameter ``clusterer`` is used to define the clustering algorithm that is
applied to the input matrix. All of scikit-learn_'s clusterers are supported.
For example, if we select :class:`SMOTE` [CBHK2002]_ as the over-sampler and
`KMeans <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html>`_
as the clustering algorithm than the clustering based over-sampling algorithm
described in [DB2018]_ is created::

   >>> from collections import Counter
   >>> from sklearn.datasets import make_classification
   >>> from sklearn.cluster import KMeans
   >>> from clover.over_sampling import SMOTE
   >>> from clover.distribution import DensityDistributor
   >>> X, y = make_classification(n_classes=3, weights=[0.10, 0.10, 0.80], random_state=0, n_informative=10)
   >>> kmeans_smote = SMOTE(clusterer=KMeans(random_state=2), distributor=DensityDistributor(), random_state=3)
   >>> X_resampled, y_resampled = kmeans_smote.fit_resample(X, y)
   >>> print(sorted(Counter(y_resampled).items()))
   [(0, 80), (1, 80), (2, 80)]

Similarly, any other combination of an over-sampler and clusterer can be
selected::

   >>> from clover.over_sampling import RandomOverSampler
   >>> from sklearn.cluster import AffinityPropagation
   >>> affinity_ros = RandomOverSampler(clusterer=AffinityPropagation(), distributor=DensityDistributor(), random_state=4)
   >>> X_resampled, y_resampled = affinity_ros.fit_resample(X, y)
   >>> print(sorted(Counter(y_resampled).items()))
   [(0, 80), (1, 80), (2, 80)]

Additionally, if the clusterer supports a neighboring structure for the
clusters through a ``neighbors_`` attribute, then it can be used to generate
inter-cluster artificial data as suggested in [DB2017]_.

.. currentmodule:: clover.distribution

Parameter ``distributor``
-------------------------

The parameter ``distributor`` is used to define the distribution of the
generated samples to the clusters. The class :class:`DensityDistributor`
is provided but any other distributor can be defined by extending the
:class:`BaseDistributor` class::

   >>> distributor = DensityDistributor()
   >>> clusterer = KMeans(n_clusters=6, random_state=1).fit(X, y)
   >>> labels = clusterer.labels_
   >>> intra_distribution, inter_distribution = distributor.fit_distribute(X, y, labels, neighbors=None)
   >>> print(distributor.filtered_clusters_)
   [(3, 0), (3, 1)]
   >>> print(distributor.clusters_density_)
   {(3, 0): 6.0, (3, 1): 6.0}
   >>> print(intra_distribution)
   {(3, 0): 1.0, (3, 1): 1.0}
   >>> print(inter_distribution)
   {}

Compatibility
-------------

The API of cluster-over-sampling is fully compatible to imbalanced-learn.
Any over-sampler from cluster-over-sampling that does not use clustering,
i.e. when ``clusterer=None``, is equivalent to the corresponding
imbalanced-learn over-sampler::

   >>> import numpy as np
   >>> from imblearn.over_sampling import SMOTE
   >>> X_res_im, y_res_im = SMOTE(random_state=5).fit_resample(X, y)
   >>> from clover.over_sampling import SMOTE
   >>> X_res_cl, y_res_cl = SMOTE(random_state=5).fit_resample(X, y)
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
