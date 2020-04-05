.. _imbalanced-learn: https://imbalanced-learn.readthedocs.io/en/stable/

============
Introduction
============

API
---

All the classes included in `cluster-over-sampling` follow the 
imbalanced-learn_ API using the base over-sampler functionality. 
More specifically:

They implement a ``fit`` method to learn from data::

      oversampler = object.fit(data, targets)

They implement a ``fit_resample`` method to resample data sets::

      data_resampled, targets_resampled = object.fit_resample(data, targets)

The following inputs are used:

* ``data``: array-like (2-D list, pandas.DataFrame, numpy.array) or sparse
  matrices.
* ``targets``: array-like (1-D list, pandas.Series, numpy.array).

Imbalanced learning problem
---------------------------

Classification of imbalanced datasets is a challenging task for standard
algorithms. Although many methods exist to address this problem in different
ways, generating artificial data for the minority class is a more general
approach compared to algorithmic modifications. For a visual representation,
the reader is referred to imbalanced-learn_.

Clustering the input space
--------------------------

SMOTE algorithm, as well as any other over-sampling method based on the SMOTE
mechanism, generates synthetic samples along line segments that join minority
class instances. This approach addresses only the issue of between-classes
imbalance. On the other hand, by clustering the input space and applying any
over-sampling algorithm for each resulting cluster with appropriate resampling
ratio, the within-classes imbalanced issue can be addressed. The
clustering-over-sampling package extends imbalanced-learn_'s functionality by
supporting the clustering of the input space and the distribution of generated
samples on the resulting clusters.
