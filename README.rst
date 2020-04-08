.. -*- mode: rst -*-

.. _scikit-learn: http://scikit-learn.org/stable/

.. _imbalanced-learn: http://imbalanced-learn.org/en/stable/

|Travis|_ |Codecov|_ |CircleCI|_ |ReadTheDocs|_ |PythonVersion|_ |Pypi|_ |Conda|_ |DOI|_ |Black|_

.. |Travis| image:: https://travis-ci.org/AlgoWit/cluster-over-sampling.svg?branch=master
.. _Travis: https://travis-ci.org/AlgoWit/cluster-over-sampling

.. |Codecov| image:: https://codecov.io/gh/AlgoWit/cluster-over-sampling/branch/master/graph/badge.svg
.. _Codecov: https://codecov.io/gh/AlgoWit/cluster-over-sampling

.. |CircleCI| image:: https://circleci.com/gh/AlgoWit/cluster-over-sampling/tree/master.svg?style=svg
.. _CircleCI: https://circleci.com/gh/AlgoWit/cluster-over-sampling/tree/master

.. |ReadTheDocs| image:: https://readthedocs.org/projects/cluster-over-sampling/badge/?version=latest
.. _ReadTheDocs: https://cluster-over-sampling.readthedocs.io/en/latest/?badge=latest

.. |PythonVersion| image:: https://img.shields.io/pypi/pyversions/cluster-over-sampling.svg
.. _PythonVersion: https://img.shields.io/pypi/pyversions/cluster-over-sampling.svg

.. |Pypi| image:: https://badge.fury.io/py/cluster-over-sampling.svg
.. _Pypi: https://badge.fury.io/py/cluster-over-sampling

.. |Conda| image:: https://anaconda.org/algowit/cluster-over-sampling/badges/installer/conda.svg
.. _Conda: https://conda.anaconda.org/algowit

.. |DOI| image:: https://zenodo.org/badge/DOI/10.1016/j.eswa.2017.03.073.svg
.. _DOI: https://doi.org/10.1016/j.eswa.2017.03.073

.. |Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
.. _Black: https://github.com/ambv/black

=====================
cluster-over-sampling
=====================

Implementation of a general interface for clustering based over-sampling
algorithms as described in [1]_, [2]_. It is compatible with scikit-learn_ and
imbalanced-learn_.

Instructions
------------

Installation documentation, API documentation, and examples can be found on the
documentation_.

.. _documentation: https://cluster-over-sampling.readthedocs.io/en/latest/

Dependencies
------------

cluster-over-sampling is tested to work under Python 3.6+. The dependencies
are the following:

- numpy(>=1.1)
- scikit-learn(>=0.21)
- imbalanced-learn(>=0.6.2)

Optional dependencies for SOMO and Geometric SOMO are the following:

- som-learn(>=0.1.1)
- geometric-smote(>=0.1.3)

Additionally, to run the examples, you need matplotlib(>=2.0.0) and
pandas(>=0.22).

Installation
------------

cluster-over-sampling is currently available on the PyPi's repository
and you can install it via `pip`::

  pip install -U cluster-over-sampling

The package is released also in Anaconda Cloud platform::

  conda install -c algowit cluster-over-sampling

If you prefer, you can clone it and run the setup.py file. Use the following
commands to get a copy from GitHub and install all dependencies::

  git clone https://github.com/AlgoWit/cluster-over-sampling.git
  cd cluster-over-sampling
  pip install .

Or install using pip and GitHub::

  pip install -U git+https://github.com/AlgoWit/cluster-over-sampling.git

Testing
-------

After installation, you can use `pytest` to run the test suite::

  make test

About
-----

If you use cluster-over-sampling in a scientific publication, we would
appreciate citations to any of the following papers::

  @article{Douzas2017,
    doi = {10.1016/j.eswa.2017.03.073},
    url = {https://doi.org/10.1016/j.eswa.2017.03.073},
    year = {2017},
    month = oct,
    publisher = {Elsevier {BV}},
    volume = {82},
    pages = {40--52},
    author = {Georgios Douzas and Fernando Bacao},
    title = {Self-Organizing Map Oversampling ({SOMO}) for imbalanced data set learning},
    journal = {Expert Systems with Applications}
  }

  @article{Douzas2018,
    doi = {10.1016/j.ins.2018.06.056},
    url = {https://doi.org/10.1016/j.ins.2018.06.056},
    year = {2018},
    month = oct,
    publisher = {Elsevier {BV}},
    volume = {465},
    pages = {1--20},
    author = {Georgios Douzas and Fernando Bacao and Felix Last},
    title = {Improving imbalanced learning through a heuristic oversampling method based on k-means and {SMOTE}},
    journal = {Information Sciences}
  }

Learning from class-imbalanced data continues to be a common and challenging
problem in supervised learning as standard classification algorithms are
designed to handle balanced class distributions. While different strategies
exist to tackle this problem, methods which generate artificial data to achieve
a balanced class distribution are more versatile than modifications to the
classification algorithm. SMOTE algorithm [3]_, as well as any other
over-sampling method based on the SMOTE mechanism, generates synthetic samples
along line segments that join minority class instances. SMOTE addresses only
the issue of between-classes imbalance. On the other hand, by clustering the
input space and applying any over-sampling algorithm for each resulting cluster
with appropriate resampling ratio, the within-classes imbalanced issue can be
addressed. SOMO [1]_ and KMeans-SMOTE [2]_ are specific realizations of this
approach.

References:
-----------

.. [1] G. Douzas, F. Bacao, "Self-Organizing Map Oversampling (SOMO)
   for imbalanced data set learning", Expert Systems with Applications,
   vol. 82, pp. 40-52, 2017.

.. [2] G. Douzas, F. Bacao, F. Last, "Improving imbalanced learning
   through a heuristic oversampling method based on k-means and SMOTE",
   Information Sciences, vol. 465, pp. 1-20, 2018.

.. [3] N. V. Chawla, K. W. Bowyer, L. O. Hall, W. P. Kegelmeyer, "SMOTE:
   synthetic minority over-sampling technique", Journal of Artificial
   Intelligence Research, vol. 16, pp. 321-357, 2002.
