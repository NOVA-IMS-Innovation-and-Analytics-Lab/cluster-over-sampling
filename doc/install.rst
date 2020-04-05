.. _install_guide:

============
Installation
============

Prerequisites
-------------

.. currentmodule:: clover.over_sampling

The `cluster-over-sampling` package requires the following dependencies:

* numpy (>=1.11)
* scipy (>=0.17)
* scikit-learn (>=0.21)
* imbalanced-learn (>=0.6.0)

In order to use :class:`SOMO` class the following package
is required:

* som-learn (>=0.1.1)

Additionally, :class:`GeometricSOMO` class has the
following dependency:

* geometric-smote (>=0.1.3)

Install
-------

`cluster-over-sampling` is currently available on the PyPi's repositories
and you can install it via `pip`::

  pip install -U cluster-over-sampling

The package is released also in Anaconda Cloud platform::

  conda install -c algowit cluster-over-sampling

If you prefer, you can clone it and run the setup.py file. Use the following
commands to get a copy from Github and install all dependencies::

  git clone https://github.com/AlgoWit/cluster-over-sampling.git
  cd cluster-over-sampling
  pip install .

Or install using pip and GitHub::

  pip install -U git+https://github.com/AlgoWit/cluster-over-sampling.git
