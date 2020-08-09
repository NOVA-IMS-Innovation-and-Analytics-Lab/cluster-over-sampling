"""Clustering-based over-sampling.

[SOMO oversampling algorithm]: <https://www.sciencedirect.com/science/article/abs/pii/S0957417417302324>
[KMeans-SMOTE oversampling algorithm]: <https://www.sciencedirect.com/science/article/abs/pii/S0020025518304997>
[G-SOMO oversampling algorithm]: <https://www.sciencedirect.com/science/article/abs/pii/S095741742100662X>

The module provides the implementation of an interface for clustering-based over-sampling. It
has two submodules:

- [`distribution`][clover.distribution]: Provides the classes to distrubute the generated samples into clusters.

    - [`DensityDistributor`][clover.distribution.DensityDistributor]: Density based distributor.

- [`over_sampling`][clover.over_sampling]: Provides the clustering-based oversampling algorithms.

    - [`ClusterOverSampler`][clover.over_sampling.ClusterOverSampler]: Combinations of oversampler and clusterer.
    - [`KMeansSMOTE`][clover.over_sampling.KMeansSMOTE]: [KMeans-SMOTE oversampling algorithm] oversampling algorithm.
    - [`SOMO`][clover.over_sampling.SOMO]: [SOMO oversampling algorithm].
    - [`GeometricSOMO`][clover.over_sampling.GeometricSOMO]: [G-SOMO oversampling algorithm].
"""


from nptyping import Float, Int, NDArray, Shape

InputData = NDArray[Shape['*, *'], Float]
Targets = NDArray[Shape['*'], Float]
Labels = NDArray[Shape["*"], Int]
Neighbors = NDArray[Shape["*, 2"], Int]
MultiLabel = tuple[int, int]
IntraDistribution = dict[MultiLabel, float]
InterDistribution = dict[tuple[MultiLabel, MultiLabel], float]
Density = dict[MultiLabel, float]
