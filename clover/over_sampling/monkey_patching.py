"""
Reimplements standard oversamplers and monkey patch
their methods to make them compatible with the clustering
based oversampling API.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: BSD 3 clause

from re import sub

from imblearn.utils._docstring import Substitution
from imblearn.over_sampling import (
    RandomOverSampler as _RandomOverSampler,
    SMOTE as _SMOTE,
    BorderlineSMOTE as _BorderlineSMOTE,
    SVMSMOTE as _SVMSMOTE,
    ADASYN as _ADASYN,
)

from .base import BaseClusterOverSampler


def _generate_docstring(oversampler, init_docstring):
    """"Generate docstring for an oversampler."""

    # Extract and modify docstring
    intro, doc = init_docstring.split('Parameters')
    intro = sub(r'Read.+\.', '', intro)
    doc = (
        doc.replace('imblearn', 'clover')
        .replace('0.4', '0.4 (imbalanced-learn warning)')
        .replace('----------', '', 1)
    )

    # Create new docstring
    oversampler.__doc__ = """{intro}

    Parameters
    ----------

    {clusterer}

    {distributor}

    {doc}
    """
    oversampler = Substitution(
        intro=intro,
        clusterer=BaseClusterOverSampler._clusterer_docstring,
        distributor=BaseClusterOverSampler._distributor_docstring,
        doc=doc,
    )(oversampler)

    return oversampler


def monkey_patch_attributes(init_oversampler):
    """Parametrized decorator to monkey patch attributes for oversamplers."""

    SMOTE_ATTRIBUTES = (
        '_validate_estimator',
        '_make_samples',
        '_generate_sample',
        '_in_danger_noise',
        '_sample',
    )
    ATTRIBUTES_MAPPING = {
        _RandomOverSampler: (),
        _SMOTE: SMOTE_ATTRIBUTES,
        _BorderlineSMOTE: SMOTE_ATTRIBUTES,
        _SVMSMOTE: SMOTE_ATTRIBUTES,
        _ADASYN: ('_validate_estimator',),
    }

    def _monkey_patch_attributes(oversampler):
        """Decorator function."""

        # Method to generate new samples
        def _fit_resample_cluster(self, X, y):
            X_res, y_res, *indices_res = init_oversampler._fit_resample(self, X, y)
            X_new, y_new = X_res[len(X) :], y_res[len(X) :]
            if indices_res and self.return_indices:
                indices_new = indices_res[0][len(X) :]
                return X_new, y_new, indices_new
            return X_new, y_new

        # Monkey patching
        oversampler._fit_resample_cluster = _fit_resample_cluster
        for attribute in ATTRIBUTES_MAPPING[init_oversampler]:
            setattr(oversampler, attribute, getattr(init_oversampler, attribute))

        # Generate docstring
        oversampler = _generate_docstring(oversampler, init_oversampler.__doc__)

        return oversampler

    return _monkey_patch_attributes


@monkey_patch_attributes(_RandomOverSampler)
class RandomOverSampler(BaseClusterOverSampler, _RandomOverSampler):
    def __init__(
        self,
        clusterer=None,
        distributor=None,
        sampling_strategy='auto',
        return_indices=False,
        random_state=None,
        n_jobs=1,
        ratio=None,
    ):
        super(RandomOverSampler, self).__init__(
            clusterer=clusterer,
            distributor=distributor,
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            n_jobs=n_jobs,
            ratio=ratio,
        )
        self.return_indices = return_indices

    def _fit_resample_cluster(self, X, y):
        pass


@monkey_patch_attributes(_SMOTE)
class SMOTE(BaseClusterOverSampler, _SMOTE):
    def __init__(
        self,
        clusterer=None,
        distributor=None,
        sampling_strategy='auto',
        random_state=None,
        k_neighbors=5,
        m_neighbors='deprecated',
        out_step='deprecated',
        kind='deprecated',
        svm_estimator='deprecated',
        n_jobs=1,
        ratio=None,
    ):
        super(SMOTE, self).__init__(
            clusterer=clusterer,
            distributor=distributor,
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            n_jobs=n_jobs,
            ratio=ratio,
        )
        self.k_neighbors = k_neighbors
        self.kind = kind
        self.m_neighbors = m_neighbors
        self.out_step = out_step
        self.svm_estimator = svm_estimator

    def _fit_resample_cluster(self, X, y):
        pass


@monkey_patch_attributes(_BorderlineSMOTE)
class BorderlineSMOTE(BaseClusterOverSampler, _BorderlineSMOTE):
    def __init__(
        self,
        clusterer=None,
        distributor=None,
        sampling_strategy='auto',
        random_state=None,
        k_neighbors=5,
        m_neighbors=10,
        kind='borderline-1',
        n_jobs=1,
    ):
        super(BorderlineSMOTE, self).__init__(
            clusterer=clusterer,
            distributor=distributor,
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        self.k_neighbors = k_neighbors
        self.m_neighbors = m_neighbors
        self.kind = kind

    def _fit_resample_cluster(self, X, y):
        pass


@monkey_patch_attributes(_SVMSMOTE)
class SVMSMOTE(BaseClusterOverSampler, _SVMSMOTE):
    def __init__(
        self,
        clusterer=None,
        distributor=None,
        sampling_strategy='auto',
        random_state=None,
        k_neighbors=5,
        m_neighbors=10,
        svm_estimator=None,
        out_step=0.5,
        n_jobs=1,
    ):
        super(SVMSMOTE, self).__init__(
            clusterer=clusterer,
            distributor=distributor,
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        self.k_neighbors = k_neighbors
        self.m_neighbors = m_neighbors
        self.svm_estimator = svm_estimator
        self.out_step = out_step

    def _fit_resample_cluster(self, X, y):
        pass


@monkey_patch_attributes(_ADASYN)
class ADASYN(BaseClusterOverSampler, _ADASYN):
    def __init__(
        self,
        clusterer=None,
        distributor=None,
        sampling_strategy='auto',
        random_state=None,
        n_neighbors=5,
        n_jobs=1,
        ratio=None,
    ):
        super(ADASYN, self).__init__(
            clusterer=clusterer,
            distributor=distributor,
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            n_jobs=n_jobs,
            ratio=ratio,
        )
        self.n_neighbors = n_neighbors

    def _fit_resample_cluster(self, X, y):
        pass
