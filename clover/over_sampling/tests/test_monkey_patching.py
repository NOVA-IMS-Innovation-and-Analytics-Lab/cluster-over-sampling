import inspect

import pytest
import numpy as np
from imblearn.over_sampling.tests import (
    test_random_over_sampler,
    test_smote,
    test_borderline_smote,
    test_svm_smote,
    test_adasyn,
)

from clover.over_sampling.monkey_patching import (
    RandomOverSampler,
    SMOTE,
    BorderlineSMOTE,
    SVMSMOTE,
    ADASYN,
)


def data():
    X = np.array(
        [
            [0.11622591, -0.0317206],
            [0.77481731, 0.60935141],
            [1.25192108, -0.22367336],
            [0.53366841, -0.30312976],
            [1.52091956, -0.49283504],
            [-0.28162401, -2.10400981],
            [0.83680821, 1.72827342],
            [0.3084254, 0.33299982],
            [0.70472253, -0.73309052],
            [0.28893132, -0.38761769],
            [1.15514042, 0.0129463],
            [0.88407872, 0.35454207],
            [1.31301027, -0.92648734],
            [-1.11515198, -0.93689695],
            [-0.18410027, -0.45194484],
            [0.9281014, 0.53085498],
            [-0.14374509, 0.27370049],
            [-0.41635887, -0.38299653],
            [0.08711622, 0.93259929],
            [1.70580611, -0.11219234],
        ]
    )
    y = np.array([0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0])
    return X, y


def run_imblearn_tests(mod, cls):
    """Run all imblearn's test for a module."""
    setattr(mod, cls.__name__, cls)
    for name, test in inspect.getmembers(mod):
        if inspect.isfunction(test) and name.startswith('test_'):
            try:
                parametrized_marks = [
                    mark for mark in test.pytestmark if mark.name == 'parametrize'
                ]
                args = parametrized_marks[0].args[1]
                for arg in args:
                    if name.startswith('test_borderline'):
                        test(arg, data())
                    else:
                        test(*arg) if isinstance(arg, tuple) else test(arg)
            except (AttributeError, IndexError):
                if name.startswith('test_borderline') or name.startswith('test_svm'):
                    test(data())
                else:
                    test()


@pytest.mark.filterwarnings("ignore:'return_indices' is deprecated from 0.4")
def test_random_over_sampler_module():
    """Test RandomOverSampler class."""
    run_imblearn_tests(test_random_over_sampler, RandomOverSampler)


@pytest.mark.filterwarnings('ignore:"kind" is deprecated in 0.4 and will be')
@pytest.mark.filterwarnings('ignore:"svm_estimator" is deprecated in 0.4 and')
@pytest.mark.filterwarnings('ignore:"out_step" is deprecated in 0.4 and')
@pytest.mark.filterwarnings('ignore:"m_neighbors" is deprecated in 0.4 and')
def test_smote_module():
    """Test SMOTE class."""
    run_imblearn_tests(test_smote, SMOTE)


def test_borderline_smote_module():
    """Test BorderlineSMOTE class."""
    run_imblearn_tests(test_borderline_smote, BorderlineSMOTE)


def test_svm_smote_module():
    """Test SVMSMOTE class."""
    run_imblearn_tests(test_svm_smote, SVMSMOTE)


def test_adasyn_module():
    """Test ADASYN class."""
    run_imblearn_tests(test_adasyn, ADASYN)
