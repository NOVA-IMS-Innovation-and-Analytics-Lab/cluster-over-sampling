#! /usr/bin/env python
"""Clustering-based over-sampling."""

import codecs
import os

from setuptools import find_packages, setup

ver_file = os.path.join('clover', '_version.py')
with open(ver_file) as f:
    exec(f.read())

DISTNAME = 'cluster-over-sampling'
DESCRIPTION = 'Clustering-based over-sampling.'
with codecs.open('README.rst', encoding='utf-8-sig') as f:
    LONG_DESCRIPTION = f.read()
MAINTAINER = 'G. Douzas'
MAINTAINER_EMAIL = 'gdouzas@icloud.com'
URL = 'https://github.com/AlgoWit/cluster-over-sampling'
LICENSE = 'MIT'
DOWNLOAD_URL = 'https://github.com/AlgoWit/cluster-over-sampling'
VERSION = __version__
INSTALL_REQUIRES = ['scipy>=0.17', 'numpy>=1.1', 'scikit-learn>=0.22', 'imbalanced-learn>=0.6.2']
CLASSIFIERS = ['Intended Audience :: Science/Research',
               'Intended Audience :: Developers',
               'License :: OSI Approved',
               'Programming Language :: Python',
               'Topic :: Software Development',
               'Topic :: Scientific/Engineering',
               'Operating System :: Microsoft :: Windows',
               'Operating System :: POSIX',
               'Operating System :: Unix',
               'Operating System :: MacOS',
               'Programming Language :: Python :: 3.6',
               'Programming Language :: Python :: 3.7',
               'Programming Language :: Python :: 3.8']
EXTRAS_REQUIRE = {
    'tests': [
        'pytest',
        'pytest-cov',
    ],
    'docs': [
        'sphinx==1.8.5',
        'sphinx-gallery',
        'sphinx_rtd_theme',
        'numpydoc',
        'matplotlib',
        'pandas'
    ],
    'optional': [
        'som-learn>=0.1.1',
        'geometric-smote>=0.1.3'
    ]
}

setup(
    name=DISTNAME,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    description=DESCRIPTION,
    license=LICENSE,
    url=URL,
    version=VERSION,
    download_url=DOWNLOAD_URL,
    long_description=LONG_DESCRIPTION,
    zip_safe=False,
    classifiers=CLASSIFIERS,
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE
)