#!/bin/bash

mkdir build_conda
conda config --set anaconda_upload yes
conda build --output-folder ./build_conda --user AlgoWit cluster-over-sampling
rm -r build_conda
