#!/bin/bash
# Install required Python packages
# Students should add any additional packages they need here
conda create -n nlp_ass1 python=3.10
conda activate nlp_ass1
conda install -c conda-forge numpy scipy pandas -y
pip3 install ortools matplotlib tqdm k-means-constrained torch