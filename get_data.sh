#!/bin/bash

data_dir=data
mkdir $data_dir

echo "Scikit-learn..."
wget -O sklearn.zip https://github.com/scikit-learn/scikit-learn/archive/master.zip
unzip -q -d $data_dir/sklearn sklearn.zip

echo "Tensorflow..."
wget -O tensorflow.zip https://github.com/tensorflow/tensorflow/archive/master.zip
unzip -q -d $data_dir/tensorflow tensorflow.zip

echo "PyTorch..."
wget -O pytorch.zip https://github.com/pytorch/pytorch/archive/master.zip
unzip -q -d $data_dir/pytorch pytorch.zip

echo "Cleanup..."
rm sklearn.zip
rm tensorflow.zip
rm pytorch.zip
