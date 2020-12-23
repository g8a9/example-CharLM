#!/bin/bash

mkdir source_data

echo "Scikit-learn..."
wget -O sklearn.zip https://github.com/scikit-learn/scikit-learn/archive/master.zip
unzip -q -d sklearn sklearn.zip
mv sklearn source_data

echo "Tensorflow..."
wget -O tensorflow.zip https://github.com/tensorflow/tensorflow/archive/master.zip
unzip -q -d tensorflow tensorflow.zip
mv tensorflow source_data

echo "PyTorch..."
wget -O pytorch.zip https://github.com/pytorch/pytorch/archive/master.zip
unzip -q -d pytorch pytorch.zip
mv pytorch source_data

echo "Cleanup..."
rm sklearn.zip
rm tensorflow.zip
rm pytorch.zip
