#!/bin/bash

source_folder=$1

python train.py $source_folder \
	--gpus 1 \
	--max_epochs 100 \
	--batch_size 4096 \
	--hidden_size 512 \
	--n_layers 3 \
	--gradient_clip_val 1 \
	--num_workers 32 \
