#!/usr/bin/env bash

# activate base
source /app/dev/miniconda3/bin/activate base
conda env list

# update source
cd /app/source/ChatGLM-6B/ptuning || exit 1
git pull

# start training
bash train.sh 1>log_train.log 2>&1 &
