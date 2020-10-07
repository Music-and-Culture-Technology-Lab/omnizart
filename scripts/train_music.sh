#!/bin/bash

omnizart music train-model \
    -d /data/omnizart/tf_dataset_experiment/feature \
    -y aspp \
    --label-type pop-note-stream \
    --loss-function bce \
    --timesteps 128 \
    -e 4 \
    -s 17 \
    -b 8 \
    -vs 20 \
    -vb 8 \
    --early-stop 1

