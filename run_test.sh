#!/usr/bin/env bash

python demo.py  --net_name aod \
                --use_gpu true \
                --gpu 3 \
                --model_dir ./models
                --ckpt AOD_9.pkl
