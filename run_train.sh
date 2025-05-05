#!/usr/bin/env bash

python train.py --epochs 10 \
                --net_name aod-local \
                --lr 1e-4 \
                --use_gpu false \
                --gpu -1 \
                --ori_data_path ./data/train/ori/ \
                --haze_data_path ./data/train/haze/ \
                --val_ori_data_path ./data/train/ori/ \
                --val_haze_data_path ./data/train/haze/ \
                --num_workers 0 \
                --batch_size 4 \
                --val_batch_size 4 \
                --print_gap 10 \
                --model_dir ./models \
                --log_dir ./logs \
                --sample_output_folder ./samples
