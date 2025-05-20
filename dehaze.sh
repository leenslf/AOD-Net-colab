#!/usr/bin/env bash
# 📍 Paths
NET_NAME=""                         # no subfolder (or set to your net module name)
CKPT_NAME="AOD_9.pkl"
MODEL_DIR="/content/AOD-Net-colab/models"
TEST_IMG_DIR="/content/drive/MyDrive/clearDive/100_raw"
SAVE_DIR="/content/drive/MyDrive/clearDive/100_dehazed"

# ✅ Ensure results directory exists
mkdir -p "$SAVE_DIR"

# 🚀 Run demo.py with corrected model path
python demo.py \
  --net_name "$NET_NAME" \
  --use_gpu true \
  --model_dir "$MODEL_DIR" \
  --ckpt "$CKPT_NAME" \
  --test_img_dir "$TEST_IMG_DIR" \
  --sample_output_folder "$SAVE_DIR"
