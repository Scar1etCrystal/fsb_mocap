#!/usr/bin/env bash
# 阶段 B ②：后台跑 setsid nohup bash run_fsb_shared.sh > /root/autodl-tmp/logs/fsb_shared.log 2>&1 < /dev/null &
cd /root/autodl-tmp
source fsb_frontend/cloud/env.sh                     # FastSAM3DBody 推荐设置（IMG_SIZE、FOV_* …）与 HF/torch 缓存目录
export PATH=/root/autodl-tmp/envs/hamer311/bin:$PATH  # 换成 HAMER 同版本环境的 python
export PYTHONPATH=/root/autodl-tmp/fsb_overlay_py311:$PYTHONPATH
export HF_ENDPOINT=https://hf-mirror.com
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
python fsb_frontend/cloud/hamer/run_fsb_shared.py "$@"
