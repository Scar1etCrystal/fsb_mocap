# 云端运行环境（Fast SAM 3D Body 推荐设置 + 我们的路径）；用法：source env.sh
export FAST_SAM3D_ROOT=/root/autodl-tmp/envs/fast_sam_3d_body
export PATH=$FAST_SAM3D_ROOT/bin:$PATH
export HF_HOME=/root/autodl-tmp/hf-cache TORCH_HOME=/root/autodl-tmp/torch-hub YOLO_CONFIG_DIR=/root/autodl-tmp/yolo-cfg
export PYTHONPATH=/root/autodl-tmp/Fast-SAM-3D-Body:/root/autodl-tmp/fsb_frontend
export IMG_SIZE=512 LAYER_DTYPE=fp32 SKIP_KEYPOINT_PROMPT=1 GPU_HAND_PREP=1
export BODY_INTERM_PRED_LAYERS=0,1,2 HAND_INTERM_PRED_LAYERS=0,1 KEYPOINT_PROMPT_INTERM_INTERVAL=999
export MHR_USE_CUDA_GRAPH=0 FOV_TRT=0 FOV_FAST=1 FOV_MODEL=s FOV_LEVEL=0
export USE_COMPILE=${USE_COMPILE:-0} USE_COMPILE_BACKBONE=${USE_COMPILE:-0} DECODER_COMPILE=${USE_COMPILE:-0}
# 注意：不设 MHR_NO_CORRECTIVES（保留姿态校正，网格更准；骨架不受影响）
