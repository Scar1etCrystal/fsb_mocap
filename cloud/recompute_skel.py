"""用最终输出参数重算 MHR 骨架（关节全局旋转 + 位置），保证与最终网格一致。

原因（2026-10-04 实测）：pred_global_rots 的手部关节与 pred_joint_coords 不一致（手腕→指根方向在腕坐标系里
漂 30–50°）——SAM3D 先出身体、再单独解手、按判据把手融合回去，pred_global_rots 是融合前的值。
用法：python recompute_skel.py <take 目录>   → 每个 sam3d_part_XXXX.npz 旁写 skel_part_XXXX.npz
"""
import glob
import os
import sys

import numpy as np
import torch

FSB = os.environ.get("FSB_ROOT", "/root/autodl-tmp/Fast-SAM-3D-Body")  # 新流程见 fsb/pipeline.py + winpack/run_video.py
sys.path.insert(0, FSB)
from sam_3d_body import load_sam_3d_body  # noqa: E402

model, _ = load_sam_3d_body(os.path.join(FSB, "checkpoints/sam-3d-body-dinov3/model.ckpt"), device="cuda",
                            mhr_path=os.path.join(FSB, "checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt"))
head = model.head_pose
for part in sorted(glob.glob(os.path.join(sys.argv[1], "sam3d_part_*.npz"))):
    z = np.load(part)
    t = lambda k: torch.from_numpy(z[k].astype(np.float32)).cuda()
    with torch.no_grad():
        out = head.mhr_forward(global_trans=t("global_rot") * 0, global_rot=t("global_rot"), body_pose_params=t("body_pose_params"),
                               hand_pose_params=t("hand_pose_params"), scale_params=t("scale_params"), shape_params=t("shape_params"),
                               expr_params=torch.zeros(len(z["global_rot"]), 72).cuda(), return_joint_rotations=True,
                               return_joint_coords=True)
    verts, jcoords, jrots = out                      # 返回顺序：verts, [keypoints], joint_coords, [model_params], joint_rots
    dst = part.replace("sam3d_part_", "skel_part_")
    np.savez(dst, joint_coords_model=jcoords.detach().cpu().numpy().astype(np.float32),
             joint_rots_model=jrots.detach().cpu().numpy().astype(np.float32))
    res = {"joint_coords": jcoords, "joint_rots": jrots}
    print(dst, {k: tuple(v.shape) for k, v in res.items() if v is not None and hasattr(v, "shape")})
