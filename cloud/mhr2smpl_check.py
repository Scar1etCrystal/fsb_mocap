"""交叉验证用：Fast SAM 3D Body 自带的前馈 MHR→SMPL（身体 21 关节）跑整段，输出与我们的骨骼重定向对比。

复用 mocap/core/multiview_mhr2smpl.MultiViewFusionRunner 的网络与预处理（单视角），但 infer() 不返回根朝向，
这里按同一流程批量算：顶点与 cam_t 翻 y/z（→ 模型轴向的相机系）→ 重心映射到 SMPL → 取 1500 点去中心 → 网络。
用法：python mhr2smpl_check.py <take 目录>   → mhr2smpl.npz（go_aa (F,3) 模型轴相机系，body_pose (F,63)，betas (F,10)）
"""
import glob
import os
import sys

import numpy as np
import torch

FSB = "/root/autodl-tmp/Fast-SAM-3D-Body"
sys.path.insert(0, FSB)
from mocap.core.multiview_mhr2smpl import MultiViewFusionRunner, rot6d_to_aa, split_output  # noqa: E402

take = sys.argv[1]
runner = MultiViewFusionRunner(smpl_model_path=os.path.join(FSB, "mhr2smpl/data/SMPL_NEUTRAL.pkl"),
                               model_dir=os.path.join(FSB, "mhr2smpl/experiments/multiview_n30000_e500"),
                               mapping_path=os.path.join(FSB, "mhr2smpl/data/mhr2smpl_mapping.npz"), device="cuda")
gos, bps, bes = [], [], []
for part in sorted(glob.glob(os.path.join(take, "sam3d_part_*.npz"))):
    z = np.load(part)
    feats = np.stack([runner._preprocess_view(v.astype(np.float32), t) for v, t in zip(z["pred_vertices"], z["pred_cam_t"])])
    x = torch.from_numpy(feats).unsqueeze(1).to(runner.device)                 # (B, 1 视角, 4500)
    with torch.no_grad():
        params, _ = runner.model(x, torch.ones(len(feats), 1, dtype=torch.bool, device=runner.device))
        go6d, body_pose, betas = split_output(params)
        gos.append(rot6d_to_aa(go6d).cpu().numpy()); bps.append(body_pose.cpu().numpy()); bes.append(betas.cpu().numpy())
np.savez(os.path.join(take, "mhr2smpl.npz"), go_aa=np.concatenate(gos), body_pose=np.concatenate(bps), betas=np.concatenate(bes))
print("mhr2smpl", take, np.concatenate(bps).shape)
