"""在云端（CPU 即可）导出：MHR 零位骨架 + 随机姿态骨架状态 + SMPL-X 静止骨架 → 一个 npz，拿回本地开发重定向。"""
import sys
import numpy as np
import torch

OUT = sys.argv[1]
torch.manual_seed(0)
m = torch.jit.load("/root/autodl-tmp/Fast-SAM-3D-Body/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt", map_location="cpu")
sk = m.character_torch.skeleton
names = list(sk.joint_names)
parents = np.array([int(p) for p in sk.joint_parents])
n_total = m.character_torch.parameter_transform.parameter_transform.shape[1]
n_model = n_total - 45          # forward 会把 45 维 identity 系数拼到模型参数后面（249 = 204 + 45）
pose_mask = m.character_torch.parameter_transform.pose_parameters.numpy()[:n_model]
rigid_mask = m.character_torch.parameter_transform.rigid_parameters.numpy()[:n_model]
scale_mask = m.character_torch.parameter_transform.scaling_parameters.numpy()[:n_model]
print("model params", n_model, "pose", pose_mask.sum(), "rigid", rigid_mask.sum(), "scale", scale_mask.sum())


def run(model_params, chunk=16):
    """分批前向：2026-10-04 无卡模式一次 256 个姿态（带姿态校正）被 OOM 杀掉（rc=137）。"""
    vs, ss = [], []
    with torch.no_grad():
        for i in range(0, model_params.shape[0], chunk):
            mp = model_params[i:i + chunk]
            v, s = m(torch.zeros(mp.shape[0], 45), mp, torch.zeros(mp.shape[0], 72), True)
            vs.append(v[:, ::7].clone()); ss.append(s.clone())
    verts, skel = torch.cat(vs), torch.cat(ss)
    skel = skel.numpy().astype(np.float64)
    pos = skel[..., :3] * 0.01                      # cm → m（与 mhr_head 相同）
    q = skel[..., 3:7]                              # (x,y,z,w)
    return pos, q, verts.numpy() * 0.01


zero = torch.zeros(1, n_model)
pos0, q0, v0 = run(zero)
# 随机姿态：只动"姿态"参数（不动全局刚体和缩放），幅度 0.6 rad 量级
N = 256
mp = torch.zeros(N, n_model)
idx = torch.from_numpy(np.nonzero(pose_mask & ~rigid_mask & ~scale_mask)[0])
mp[:, idx] = torch.randn(N, len(idx)) * 0.35
# 全局朝向也随机（刚体旋转参数）
rig = torch.from_numpy(np.nonzero(rigid_mask)[0])
print("rigid param idx", rig.tolist())
pos, q, _ = run(mp)
smplx = np.load("/root/autodl-tmp/mhr_assets/SMPLX_NEUTRAL.npz", allow_pickle=True)
J = smplx["J_regressor"] @ smplx["v_template"]
kt = smplx["kintree_table"]
np.savez_compressed(OUT, mhr_names=np.array(names), mhr_parents=parents, mhr_rest_pos=pos0[0], mhr_rest_quat_xyzw=q0[0],
                    mhr_rand_pos=pos, mhr_rand_quat_xyzw=q, mhr_rest_verts=v0[0],
                    smplx_rest_joints=J[:55], smplx_parents=np.array([int(x) if int(x) < 2**31 else -1 for x in kt[0]][:55]))
print("saved", OUT, "smplx joints", J.shape, "mhr", pos0.shape, "rand", pos.shape)
print("mhr rest: root", pos0[0, 1].round(3), "head", pos0[0, names.index("c_head")].round(3), "l_ball", pos0[0, names.index("l_ball")].round(3),
      "l_wrist", pos0[0, names.index("l_wrist")].round(3))
print("smplx rest: pelvis", J[0].round(3), "head", J[15].round(3), "l_foot", J[10].round(3), "l_wrist", J[20].round(3))
