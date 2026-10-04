"""相机系 → 世界系（对齐 GVHMR global 约定）：单帧 HMR 只给相机系结果，视频动捕要的是落地的世界运动。

GVHMR global 约定（用参考素材反推 + hand_body_merge.py 的 RZ180 补偿印证）：
  Y 朝上（重力反方向）；Z = 首帧相机视线在水平面上的投影；X = Y × Z（相机的左边）；
  首帧根位移 transl 的 XZ = 0；单位米。水平相机时 世界 = RZ180·相机（OpenCV：x 右 y 下 z 前）。

做法（静止相机）：
  1. 初始"上"= 躯干方向（骨盆→颈）在全片的平均。
  2. 脚的接地点：每帧脚部关节里最低、且速度小的点；对全片接地点做 RANSAC 平面拟合，
     平面法线 = 重力方向（比躯干平均准：人会弯腰、前倾，地面不会）。迭代 3 次。
  3. 旋转 R_wc 把相机系转到世界系；SMPL-X 根旋转绕静止骨盆关节 J0 转，所以
       transl_w = R_wc · (J0 + transl_c) − J0 + 偏移，global_orient_w = R_wc · global_orient_c
  4. 偏移：地面 → y = 0，首帧 transl 的 XZ → 0。
报告：相机俯仰/侧倾、平面残差、接地点数、躯干平均与地面法线的夹角（> 15° 说明可能是运动相机，需人工看）。
"""
from __future__ import annotations

import numpy as np

from . import rot

UP_CAM_LEVEL = np.array([0.0, -1.0, 0.0])         # OpenCV 相机系里水平相机的"上"


def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / max(np.linalg.norm(v), 1e-12)


def _fit_plane(P, thr=0.02, iters=400, seed=0):
    """RANSAC + 内点最小二乘。返回 (n, d, 内点掩码)，平面 n·p = d。"""
    rng = np.random.default_rng(seed)
    best = None
    n_pts = len(P)
    for _ in range(iters if n_pts >= 3 else 0):
        i = rng.choice(n_pts, 3, replace=False)
        n = np.cross(P[i[1]] - P[i[0]], P[i[2]] - P[i[0]])
        if np.linalg.norm(n) < 1e-9:
            continue
        n = n / np.linalg.norm(n)
        inl = np.abs(P @ n - n @ P[i[0]]) < thr
        if best is None or inl.sum() > best.sum():
            best = inl
    inl = best if best is not None else np.ones(n_pts, bool)
    Q = P[inl]
    c = Q.mean(0)
    _, _, Vt = np.linalg.svd(Q - c)
    n = Vt[2]
    return n, float(n @ c), inl


def estimate_up(joints_c, foot_idx, pelvis_idx=0, neck_idx=12, *, fps=30.0, band=0.04, speed_thr=0.6, iters=3):
    """joints_c:(F,J,3) 相机系 → (up 单位向量, 地面 d（up·p = d）, 报告)。speed_thr 单位 m/s。"""
    J = np.asarray(joints_c, dtype=np.float64)
    torso = _unit(np.mean([_unit(v) for v in (J[:, neck_idx] - J[:, pelvis_idx])], axis=0))
    up = torso.copy()
    feet = J[:, list(foot_idx)]                                    # (F,K,3)
    vel = np.zeros(feet.shape[:2])
    vel[1:] = np.linalg.norm(np.diff(feet, axis=0), axis=2) * fps
    info = {}
    for it in range(iters):
        h = feet @ up                                              # (F,K) 沿 up 的高度
        low = h.min(axis=1, keepdims=True)
        floor_h = np.percentile(low, 20)                           # 跳舞：大部分时间至少一只脚着地
        cand = (h - floor_h < band) & (vel < speed_thr)
        P = feet[cand]
        if len(P) < 30:
            info["warning"] = f"接地点只有 {len(P)} 个，退回躯干方向"
            return torso, float(np.percentile(low, 5)), dict(info, contacts=int(len(P)), torso_vs_floor_deg=0.0)
        n, d, inl = _fit_plane(P)
        if n @ torso < 0:
            n, d = -n, -d
        up = n
        info = {"contacts": int(len(P)), "inliers": int(inl.sum()),
                "plane_rms_m": round(float(np.sqrt(np.mean((P[inl] @ n - d) ** 2))), 4)}
    info["torso_vs_floor_deg"] = round(float(np.degrees(np.arccos(np.clip(torso @ up, -1, 1)))), 2)
    return up, d, info


def camera_rotation(up_c) -> np.ndarray:
    """R_wc：相机系向量 → 世界系（Y=up，Z=相机视线水平投影，X=Y×Z）。"""
    y = _unit(up_c)
    z = np.array([0.0, 0.0, 1.0])
    z = _unit(z - (z @ y) * y)
    x = np.cross(y, z)
    return np.stack([x, y, z])                                     # 行 = 世界轴在相机系里的表示


def camera_tilt(up_c) -> dict:
    """相机俯仰/侧倾（度）：水平相机 up = (0,−1,0)。俯仰为正 = 相机朝下看。"""
    u = _unit(up_c)
    pitch = np.degrees(np.arctan2(-u[2], -u[1]))
    roll = np.degrees(np.arctan2(u[0], -u[1]))
    return {"pitch_deg": round(float(pitch), 2), "roll_deg": round(float(roll), 2)}


def to_world(global_orient_c, transl_c, J0, up_c, floor_d):
    """相机系 SMPL-X 根参数 → 世界系（地面 y=0，首帧 transl 的 XZ=0）。J0 = 该体型静止骨盆关节位置。"""
    R_wc = camera_rotation(up_c)
    go = rot.rotvec_to_mat(np.asarray(global_orient_c, dtype=np.float64))
    go_w = rot.mat_to_rotvec(np.einsum("ij,fjk->fik", R_wc, go))
    J0 = np.asarray(J0, dtype=np.float64).reshape(3)
    pelvis_c = np.asarray(transl_c, dtype=np.float64) + J0         # 骨盆在相机系的位置
    t_w = pelvis_c @ R_wc.T - J0
    t_w[:, 1] -= floor_d                                           # 地面在世界 y = up·p = floor_d 处
    t_w[:, 0] -= t_w[0, 0]
    t_w[:, 2] -= t_w[0, 2]
    return go_w, t_w, R_wc
