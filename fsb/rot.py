"""旋转工具（只用 numpy）：轴角 / 旋转矩阵 / 四元数 互转、测地角、SMPL-X 旋转链。

约定：轴角 (…,3)，矩阵 (…,3,3)，四元数 (…,4) = (w, x, y, z)。
"""
from __future__ import annotations

import numpy as np

# SMPL-X 55 关节（与 PoseCapture 插件 SMPLX_JOINT_NAMES / SMPLX_PARENT_INDICES 逐项一致）
SMPLX_JOINT_NAMES = (
    "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee", "Spine2", "L_Ankle", "R_Ankle",
    "Spine3", "L_Foot", "R_Foot", "Neck", "L_Collar", "R_Collar", "Head", "L_Shoulder",
    "R_Shoulder", "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist", "Jaw", "L_Eye", "R_Eye",
    "L_Index1", "L_Index2", "L_Index3", "L_Middle1", "L_Middle2", "L_Middle3",
    "L_Pinky1", "L_Pinky2", "L_Pinky3", "L_Ring1", "L_Ring2", "L_Ring3",
    "L_Thumb1", "L_Thumb2", "L_Thumb3", "R_Index1", "R_Index2", "R_Index3",
    "R_Middle1", "R_Middle2", "R_Middle3", "R_Pinky1", "R_Pinky2", "R_Pinky3",
    "R_Ring1", "R_Ring2", "R_Ring3", "R_Thumb1", "R_Thumb2", "R_Thumb3",
)
SMPLX_PARENTS = (
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19,
    15, 15, 15,
    20, 25, 26, 20, 28, 29, 20, 31, 32, 20, 34, 35, 20, 37, 38,
    21, 40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 50, 21, 52, 53,
)
# body_pose 的 21 个关节 = SMPL-X 关节 1..21；手腕在 body_pose 里的下标
BODY_JOINTS = SMPLX_JOINT_NAMES[1:22]
WRIST_BODY_IDX = {"left": 19, "right": 20}          # body_pose.reshape(21,3)[19/20]
WRIST_COLS = {"left": slice(57, 60), "right": slice(60, 63)}
# HaMeR 把左手当镜像右手处理：真左手 = M · 原始 · M
LEFT_MIRROR = np.diag([-1.0, 1.0, 1.0])


def rotvec_to_mat(rv) -> np.ndarray:
    rv = np.asarray(rv, dtype=np.float64)
    th = np.linalg.norm(rv, axis=-1, keepdims=True)
    k = rv / np.maximum(th, 1e-12)
    kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]
    zeros = np.zeros_like(kx)
    K = np.stack([zeros, -kz, ky, kz, zeros, -kx, -ky, kx, zeros], axis=-1).reshape(rv.shape[:-1] + (3, 3))
    th = th[..., None]
    return np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)


def mat_to_quat(R) -> np.ndarray:
    """Shepperd 法，数值稳；返回 w ≥ 0 的单位四元数。"""
    R = np.asarray(R, dtype=np.float64)
    m00, m11, m22 = R[..., 0, 0], R[..., 1, 1], R[..., 2, 2]
    tr = m00 + m11 + m22
    cand = np.stack([tr, m00, m11, m22], axis=-1)
    which = np.argmax(cand, axis=-1)
    q = np.empty(R.shape[:-2] + (4,))
    s = np.sqrt(np.maximum(1.0 + 2.0 * np.take_along_axis(cand, which[..., None], -1)[..., 0] - tr, 1e-24)) * 2.0
    # w 最大
    m = which == 0
    q[m, 0] = 0.25 * s[m]
    q[m, 1] = (R[m][:, 2, 1] - R[m][:, 1, 2]) / s[m]
    q[m, 2] = (R[m][:, 0, 2] - R[m][:, 2, 0]) / s[m]
    q[m, 3] = (R[m][:, 1, 0] - R[m][:, 0, 1]) / s[m]
    for i in (1, 2, 3):
        m = which == i
        if not m.any():
            continue
        Rm = R[m]
        j, k = (i % 3) + 1, ((i + 1) % 3) + 1           # 循环下一个轴
        a, b, c = i - 1, j - 1, k - 1
        q[m, i] = 0.25 * s[m]
        q[m, 0] = (Rm[:, c, b] - Rm[:, b, c]) / s[m]
        q[m, j] = (Rm[:, b, a] + Rm[:, a, b]) / s[m]
        q[m, k] = (Rm[:, c, a] + Rm[:, a, c]) / s[m]
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    q *= np.where(q[..., :1] < 0, -1.0, 1.0)
    return q


def quat_to_mat(q) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], axis=-1).reshape(q.shape[:-1] + (3, 3))


def quat_to_rotvec(q) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    q = q * np.where(q[..., :1] < 0, -1.0, 1.0)
    v = q[..., 1:]
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    ang = 2.0 * np.arctan2(n, q[..., :1])
    return np.where(n > 1e-12, v / np.maximum(n, 1e-12) * ang, 2.0 * v)


def mat_to_rotvec(R) -> np.ndarray:
    """经四元数转换，π 附近也稳。"""
    return quat_to_rotvec(mat_to_quat(R))


def project_so3(M) -> np.ndarray:
    """最近的旋转矩阵（SVD），去掉数值漂移/反射。"""
    M = np.asarray(M, dtype=np.float64)
    U, _, Vt = np.linalg.svd(M)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.ones(M.shape[:-2] + (3,))
    D[..., 2] = d
    return (U * D[..., None, :]) @ Vt


def geodesic_deg(Ra, Rb) -> np.ndarray:
    """两组旋转矩阵之间的夹角（度）。用相对旋转的四元数 2·atan2(|v|,|w|)：

    arccos((tr−1)/2) 在小角度处病态（float32 矩阵的 1e-7 舍入就放大成 ~0.02°），
    atan2 形式在 0° 与 180° 附近都准。"""
    Ra = np.asarray(Ra, dtype=np.float64)
    Rb = np.asarray(Rb, dtype=np.float64)
    q = mat_to_quat(np.swapaxes(Ra, -1, -2) @ Rb)
    return np.degrees(2.0 * np.arctan2(np.linalg.norm(q[..., 1:], axis=-1), np.abs(q[..., 0])))


def quat_continuity(q) -> np.ndarray:
    """沿第 0 维翻号，使相邻四元数点积 ≥ 0（插值/滤波前必做）。"""
    q = np.array(q, dtype=np.float64, copy=True)
    for t in range(1, q.shape[0]):
        flip = np.sum(q[t] * q[t - 1], axis=-1) < 0
        q[t][flip] *= -1.0
    return q


def mirror_lr(R) -> np.ndarray:
    """M·R·M（M = diag(-1,1,1)）：左右手表示互换，自逆。"""
    return LEFT_MIRROR @ np.asarray(R, dtype=np.float64) @ LEFT_MIRROR


def chain_global_rotations(global_orient, body_pose, hand_left=None, hand_right=None) -> np.ndarray:
    """只算旋转的正运动学：(F,3) + (F,63) [+ 手 (F,45)×2] → (F,J,3,3) 各关节世界旋转。

    J = 22（只有身体）或 55（带手，jaw/eyes 视为单位旋转）。"""
    go = rotvec_to_mat(np.asarray(global_orient).reshape(-1, 3))
    F = go.shape[0]
    bp = rotvec_to_mat(np.asarray(body_pose).reshape(F, 21, 3))
    with_hands = hand_left is not None and hand_right is not None
    J = 55 if with_hands else 22
    local = np.tile(np.eye(3), (F, J, 1, 1))
    local[:, 0] = go
    local[:, 1:22] = bp
    if with_hands:
        local[:, 25:40] = rotvec_to_mat(np.asarray(hand_left).reshape(F, 15, 3))
        local[:, 40:55] = rotvec_to_mat(np.asarray(hand_right).reshape(F, 15, 3))
    out = np.empty_like(local)
    for j in range(J):
        p = SMPLX_PARENTS[j]
        out[:, j] = local[:, j] if p < 0 else out[:, p] @ local[:, j]
    return out
