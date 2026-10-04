"""MHR（SAM 3D Body 骨架，127 关节）→ SMPL-X 关节集 的骨骼级重定向——不做网格拟合。

为什么不用 MHR→SMPL(-X) 网格拟合：官方路线是逐帧迭代拟合（Fast SAM 3D Body 论文里被替换掉的瓶颈），
它自带的前馈映射只出 SMPL 身体、没有手指。PoseCapture 插件只读 SMPL-X 关节集上的局部旋转（不读 betas），
所以直接把 MHR 骨架的姿态搬到 SMPL-X 关节上即可：每帧纯矩阵运算，脚和手指保留 SAM3D 自己的估计。

坐标：SAM3D 的 pred_global_rots 在 MHR 模型系（x = 角色左、y 上、z 前，与 SMPL-X 相同），
pred_joint_coords / pred_keypoints_3d / pred_cam_t 在 OpenCV 相机系（y 下）：P_cam = F·P_model，F = diag(1,−1,−1)。

做法：
  零位标定 A_j = MHR 零位姿态下 SMPL-X 关节 j 应有的全局旋转：躯干类零位对零位（A = I）；肢体按两边零位骨向
  对齐（补 A 姿态↔T 姿态的差，单子骨用最小转动）；手腕用"腕→中指根 + 小指根→食指根"成架。C_j = R0_mhr⁻¹·A_j。
  每帧 G_j = R_mhr(t)·C_j；跨多节 MHR 的骨（踝→脚尖、膝→踝、骨盆→脊柱 1）再按 MHR 关节位置做一次
  摆动校正，使 SMPL-X 骨的指向与 MHR 完全一致；局部旋转 L_j = G_parent⁻¹·G_j。
"""
from __future__ import annotations

import numpy as np

from . import rot

F_FLIP = np.diag([1.0, -1.0, -1.0])

FINGERS = ("index", "middle", "pinky", "ring", "thumb")      # SMPL-X / MANO 手指顺序


def _spec():
    """smplx 关节 → (旋转取自的 MHR 关节, [(smplx 子关节或 None, MHR 起点, MHR 终点)…], 摆动校正?)"""
    # 躯干类（骨盆/脊柱/颈/头/锁骨/脚尖）：两边零位都是直立中性姿态，零位对零位（A = I），逐帧不强制骨向——
    # SMPL-X 脊柱各节很短、关节连线本来就是折线，按骨向逐节对齐会把"骨架几何不同"当成姿态（实测拧出 61°）。
    s = {
        0: ("root", [], False),
        3: ("c_spine1", [], False),
        6: ("c_spine2", [], False),
        9: ("c_spine3", [], False),
        12: ("c_neck", [], False),
        15: ("c_head", [], False),
    }
    for side, k, o in (("l", 0, 0), ("r", 1, 1)):
        hip, knee, ankle, foot = 1 + k, 4 + k, 7 + k, 10 + k
        collar, shoulder, elbow, wrist = 13 + k, 16 + k, 18 + k, 20 + k
        s[hip] = (f"{side}_upleg", [(knee, f"{side}_upleg", f"{side}_lowleg")], True)
        s[knee] = (f"{side}_lowleg", [(ankle, f"{side}_lowleg", f"{side}_talocrural")], True)
        s[ankle] = (f"{side}_transversetarsal", [(foot, f"{side}_talocrural", f"{side}_ball")], True)
        s[foot] = (f"{side}_ball", [], False)                                    # 脚尖：零位水平朝前（两边一致）
        s[collar] = (f"{side}_clavicle", [], False)                              # 锁骨：零位对零位（两边锁骨起点结构不同）
        s[shoulder] = (f"{side}_uparm", [(elbow, f"{side}_uparm", f"{side}_lowarm")], True)
        s[elbow] = (f"{side}_lowarm", [(wrist, f"{side}_lowarm", f"{side}_wrist")], True)
        base = 25 if side == "l" else 40
        s[wrist] = (f"{side}_wrist", [(base + 3, f"{side}_wrist", f"{side}_middle1")], False)
        for i, f in enumerate(FINGERS):
            j1, j2, j3 = base + 3 * i, base + 3 * i + 1, base + 3 * i + 2
            s[j1] = (f"{side}_{f}1", [(j2, f"{side}_{f}1", f"{side}_{f}2")], True)
            s[j2] = (f"{side}_{f}2", [(j3, f"{side}_{f}2", f"{side}_{f}3")], True)
            s[j3] = (f"{side}_{f}3", [(None, f"{side}_{f}3", f"{side}_{f}_null")], True)
    return s


SPEC = _spec()
LATERAL = {                                       # 手腕：腕→中指根 + 小指根→食指根 成架（掌面朝向要准）
    20: (31, 25, "l_pinky1", "l_index1"),
    21: (46, 40, "r_pinky1", "r_index1"),
}


def _frame(primary, lateral):
    """主方向 + 侧向 → 正交架（列：主、侧（正交化）、叉积）。"""
    a = primary / np.linalg.norm(primary)
    b = lateral - a * (lateral @ a)
    b /= np.linalg.norm(b)
    return np.stack([a, b, np.cross(a, b)], 1)


def _wahba(src, dst, w=None):
    """R 使 Σ w‖R·src − dst‖² 最小（src/dst 已归一）。"""
    w = np.ones(len(src)) if w is None else np.asarray(w)
    H = (w[:, None, None] * (np.asarray(dst)[:, :, None] * np.asarray(src)[:, None, :])).sum(0)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


def _swing(a, b):
    """把单位向量 a 转到 b 的最小旋转（a、b 可批量 (...,3)）。"""
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    v = np.cross(a, b)
    c = np.clip(np.sum(a * b, axis=-1), -1.0, 1.0)
    s = np.linalg.norm(v, axis=-1)
    ang = np.arctan2(s, c)
    axis = v / np.maximum(s, 1e-12)[..., None]
    out = rot.rotvec_to_mat(axis * ang[..., None])
    # 反向（180°）时任取一根垂直轴
    anti = c < -0.999999
    if np.any(anti):
        perp = np.cross(a[anti], np.array([1.0, 0.0, 0.0]))
        bad = np.linalg.norm(perp, axis=-1) < 1e-6
        perp[bad] = np.cross(a[anti][bad], np.array([0.0, 1.0, 0.0]))
        perp /= np.linalg.norm(perp, axis=-1, keepdims=True)
        out[anti] = rot.rotvec_to_mat(perp * np.pi)
    return out


class Retargeter:
    def __init__(self, mhr_names, mhr_rest_pos, mhr_rest_quat_xyzw, smplx_rest_joints):
        self.idx = {n: i for i, n in enumerate(list(mhr_names))}
        q = np.asarray(mhr_rest_quat_xyzw, dtype=np.float64)
        self.R0 = rot.quat_to_mat(q[:, [3, 0, 1, 2]])
        self.P0 = np.asarray(mhr_rest_pos, dtype=np.float64)
        self.J = np.asarray(smplx_rest_joints, dtype=np.float64)[:55]
        self.C, self.dirs = {}, {}
        for j, (rj, vecs, _) in SPEC.items():
            src, dst = [], []
            for c, a, b in vecs:
                s = (self.J[c] - self.J[j]) if c is not None else (self.J[j] - self.J[rot.SMPLX_PARENTS[j]])
                m = self.P0[self.idx[b]] - self.P0[self.idx[a]]
                src.append(s / np.linalg.norm(s)); dst.append(m / np.linalg.norm(m))
            if j in LATERAL:
                sa, sb, ma, mb = LATERAL[j]
                Fs = _frame(src[0], self.J[sb] - self.J[sa])
                Fm = _frame(dst[0], self.P0[self.idx[mb]] - self.P0[self.idx[ma]])
                A = Fm @ Fs.T
            elif len(src) >= 2:
                A = _wahba(np.array(src), np.array(dst))
            elif len(src) == 1:
                A = _swing(src[0][None], dst[0][None])[0]
            else:
                A = np.eye(3)
            self.C[j] = self.R0[self.idx[rj]].T @ A
            self.dirs[j] = src[0] if len(src) == 1 else None          # 摆动校正用的 SMPL-X 骨向（局部=零位全局，因 SMPL-X 零位旋转为单位阵）

    def global_rotations(self, R_mhr, P_model=None, swing_fix=True):
        """R_mhr:(F,127,3,3) 模型系；P_model:(F,127,3) 模型系位置（摆动校正用）。→ (F,55,3,3) SMPL-X 全局旋转。"""
        R_mhr = np.asarray(R_mhr, dtype=np.float64)
        Fn = R_mhr.shape[0]
        G = np.tile(np.eye(3), (Fn, 55, 1, 1))
        for j, (rj, vecs, fix) in SPEC.items():
            G[:, j] = R_mhr[:, self.idx[rj]] @ self.C[j]
            if swing_fix and fix and P_model is not None and self.dirs[j] is not None:
                c, a, b = vecs[0]
                target = P_model[:, self.idx[b]] - P_model[:, self.idx[a]]
                cur = G[:, j] @ self.dirs[j]
                G[:, j] = _swing(cur, target) @ G[:, j]
        for j in (22, 23, 24):                                           # 下巴/眼：插件不用，跟头走
            G[:, j] = G[:, 15]
        return G

    @staticmethod
    def local_rotations(G):
        L = np.empty_like(G)
        for j in range(55):
            p = rot.SMPLX_PARENTS[j]
            L[:, j] = G[:, j] if p < 0 else np.swapaxes(G[:, p], -1, -2) @ G[:, j]
        return L

    def smplx_pose(self, R_mhr, P_model=None, swing_fix=True):
        """→ dict(global_orient_model (F,3), body_pose (F,63), left_hand_pose (F,45), right_hand_pose (F,45), G (F,55,3,3))"""
        G = self.global_rotations(R_mhr, P_model, swing_fix)
        L = self.local_rotations(G)
        rv = rot.mat_to_rotvec(L)
        Fn = rv.shape[0]
        return {"G": G, "global_orient_model": rv[:, 0], "body_pose": rv[:, 1:22].reshape(Fn, 63),
                "left_hand_pose": rv[:, 25:40].reshape(Fn, 45), "right_hand_pose": rv[:, 40:55].reshape(Fn, 45)}

    def bone_dir_errors(self, G, P_model):
        """各 SMPL-X 骨（有 MHR 对应向量的）指向与 MHR 的夹角（度），(F, n_bones) + 骨名列表。"""
        errs, names = [], []
        for j, (rj, vecs, _) in SPEC.items():
            for c, a, b in vecs:
                s = (self.J[c] - self.J[j]) if c is not None else (self.J[j] - self.J[rot.SMPLX_PARENTS[j]])
                d = G[:, j] @ (s / np.linalg.norm(s))
                m = P_model[:, self.idx[b]] - P_model[:, self.idx[a]]
                m = m / np.linalg.norm(m, axis=-1, keepdims=True)
                errs.append(np.degrees(np.arccos(np.clip(np.sum(d * m, -1), -1, 1))))
                names.append(f"{rot.SMPLX_JOINT_NAMES[j]}→{rot.SMPLX_JOINT_NAMES[c] if c is not None else 'tip'}")
        return np.stack(errs, 1), names


def from_rest_npz(path):
    z = np.load(path)
    return Retargeter(z["mhr_names"], z["mhr_rest_pos"], z["mhr_rest_quat_xyzw"], z["smplx_rest_joints"])
