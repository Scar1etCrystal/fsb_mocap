"""SMPL-X 线性蒙皮（numpy，零体型、无姿态校正）——只用于可视化核对重定向结果。"""
from __future__ import annotations

import numpy as np

from . import rot


class SMPLXLBS:
    def __init__(self, npz_path):
        z = np.load(npz_path, allow_pickle=True)
        self.v = z["v_template"].astype(np.float64)
        self.w = z["weights"].astype(np.float64)
        self.f = z["f"].astype(np.int64)
        self.J = z["J_regressor"].astype(np.float64) @ self.v
        self.hands_mean = {"left": z["hands_meanl"].astype(np.float64), "right": z["hands_meanr"].astype(np.float64)}

    def pose(self, G, root_pos=None):
        """G:(55,3,3) 全局旋转 → 顶点 (V,3)。骨盆放在 root_pos（缺省 = 静止骨盆位置）。"""
        p = np.zeros((55, 3))
        for j in range(55):
            par = rot.SMPLX_PARENTS[j]
            p[j] = self.J[j] if par < 0 else p[par] + G[par] @ (self.J[j] - self.J[par])
        if root_pos is not None:
            p += np.asarray(root_pos) - p[0]
        # v' = Σ w_j (G_j (v − J_j) + p_j)
        vj = np.einsum("jab,vjb->vja", G, self.v[:, None, :] - self.J[None]) + p[None]
        return np.einsum("vj,vja->va", self.w, vj), p
