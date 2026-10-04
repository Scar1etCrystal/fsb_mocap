"""时间处理（照片模型用于视频必须补的一环）：去单帧尖刺 + 切空间高斯平滑（离线、零延迟）。

单帧 HMR 逐帧独立推，结果带高频抖动和偶发单帧翻转；GVHMR 是时序网络，自带平滑。
  despike：d(t−1,t) 与 d(t,t+1) 都 > thr 且 d(t−1,t+1) < thr/2（出去又回来）→ 用两邻帧 slerp 中点替换。
  gaussian：R_s[t] = R[t]·exp(Σ_k w_k·log(R[t]ᵀR[t+k]))，对称核，匀速转动零滞后；
            conf 给了就按置信度加权（遮挡帧向邻帧靠）。
"""
from __future__ import annotations

import numpy as np

from . import rot


def _log(R):
    return rot.mat_to_rotvec(R)


def _exp(v):
    return rot.rotvec_to_mat(v)


def despike(R, thr_deg=30.0, max_pass=3):
    """R:(F,...,3,3) → (去尖刺后的 R, 被替换的帧下标列表)。"""
    R = np.array(R, dtype=np.float64, copy=True)
    fixed = set()
    for _ in range(max_pass):
        d1 = rot.geodesic_deg(R[1:-1], R[:-2])
        d2 = rot.geodesic_deg(R[2:], R[1:-1])
        d02 = rot.geodesic_deg(R[2:], R[:-2])
        sp = (d1 > thr_deg) & (d2 > thr_deg) & (d02 < thr_deg / 2)
        if not sp.any():
            break
        tt, jj = np.nonzero(sp.reshape(sp.shape[0], -1))
        Rf = R.reshape(R.shape[0], -1, 3, 3)
        for t, j in zip(tt + 1, jj):
            a, b = Rf[t - 1, j], Rf[t + 1, j]
            Rf[t, j] = a @ _exp(0.5 * _log(a.T @ b))
            fixed.add(int(t))
        R = Rf.reshape(R.shape)
    return R, sorted(fixed)


def gaussian(R, sigma=1.5, conf=None, radius=None):
    """R:(F,...,3,3)；sigma 单位帧；conf:(F,) 或 (F,…) 可选权重。"""
    R = np.asarray(R, dtype=np.float64)
    F = R.shape[0]
    shp = R.shape
    Rf = R.reshape(F, -1, 3, 3)
    radius = int(radius if radius is not None else max(1, round(3 * sigma)))
    ks = np.arange(-radius, radius + 1)
    w0 = np.exp(-0.5 * (ks / max(sigma, 1e-6)) ** 2)
    c = None if conf is None else np.asarray(conf, dtype=np.float64).reshape(F, -1)
    out = np.empty_like(Rf)
    for t in range(F):
        idx = np.clip(t + ks, 0, F - 1)
        rel = np.swapaxes(Rf[t][None], -1, -2) @ Rf[idx]           # (K,J,3,3)
        v = _log(rel)                                              # (K,J,3)
        w = np.broadcast_to(w0[:, None], v.shape[:2]).copy()
        if c is not None:
            w = w * (c[idx] if c.shape[1] == v.shape[1] else c[idx][:, :1])
        w = w / np.maximum(w.sum(0, keepdims=True), 1e-12)
        out[t] = Rf[t] @ _exp((w[..., None] * v).sum(0))
    return out.reshape(shp)


def gaussian_vec(x, sigma=1.5, radius=None):
    x = np.asarray(x, dtype=np.float64)
    F = x.shape[0]
    radius = int(radius if radius is not None else max(1, round(3 * sigma)))
    ks = np.arange(-radius, radius + 1)
    w = np.exp(-0.5 * (ks / max(sigma, 1e-6)) ** 2)
    w /= w.sum()
    out = np.zeros_like(x)
    for k, wk in zip(ks, w):
        out += wk * x[np.clip(np.arange(F) + k, 0, F - 1)]
    return out
