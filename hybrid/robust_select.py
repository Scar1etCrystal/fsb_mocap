"""HaMeR 逐帧结果的"时间 × 多裁剪"稳健选择（阶段 B）。

每帧在自己的 K 个裁剪结果里，挑与前后 w 帧全部裁剪结果（(2w+1)·K 个样本）角度差之和最小的那个：
孤立的单帧翻手（HaMeR 在手心/手背模棱两可时一帧翻 ~180°）被时间上下文投掉，而且不引入时间平移（只在本帧的候选里选）。
本帧所有裁剪都和窗口共识差很远 = 真离群 → 交给门控补帧。
"""
from __future__ import annotations

import numpy as np

from fsb import rot


def select(T, P, has, w=2):
    """T: (F, K, 3, 3) 各裁剪的手朝向；P: (F, K, 15, 3, 3) 手指；has: (F,) 有结果。
    返回 (朝向 (F,3,3), 手指 (F,15,3,3), 选中的裁剪下标 (F,), 与窗口共识的角度差 (F,))。"""
    F, K = T.shape[:2]
    Q = rot.mat_to_quat(T.reshape(-1, 3, 3)).reshape(F, K, 4)              # (w,x,y,z)
    idx = np.zeros(F, int)
    dev = np.zeros(F)
    for f in range(F):
        if not has[f]:
            continue
        lo, hi = max(0, f - w), min(F, f + w + 1)
        win = Q[lo:hi][has[lo:hi]].reshape(-1, 4)                          # 窗口内全部样本
        dots = np.abs(Q[f] @ win.T).clip(0, 1)                              # (K, S)：|<q,q'>| = cos(角/2)
        cost = (2 * np.degrees(np.arccos(dots))).sum(1)
        k = int(cost.argmin())
        idx[f] = k
        # 窗口共识 = 窗口样本里的中位样本；选中的结果离它多远
        dd = (2 * np.degrees(np.arccos(np.abs(win @ win.T).clip(0, 1)))).sum(1)
        med = win[int(dd.argmin())]
        dev[f] = 2 * np.degrees(np.arccos(min(1.0, abs(float(Q[f, k] @ med)))))
    sel = np.arange(F)
    return T[sel, idx], P[sel, idx], idx, dev
