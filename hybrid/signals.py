"""手部可信度信号 vs 用户标注的坏段（999：A–E 问题全集；1499：两处遮挡）。只读分析。"""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fsb import convert, retarget, rot  # noqa: E402
from merge_tools import load  # noqa: E402

REF = "/home/sb/fastsam_handoff/reference_output"
RZ180 = np.diag([-1.0, -1.0, 1.0])
M = np.diag([-1.0, 1.0, 1.0])
TAKES = {
    "t999": {"ref": f"{REF}/0001-0999_f_full_20260925_162525v2", "hamer": "0001-0999_f_full_20260925_162525_hamer_data.pkl",
             "fsb": f"{ROOT}/out/0001-0999_f_full_20260925_162525v2_fsb", "raw": f"{ROOT}/data/t999",
             # A–E 问题全集（pkl 帧 = Blender 帧 − 1；HAMER_HAND_REPAIR_PLAN.md 两轮修复用的区间）；E(~771) 在 L 732–778 内
             "bad": {"R": [(531, 555), (686, 695), (785, 790)], "L": [(686, 700), (732, 778), (785, 790), (799, 816)]}},
    "t1499": {"ref": f"{REF}/0001-1499_f_full_20260925_165539", "hamer": "87d473eaee19_hamer_data.pkl",
              "fsb": f"{ROOT}/out/0001-1499_f_full_20260925_165539_fsb", "raw": f"{ROOT}/data/t1499",
              "bad": {"R": [(511, 553)], "L": [(883, 897)]}},           # 遮挡（Blender 512–554 右、884–898 左）
}
SIDES = {"L": 0, "R": 1}                                              # HaMeR 导出顺序：0 左 1 右


def mask(F, segs):
    m = np.zeros(F, bool)
    for a, b in segs:
        m[a:b + 1] = True
    return m


def hamer_world(o, side, R_wc=None):
    """HaMeR 原始 global_orient（右手 MANO，左手是翻转图里的右手）→ 世界系 SMPL-X 手腕全局旋转。
    合并脚本的做法 = RZ180·o（左手再 M·…·M）；给了 R_wc 就用精确的相机→世界旋转代替 RZ180。"""
    Rw = RZ180 if R_wc is None else R_wc
    if side == "L":
        return np.einsum("ij,fjk,kl->fil", Rw @ M, o, M) if R_wc is not None else np.einsum("ij,fjk,kl->fil", M @ RZ180, o, M)
    return np.einsum("ij,fjk->fik", Rw, o)


def fsb_body(take):
    p = load(f"{TAKES[take]['fsb']}/gvhmr/hmr4d_results.pt_person-0.pkl")["smpl_params_global"]
    go, bp = np.asarray(p["global_orient"], float), np.asarray(p["body_pose"], float)
    G = rot.chain_global_rotations(go, bp)                           # (F, 22, 3, 3) 世界系
    return G


def R_wc_of(take):
    rt = retarget.from_rest_npz(os.path.join(ROOT, "fsb", "assets", "mhr_rest.npz"))
    _, _, dbg = convert.convert_take(TAKES[take]["raw"], rt, J0=np.array([0.00312, -0.35141, 0.01204]), fps=30.0)
    return dbg["R_wc"]


def auc(score, label):
    from itertools import accumulate  # noqa: F401
    o = np.argsort(score)
    r = np.empty(len(score)); r[o] = np.arange(1, len(score) + 1)
    n1, n0 = label.sum(), (~label).sum()
    return float((r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def seg_recall(score, segs, thr, tol=2):
    hit = 0
    for a, b in segs:
        if (score[max(0, a - tol):b + tol + 1] > thr).any():
            hit += 1
    return hit


def main():
    rows = []
    for take, cfg in TAKES.items():
        h = load(f"{cfg['ref']}/hamer/{cfg['hamer']}")
        G = fsb_body(take)
        F = G.shape[0]
        R_wc = R_wc_of(take)
        tilt = float(np.degrees(np.arccos(np.clip((np.trace(RZ180.T @ R_wc) - 1) / 2, -1, 1))))
        mt = np.load(f"{cfg['raw']}/main_track.npz")
        yk = np.nan_to_num(mt["kpts"], nan=0.0)
        kp2 = np.concatenate([np.load(f)["pred_keypoints_2d"] for f in sorted(
            __import__("glob").glob(f"{cfg['raw']}/sam3d_part_*.npz"))])
        print(f"== {take}: F={F}, 相机→世界 与水平假设(RZ180)差 {tilt:.2f}°")
        for side, hi in SIDES.items():
            o = np.asarray(h["global_orient"], float)[:F, hi]
            valid = ~np.isclose(o, 0).all(axis=(1, 2))
            bad = mask(F, cfg["bad"][side])
            sig = {}
            # 1 YOLO 手腕置信度（低 = 可疑）
            sig["YOLO手腕置信度(低)"] = -yk[:, 9 if side == "L" else 10, 2]
            # 2 参考 HaMeR 自带 hand_keypoint_conf（只有 999；奇数帧=0 是隔帧检测造成的，用相邻偶数帧的较小值补）
            if "hand_keypoint_conf" in h:
                c = np.asarray(h["hand_keypoint_conf"], float)[:F, hi].copy()
                odd = np.arange(1, F, 2)
                nb = np.minimum(c[np.clip(odd - 1, 0, F - 1)], c[np.clip(odd + 1, 0, F - 1)])
                c[odd] = nb
                sig["ViTPose手关键点max置信(低)"] = -c
            # 3 HaMeR 与 FSB 手腕全局朝向分歧（去掉常值偏差后）
            Hw = hamer_world(o, side, R_wc)
            Fw = G[:, 20 if side == "L" else 21]
            rel = np.einsum("fji,fjk->fik", Fw, Hw)
            qm = rot.mat_to_quat(rel[valid])
            qm = qm * np.sign(qm @ qm[len(qm) // 2])[:, None]
            Rm = rot.quat_to_mat(qm.mean(0) / np.linalg.norm(qm.mean(0)))
            dis = rot.geodesic_deg(rel, np.broadcast_to(Rm, rel.shape))
            sig["HaMeR-FSB手腕分歧"] = np.where(valid, dis, 0)
            # 4 FSB 2D 腕点 vs YOLO 腕点 / 手框
            hb = kp2[:F, 42:63] if side == "L" else kp2[:F, 21:42]
            size = np.maximum(np.ptp(hb[:, :, 0], 1), np.ptp(hb[:, :, 1], 1))
            wi = 62 if side == "L" else 41
            sig["FSB腕点-YOLO腕点偏差"] = np.linalg.norm(kp2[:F, wi] - yk[:, 9 if side == "L" else 10, :2], axis=1) / np.maximum(size, 1)
            # 5 基线：HaMeR 自身朝向帧间角速度（尝试 2 的口径）
            sp = np.r_[0, rot.geodesic_deg(Hw[1:], Hw[:-1])]
            sig["HaMeR角速度(尝试2)"] = sp
            for name, s in sig.items():
                thr = np.percentile(s, 95)
                rec = seg_recall(s, cfg["bad"][side], thr)
                fp = float(((s > thr) & ~mask(F, [(max(0, a - 2), b + 2) for a, b in cfg["bad"][side]])).sum())
                rows.append((take, side, name, auc(s, bad), rec, len(cfg["bad"][side]), fp))
                print(f"  {side} {name:24s} AUC {auc(s, bad):.2f}  段召回@P95 {rec}/{len(cfg['bad'][side])}  段外超阈帧 {int(fp)}")
    return rows


if __name__ == "__main__":
    main()
