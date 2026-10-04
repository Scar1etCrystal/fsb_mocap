"""阶段 B：锁主角、每帧检测重跑 HaMeR（cloud/hamer/hamer_locked.py）的新信号 vs 用户标注坏段。只读分析。

输入 data/hamer_b/<take>/：<stem>_hamer_raw.npz（逐帧未平滑 + 信号）、<stem>_hamer_data.pkl（同原版平滑）、
     <stem>_hamer_data_i2.pkl（锁主角但隔 2 帧检测）、<stem>_others.npz（其他人的手，仅 999）。
输出：reports/hybrid/signals_b.txt（信号 AUC / 段召回）、reports/hybrid/old_vs_locked.json（旧链路 vs 锁主角）。
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from fsb import rot  # noqa: E402
from merge_tools import load  # noqa: E402
from signals import SIDES, TAKES, R_wc_of, auc, fsb_body, hamer_world, mask, seg_recall  # noqa: E402

BDIR = os.path.join(ROOT, "data", "hamer_b")
REP = os.path.join(ROOT, "reports", "hybrid")
VIT_HAND = {"L": slice(91, 112), "R": slice(112, 133)}               # COCO-WholeBody 133 点：左手 91–111、右手 112–132
YOLO_WRIST = {"L": 9, "R": 10}


def files(take):
    d = os.path.join(BDIR, take)
    one = lambda pat: (sorted(glob.glob(os.path.join(d, pat))) or [None])[0]  # noqa: E731
    return {"raw": one("*_hamer_raw.npz"), "pkl": one("*_hamer_data.pkl"), "i2": one("*_hamer_data_i2.pkl"),
            "others": one("*_others.npz")}


def raw_signals(R, side, F):
    """逐帧信号（越大越可疑）。R = hamer_raw.npz。"""
    h = SIDES[side]
    kp = R["vit_kp"][:F, VIT_HAND[side]]                               # (F, 21, 3)
    conf = kp[..., 2]
    src = R["box_src"][:F, h]
    box = R["hand_box"][:F, h]
    L = np.maximum(np.maximum(box[:, 2] - box[:, 0], box[:, 3] - box[:, 1]), 1.0)
    vis = conf > 0.5
    err = np.linalg.norm(R["kp2d"][:F, h] - kp[..., :2], axis=-1) / L[:, None]
    reproj = np.array([np.median(e[v]) if v.sum() >= 4 else np.nan for e, v in zip(err, vis)])
    reproj = np.where(np.isfinite(reproj), reproj, np.nanmax(reproj) if np.isfinite(reproj).any() else 1.0)
    T = R["tta_orient"][:F, h]                                         # (F, K, 3, 3)
    K = T.shape[1]
    tta = np.zeros(F)
    tta_f = np.zeros(F)
    if K > 1:
        d = np.stack([rot.geodesic_deg(T[:, k], T[:, 0]) for k in range(1, K)], 1)
        tta = d.max(1)
        P = R["tta_pose"][:F, h]
        tta_f = np.stack([rot.geodesic_deg(P[:, k], P[:, 0]).mean(1) for k in range(1, K)], 1).mean(1)
    has = src > 0
    return {"ViTPose手21点置信均值(低)": np.where(has, -conf.mean(1), 0.0),
            "ViTPose手>0.5点数(少)": -vis.sum(1).astype(float),
            "手框插值(本帧没检到手)": (src == 2).astype(float),
            "HaMeR重投影误差/手框": np.where(has, reproj, 0.0),
            "多裁剪手朝向不一致(°)": np.where(has, tta, 0.0),
            "多裁剪手指不一致(°)": np.where(has, tta_f, 0.0)}


def disagree(o, side, G, R_wc):
    valid = ~np.isclose(o, 0).all(axis=(1, 2))
    Hw = hamer_world(np.where(valid[:, None, None], o, np.eye(3)), side, R_wc)
    Fw = G[:, 20 if side == "L" else 21]
    rel = np.einsum("fji,fjk->fik", Fw, Hw)
    q = rot.mat_to_quat(rel[valid])
    q = q * np.sign(q @ q[len(q) // 2])[:, None]
    Rm = rot.quat_to_mat(q.mean(0) / np.linalg.norm(q.mean(0)))
    dis = np.where(valid, rot.geodesic_deg(rel, np.broadcast_to(Rm, rel.shape)), 0.0)
    sp = np.r_[0, rot.geodesic_deg(Hw[1:], Hw[:-1])]
    return dis, np.where(valid, sp, 0.0), Hw


def old_vs_locked(take, side, F, fs):
    """旧链路（Windows，detectron2 取第一个人、隔 2 帧）vs 锁主角（隔 2 帧 i2 / 每帧）：相机系手朝向差。"""
    cfg = TAKES[take]
    hi = SIDES[side]
    old = np.asarray(load(f"{cfg['ref']}/hamer/{cfg['hamer']}")["global_orient"], float)[:F, hi]
    new = np.asarray(load(fs["pkl"])["global_orient"], float)[:F, hi]
    i2 = np.asarray(load(fs["i2"])["global_orient"], float)[:F, hi] if fs["i2"] else None
    v = lambda o: ~np.isclose(o, 0).all(axis=(1, 2))  # noqa: E731
    out = {"old_valid": int(v(old).sum()), "new_valid": int(v(new).sum())}
    both = v(old) & v(new)
    d_on = np.where(both, rot.geodesic_deg(np.where(both[:, None, None], old, np.eye(3)), np.where(both[:, None, None], new, np.eye(3))), np.nan)
    out["old_vs_every"] = {"median": float(np.nanmedian(d_on)), "p95": float(np.nanpercentile(d_on, 95)), ">45deg": int((d_on > 45).sum())}
    if i2 is not None:
        b2 = v(old) & v(i2)
        d_oi = np.where(b2, rot.geodesic_deg(np.where(b2[:, None, None], old, np.eye(3)), np.where(b2[:, None, None], i2, np.eye(3))), np.nan)
        b3 = v(i2) & v(new)
        d_in = np.where(b3, rot.geodesic_deg(np.where(b3[:, None, None], i2, np.eye(3)), np.where(b3[:, None, None], new, np.eye(3))), np.nan)
        out["old_vs_i2"] = {"median": float(np.nanmedian(d_oi)), "p95": float(np.nanpercentile(d_oi, 95)), ">45deg": int((d_oi > 45).sum()),
                            "<5deg_pct": round(100 * float(np.nanmean(d_oi < 5)), 1)}
        # 差得大的帧是不是 HaMeR 自己就不稳（多裁剪不一致）：环境复现问题会均匀分布，HaMeR 模棱两可只集中在不稳的帧
        if fs["raw"]:
            Rr = np.load(fs["raw"])
            T = Rr["tta_orient"][:F, hi].astype(float)
            if T.shape[1] > 1:
                spread = np.stack([rot.geodesic_deg(T[:, k], T[:, 0]) for k in range(1, T.shape[1])], 1).max(1)
                big = np.nan_to_num(d_oi) > 30
                out["old_vs_i2_big_frames_tta_spread_median"] = float(np.median(spread[big])) if big.any() else None
                out["all_frames_tta_spread_median"] = float(np.median(spread[Rr["box_src"][:F, hi] > 0]))
        out["i2_vs_every"] = {"median": float(np.nanmedian(d_in)), "p95": float(np.nanpercentile(d_in, 95)), ">45deg": int((d_in > 45).sum())}
    # 拿错人：旧结果离"别人的同侧手"明显比离主角近（只有 999 跑了 others）
    if fs["others"]:
        O = np.load(fs["others"])
        sel = O["side"] == hi
        best = np.full(F, np.inf)
        best_tid = np.full(F, -1)
        for f, R_, tid in zip(O["frame"][sel], O["orient"][sel], O["tid"][sel]):
            if f < F and v(old[f:f + 1])[0]:
                d = float(rot.geodesic_deg(old[f][None], R_[None])[0])
                if d < best[f]:
                    best[f], best_tid[f] = d, tid
        raw_new = None
        if fs["raw"]:
            raw_new = np.load(fs["raw"])["orient"][:F, hi].astype(float)
        dm = np.where(v(old) & v(raw_new), rot.geodesic_deg(np.where(v(old)[:, None, None], old, np.eye(3)), np.where(v(raw_new)[:, None, None], raw_new, np.eye(3))), np.inf)
        wrong = (dm > 45) & (best < 25) & (best < dm / 2)
        segs = []
        for a, b in _runs(wrong):
            if b - a + 1 >= 2:
                segs.append([int(a), int(b), int(np.bincount(best_tid[a:b + 1][best_tid[a:b + 1] >= 0]).argmax()) if (best_tid[a:b + 1] >= 0).any() else -1])
        out["wrong_person_frames"] = int(wrong.sum())
        out["wrong_person_segments"] = segs
        out["others_frames_with_same_side_hand"] = int(np.isfinite(best).sum())
    return out


def _runs(m):
    out, t, F = [], 0, len(m)
    while t < F:
        if m[t]:
            s = t
            while t < F and m[t]:
                t += 1
            out.append((s, t - 1))
        else:
            t += 1
    return out


def main():
    os.makedirs(REP, exist_ok=True)
    lines, ovl = [], {}
    for take, cfg in TAKES.items():
        fs = files(take)
        if not fs["raw"]:
            print("缺", take)
            continue
        R = np.load(fs["raw"])
        G = fsb_body(take)
        F = min(G.shape[0], R["orient"].shape[0])
        R_wc = R_wc_of(take)
        yk = np.nan_to_num(np.load(f"{cfg['raw']}/main_track.npz")["kpts"], nan=0.0)
        lines.append(f"== {take}: F={F}；手框 检到/插值/无：" + "；".join(
            f"{s} {int((R['box_src'][:F, h] == 1).sum())}/{int((R['box_src'][:F, h] == 2).sum())}/{int((R['box_src'][:F, h] == 0).sum())}"
            for s, h in SIDES.items()))
        ovl[take] = {}
        for side, hi in SIDES.items():
            bad = mask(F, cfg["bad"][side])
            sig = raw_signals(R, side, F)
            o_new = R["orient"][:F, hi].astype(float)
            dis, sp, _ = disagree(o_new, side, G[:F], R_wc)
            sig["HaMeR-FSB分歧(新,未平滑)"] = dis
            sig["HaMeR角速度(新,未平滑)"] = sp
            sig["YOLO腕置信度(低)"] = -yk[:F, YOLO_WRIST[side], 2]
            o_old = np.asarray(load(f"{cfg['ref']}/hamer/{cfg['hamer']}")["global_orient"], float)[:F, hi]
            dis_old, sp_old, _ = disagree(o_old, side, G[:F], R_wc)
            sig["(对照)HaMeR-FSB分歧(旧)"] = dis_old
            for name, s in sig.items():
                thr = np.percentile(s, 95)
                rec = seg_recall(s, cfg["bad"][side], thr)
                fp = int(((s > thr) & ~mask(F, [(max(0, a - 2), b + 2) for a, b in cfg["bad"][side]])).sum())
                lines.append(f"  {side} {name:26s} AUC {auc(s, bad):.2f}  段召回@P95 {rec}/{len(cfg['bad'][side])}  段外超阈帧 {fp}")
            ovl[take][side] = old_vs_locked(take, side, F, fs)
            lines.append(f"  {side} 旧 vs 锁主角：{json.dumps(ovl[take][side], ensure_ascii=False)}")
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(REP, "signals_b.txt"), "w", encoding="utf-8") as fh:
        fh.write(txt + "\n")
    with open(os.path.join(REP, "old_vs_locked.json"), "w", encoding="utf-8") as fh:
        json.dump(ovl, fh, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
