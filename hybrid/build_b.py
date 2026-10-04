"""阶段 B：FSB 身体 + 锁主角重跑的 HaMeR 手（cloud/hamer/hamer_locked.py），合并仍用用户原件、插件默认参数。

变体（都 = 阶段 A 的 H4 配方：俯仰补偿 + 门控补帧 + 合并前跨帧连续的小臂扭转分配，合并不开 --forearm_twist）：
  B4  手 = 锁主角、每帧检测的 HaMeR，平滑方式同原版（savgol 11，先平滑再门控）——只换 HaMeR 输入，配方不动
  B5  手 = 同一次运行的逐帧原始结果：先门控（H4 的信号 + 新信号）→ 补帧 → 再 savgol 平滑，坏帧不会被平滑摊到邻帧
  B6  B5 + 多裁剪投票：每帧 7 个裁剪（原版裁剪 + 缩放/平移）里取"中位"那个结果（与其余结果角度差之和最小），
      HaMeR 在模棱两可的帧上裁剪挪几个像素就翻手，少数派被投掉
  B7  = B4 + 只加专抓遮挡的三个新信号（绝对阈值，干净的视频几乎不多砍）：本帧没检到这只手（手框是插值来的）、
      主角 ViTPose 手部 21 点置信度均值 < 0.5、HaMeR 重投影误差（5 帧中位）> 半个手框。
      B5/B6 的"每个新信号都按本视频 P95"会把干净视频也硬砍 5%×信号数（实测门控 35–44%），所以不推荐
新信号（B5）：本帧没检到这只手（手框是插值来的）、主角 ViTPose 手部关键点置信度、HaMeR 重投影误差、多裁剪不一致。
"""
from __future__ import annotations

import json
import os
import pickle
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from fsb import contract, rot  # noqa: E402
from merge_tools import HANDOFF_HAMER, load, run_merge  # noqa: E402
from signals import M, RZ180, SIDES, TAKES, R_wc_of  # noqa: E402
from build_hybrid import (AXIS, HAMERLIKE_PY, OUT, close_dilate, distribute_twist, fill_gaps, gate_signals,  # noqa: E402
                          runs, twist_deg, unwrap_deg, world_from_raw)
from signals_b import files, raw_signals  # noqa: E402

VARIANTS = ("B4", "B5", "B6", "B7")
REPROJ_THR, REPROJ_WIN, CONF_THR = 0.5, 5, 0.5                    # B7：重投影误差 / 手框（5 帧中位）、ViTPose 手部置信度
# B5 新信号的阈值（按本视频分位数 + 绝对下限，零手调；依据见 reports/hybrid/signals_b.txt）
NEW_Q = float(os.environ.get("HYB_NEW_Q", "95"))


def savgol_rot(Rm, window=11, poly=3):
    """原版 hamer_video.smooth_rotations 的矩阵分支：四元数同号化 → savgol → 归一化。Rm: (F, J, 3, 3)。"""
    from scipy.signal import savgol_filter
    F, J = Rm.shape[:2]
    out = np.empty_like(Rm)
    for j in range(J):
        q = rot.mat_to_quat(Rm[:, j])                                   # (w, x, y, z)；同号化 + savgol 与分量顺序无关
        for i in range(1, F):
            if q[i] @ q[i - 1] < 0:
                q[i] = -q[i]
        s = savgol_filter(q, window, poly, axis=0)
        s /= np.linalg.norm(s, axis=1, keepdims=True) + 1e-8
        out[:, j] = rot.quat_to_mat(s)
    return out


def tta_medoid(R, hi, n):
    """每帧多裁剪结果里的中位样本（手朝向两两角度差之和最小）；手指取同一个裁剪的结果，保持一致。"""
    T = R["tta_orient"][:n, hi].astype(np.float64)                     # (F, K, 3, 3)
    P = R["tta_pose"][:n, hi].astype(np.float64)
    K = T.shape[1]
    D = np.zeros((n, K, K))
    for a in range(K):
        for b in range(a + 1, K):
            D[:, a, b] = D[:, b, a] = rot.geodesic_deg(T[:, a], T[:, b])
    m = D.sum(2).argmin(1)
    idx = np.arange(n)
    spread = np.median(D[idx, m], axis=1)                                # 到中位样本的角度差中位数（含自身 0）
    return T[idx, m], P[idx, m], m, spread


def new_bad(R, side, F, valid, tta_spread=None):
    """B5/B6 的新信号门控（越大越可疑的信号取本视频分位数阈值）。
    tta_spread：B6 传"各裁剪到中位样本的角度差中位数"（没有多数派 = 真不确定）；B5 用"其余裁剪与原版裁剪的最大差"。"""
    sig = raw_signals(R, side, F)
    if tta_spread is not None:
        sig["多裁剪手朝向不一致(°)"] = np.where(R["box_src"][:F, SIDES[side]] > 0, tta_spread[:F], 0.0)
    out = {"no_hand": valid & (R["box_src"][:F, SIDES[side]] == 2)}
    for key, name in (("vit_low", "ViTPose手21点置信均值(低)"), ("reproj", "HaMeR重投影误差/手框"), ("tta", "多裁剪手朝向不一致(°)")):
        s = sig[name]
        out[key] = valid & (s > np.percentile(s[valid], NEW_Q))
    return out


def occlusion_bad(R, side, F, valid):
    """B7 的遮挡信号：本帧没检到手 | 主角手部 ViTPose 置信度均值 < 0.5 | 重投影误差 5 帧中位 > 半个手框。"""
    from scipy.ndimage import median_filter
    sig = raw_signals(R, side, F)
    hi = SIDES[side]
    rep5 = median_filter(sig["HaMeR重投影误差/手框"], size=REPROJ_WIN, mode="nearest")
    return {"no_hand": valid & (R["box_src"][:F, hi] == 2),
            "vit_low": valid & (-sig["ViTPose手21点置信均值(低)"] < CONF_THR) & (R["box_src"][:F, hi] > 0),
            "reproj": valid & (rep5 > REPROJ_THR)}


def build(take, variant):
    cfg = TAKES[take]
    fs = files(take)
    R = np.load(fs["raw"])
    hp_pkl = load(fs["pkl"])
    person_p = f"{cfg['fsb']}/gvhmr/hmr4d_results.pt_person-0.pkl"
    person = load(person_p)["smpl_params_global"]
    G = rot.chain_global_rotations(np.asarray(person["global_orient"], float), np.asarray(person["body_pose"], float))
    F = G.shape[0]
    R_wc = R_wc_of(take)
    yk = np.nan_to_num(np.load(f"{cfg['raw']}/main_track.npz")["kpts"], nan=0.0)
    out = {k: np.array(v, copy=True) for k, v in hp_pkl.items()}
    meta = {"take": take, "variant": variant, "hamer": os.path.basename(fs["pkl"]), "gated": {}, "filled": {}}
    for side, hi in SIDES.items():
        if variant in ("B4", "B7"):
            o = np.asarray(hp_pkl["global_orient"], np.float64)[:, hi].copy()
            hp = np.asarray(hp_pkl["hand_pose"], np.float64)[:, hi].copy()
        else:                                                              # B5/B6：逐帧原始结果（没检到手的帧也有插值手框的结果）
            o = np.zeros_like(np.asarray(hp_pkl["global_orient"], np.float64)[:, hi])
            hp = np.zeros_like(np.asarray(hp_pkl["hand_pose"], np.float64)[:, hi])
            n0 = min(len(o), R["orient"].shape[0])
            has = R["box_src"][:n0, hi] > 0
            if variant == "B6":
                To, Tp, mi, spread = tta_medoid(R, hi, n0)
                o[:n0][has], hp[:n0][has] = To[has], Tp[has]
                meta.setdefault("tta_medoid_not_base_pct", {})[side] = round(100 * float((mi[has] != 0).mean()), 1)
            else:
                o[:n0][has] = R["orient"][:n0, hi][has]
                hp[:n0][has] = R["pose"][:n0, hi][has]
        n = min(len(o), F)
        valid = ~np.isclose(o, 0).all(axis=(1, 2))
        C = RZ180 @ M @ R_wc @ M if side == "L" else RZ180 @ R_wc        # 俯仰补偿（同 H1+）
        o_cam = o.copy()
        o[valid] = np.einsum("ij,fjk->fik", C, o[valid])
        Hw = world_from_raw(np.where(valid[:, None, None], o_cam, np.eye(3))[:n], side, R_wc)
        Fw = G[:n, 20 if side == "L" else 21]
        sg = gate_signals(Hw, Fw, valid[:n], yk[:n, 9 if side == "L" else 10, 2])
        bad = sg["disagree"] | sg["occluded"] | sg["speed"]
        Wl = np.einsum("fji,fjk->fik", G[:n, 18 if side == "L" else 19], Hw)
        tu, implausible = unwrap_deg(twist_deg(Wl, AXIS[side]))
        tj = np.abs(np.diff(tu))
        thr = max(40.0, float(np.percentile(tj[valid[1:n] & valid[:n - 1]], 98)))
        jump = (np.r_[False, tj > thr] | np.r_[tj > thr, False]) | implausible
        bad |= valid[:n] & jump
        by = {k: v for k, v in sg.items() if not k.startswith("_")}
        by["twist_flip"] = valid[:n] & jump
        if variant == "B7":
            nb = occlusion_bad(R, side, n, valid[:n])
            for k, v in nb.items():
                bad |= v
                by[k] = v
        if variant in ("B5", "B6"):
            nb = new_bad(R, side, n, valid[:n], tta_spread=spread if variant == "B6" else None)
            for k, v in nb.items():
                bad |= v
                by[k] = v
        bad = close_dilate(bad, gap=6, dil=2)
        keep = valid[:n] & ~bad
        o_new, hp_new, filled = fill_gaps(o[:n], hp[:n], keep, side, R_wc, G[:n, 18 if side == "L" else 19])
        if variant in ("B5", "B6"):                                        # 门控、补帧之后再平滑（原版是先平滑）
            span = np.nonzero(~np.isclose(o_new, 0).all(axis=(1, 2)))[0]
            if len(span) >= 11:
                a, b = span[0], span[-1] + 1
                o_new[a:b] = savgol_rot(o_new[a:b, None])[:, 0]
                hp_new[a:b] = savgol_rot(hp_new[a:b])
        o[:n], hp[:n] = o_new, hp_new
        meta["gated"][side] = [[int(a), int(b)] for a, b in runs(bad)]
        meta["filled"][side] = {"frames": int(filled), "pct": round(100 * filled / n, 1),
                                "by_signal_pct": {k: round(100 * float(v.mean()), 1) for k, v in by.items()}}
        meta.setdefault("twist_jump_thr", {})[side] = round(thr, 1)
        out["global_orient"][:, hi] = o.astype(np.float32)
        out["hand_pose"][:, hi] = hp.astype(np.float32)
        if variant in ("B5", "B6"):
            out["global_orient_valid"][:, hi] = (~np.isclose(o, 0).all(axis=(1, 2))).astype(np.uint8)
            out["global_orient_abs_lr"][:, hi] = (o if side == "R" else np.einsum("ij,fjk,kl->fil", M, o, M)).astype(np.float32)
    d = os.path.join(OUT, f"{take}_{variant}")
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(os.path.join(d, "gvhmr")); os.makedirs(os.path.join(d, "hamer"))
    shutil.copy2(person_p, os.path.join(d, "gvhmr", "hmr4d_results.pt_person-0.pkl"))
    vid = f"{cfg['ref']}/gvhmr/0_input_video.mp4"
    if os.path.isfile(vid):
        shutil.copy2(vid, os.path.join(d, "gvhmr", "0_input_video.mp4"))
    hp_path = os.path.join(d, "hamer", cfg["hamer"])
    with open(hp_path, "wb") as fh:
        pickle.dump(out, fh, protocol=4)
    pp = load(person_p)
    bp = np.asarray(pp["smpl_params_global"]["body_pose"], np.float64)
    for side, hi in SIDES.items():
        bp, st = distribute_twist(np.asarray(out["global_orient"], np.float64)[:F, hi], side, G, bp)
        meta.setdefault("twist", {})[side] = st
    for br in ("smpl_params_global", "smpl_params_incam"):
        pp[br]["body_pose"] = bp.astype(np.float32)
    merge_body = os.path.join(d, "hamer", "_merge_body_twisted.pkl")
    with open(merge_body, "wb") as fh:
        pickle.dump(pp, fh, protocol=4)
    merged = os.path.join(d, "hamer", "merged.pkl")
    log = run_merge(merge_body, hp_path, merged, fps=30.0, script_dir=HANDOFF_HAMER, python=HAMERLIKE_PY)
    m = load(merged)
    m.setdefault("hamer_merge_info", {})["fsb_hybrid"] = {"variant": variant, "hamer": "locked main subject, every frame"}
    with open(merged, "wb") as fh:
        pickle.dump(m, fh, protocol=4)
    shutil.copyfile(merged, os.path.join(d, os.path.basename(d) + ".pkl"))
    meta["merge_log_tail"] = [l for l in log.splitlines() if "[HandBodyMerge]" in l][-8:]
    with open(os.path.join(d, "hybrid_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=1)
    return d, contract.validate_take(d), meta


if __name__ == "__main__":
    want = sys.argv[1:] or [f"{t}_{v}" for t in TAKES for v in VARIANTS]
    for tv in want:
        take, variant = tv.rsplit("_", 1)
        d, rep, meta = build(take, variant)
        print(f"{tv}: 合同 {'OK' if rep['ok'] else 'FAIL ' + '; '.join(rep['errors'][:2])}  门控 "
              f"{json.dumps({s: meta['filled'][s]['pct'] for s in meta['filled']}, ensure_ascii=False)}", flush=True)
