"""阶段 A：FastSAM3DBody 身体 + HaMeR 手（用户现成的 hamer_data.pkl）+ 用户自己的 hand_body_merge.py（原件、插件默认参数）。

变体：
  H0  原样混合：只把身体换成 FSB
  H1  + 相机俯仰补偿：HaMeR 手朝向用 FSB 地面拟合出的相机→世界旋转 R_wc 转到世界系（合并脚本写死的是"相机水平"的 RZ180）
  H2  + 语义门控：HaMeR 与 FSB 手腕朝向分歧大、或 YOLO 手腕置信度低（遮挡）的帧标成"没看见"，
      用"敷衍"手势补：短缺口按手腕相对前臂的局部旋转插值，长缺口保持再在末尾过渡；手指同法
  H3  H2 + 角速度门控（HaMeR 自身朝向帧间突跳）
合并在与 HAMER 环境同版本的 Python（3.11 / numpy 2.4.4 / scipy 1.17.1）里跑：合并脚本对数值环境敏感（见报告）。
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
from merge_tools import HANDOFF_HAMER, STAGE1_HAMER, load, run_merge  # noqa: E402
from signals import M, RZ180, SIDES, TAKES, R_wc_of, mask  # noqa: E402

HAMERLIKE_PY = "/tmp/claude-1000/-home-sb-exp-lvm/e6b9d07f-50b6-475e-85da-fcb89c30441f/scratchpad/hamerlike_venv/bin/python"
OUT = os.path.join(ROOT, "out_hybrid")
VARIANTS = ("H0", "H1", "H2", "H3", "H1f", "H2f", "H4")       # f = 合并时开"前臂扭转"（--forearm_twist）；H4 见下
SHORT_GAP, BLEND = 15, 10                                             # 与合并脚本 --interp_missing_frames 15 同口径


def slerp(A, B, t):
    """矩阵 slerp：A·exp(t·log(Aᵀ B))。"""
    rel = rot.mat_to_rotvec(A.T @ B)
    return A @ rot.rotvec_to_mat(rel * t)


def runs(m):
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


def close_dilate(m, gap=3, dil=1):
    m = m.copy()
    r = runs(m)
    for (a0, b0), (a1, b1) in zip(r[:-1], r[1:]):
        if a1 - b0 - 1 <= gap:
            m[b0 + 1:a1] = True
    if dil:
        idx = np.nonzero(m)[0]
        for d in range(1, dil + 1):
            m[np.clip(idx - d, 0, len(m) - 1)] = True
            m[np.clip(idx + d, 0, len(m) - 1)] = True
    return m


def world_from_raw(o, side, R_wc):
    return np.einsum("ij,fjk,kl->fil", R_wc @ M, o, M) if side == "L" else np.einsum("ij,fjk->fik", R_wc, o)


def merge_world(o, side):
    """合并脚本把 hamer_data 里的朝向变成世界系手腕全局旋转的方式：右 RZ180·o，左 M·RZ180·o·M。"""
    return np.einsum("ij,fjk,kl->fil", M @ RZ180, o, M) if side == "L" else np.einsum("ij,fjk->fik", RZ180, o)


def raw_from_world(W, side):
    """合并脚本：右 world = RZ180·o，左 world = M·RZ180·o·M → 反解 o（输入已含俯仰补偿时用）。"""
    return np.einsum("ij,fjk,kl->fil", RZ180 @ M, W, M) if side == "L" else np.einsum("ij,fjk->fik", RZ180, W)


def gate_signals(Hw, Fw, valid, yconf):
    """返回 dict：各信号的"可疑"布尔掩码（阈值都按本视频分位数，零手调）。"""
    rel = np.einsum("fji,fjk->fik", Fw, Hw)
    q = rot.mat_to_quat(rel[valid])
    q = q * np.sign(q @ q[len(q) // 2])[:, None]
    Rm = rot.quat_to_mat(q.mean(0) / np.linalg.norm(q.mean(0)))
    dis = np.where(valid, rot.geodesic_deg(rel, np.broadcast_to(Rm, rel.shape)), 0.0)
    sp = rot.geodesic_deg(Hw[1:], Hw[:-1])
    spd = np.maximum(np.r_[0, sp], np.r_[sp, 0])                     # 突跳帧的进、出两侧都大
    # 遮挡：YOLO 腕点置信度相对本视频明显下陷（前臂整段被挡时实测从 0.99 掉到 0.5–0.6）；
    # 只用 5% 分位会在置信度普遍很高的视频里硬标 5% 的帧（0.5–0.7 其实看得见），所以再加绝对上限 0.6
    ylow = yconf < min(0.6, float(np.percentile(yconf, 5)))
    return {"disagree": valid & (dis > np.percentile(dis[valid], 95)),
            "occluded": ylow,
            "speed": valid & (spd > np.percentile(spd[valid], 95)),
            "_values": {"dis": dis, "spd": spd}}


def fill_gaps(o_raw, hp_raw, keep, side, R_wc, elbow_w):
    """keep=False 的帧用"敷衍"手势补：手腕相对前臂的局部旋转 + 手指，短缺口插值、长缺口保持后过渡、首尾保持。"""
    F = len(keep)
    o, hp = o_raw.copy(), hp_raw.copy()
    W = merge_world(o_raw, side)                                      # o_raw 已含俯仰补偿：合并后的世界朝向
    L = np.einsum("fji,fjk->fik", elbow_w, W)                         # 局部 = 肘全局ᵀ · 手全局
    good = np.nonzero(keep)[0]
    if not len(good):
        return o, hp, 0
    filled = 0
    for a, b in runs(~keep):
        prev = a - 1 if a > 0 and keep[a - 1] else None
        nxt = b + 1 if b + 1 < F and keep[b + 1] else None
        for t in range(a, b + 1):
            if prev is None and nxt is None:
                continue
            if prev is None or nxt is None:
                src = nxt if prev is None else prev
                Lt, ht = L[src], hp_raw[src]
            else:
                g = b - a + 1
                if g <= SHORT_GAP:
                    s = (t - prev) / (nxt - prev)
                else:
                    s = 0.0 if t < b + 1 - BLEND else (t - (b - BLEND)) / (BLEND + 1)
                Lt = slerp(L[prev], L[nxt], s)
                ht = np.stack([slerp(hp_raw[prev, j], hp_raw[nxt, j], s) for j in range(15)])
            Wt = elbow_w[t] @ Lt
            o[t] = raw_from_world(Wt[None], side)[0]
            hp[t] = ht
            filled += 1
    return o, hp, filled


TWIST_SHARE, TWIST_MAX = 0.55, 120.0                                 # 与合并脚本 forearm_twist 同比例；小臂分到的扭转封顶
AXIS = {"L": np.array([1.0, 0.0, 0.0]), "R": np.array([-1.0, 0.0, 0.0])}  # SMPL-X T 姿态小臂轴（同合并脚本）


def twist_deg(R, axis):
    """绕小臂轴的扭转角（swing-twist 分解），(-180, 180]。"""
    q = rot.mat_to_quat(R)                                           # (w, x, y, z)
    t = np.degrees(2 * np.arctan2(q[:, 1:] @ axis, q[:, 0]))
    return (t + 180) % 360 - 180


TWIST_BOUND = float(os.environ.get("HYB_TWIST_BOUND", "9999"))  # 超过即判异常（交门控）；默认关：FSB 上臂旋转偏时正常的掌心朝上也会超界（t1499 实测误报）
TWIST_RATE = 25.0                                                     # 分给小臂的扭转每帧最多变化（°），HaMeR 一帧翻手时兜底
TWIST_FREE = 150.0                                                    # 生理范围内不罚


def unwrap_deg(t, bound=TWIST_BOUND, free=TWIST_FREE, w_range=0.05):
    """全局最优的扭转连续化（Viterbi）：每帧在 τ−360 / τ / τ+360 三种表示里选，
    代价 = 帧间变化² + w_range·超出生理范围²，一次性选出整段最平滑、又尽量在 ±free 内的一串。
    贪心逐帧 unwrap 会被大弯曲时扭转分解的噪声带偏并一路累积（t1499 左手实测攒到 417°）；这里没有累积。
    返回 (连续扭转, 超出 ±bound 的帧掩码)。"""
    t = np.asarray(t, float)
    F = len(t)
    if not F:
        return t.copy(), np.zeros(0, bool)
    ks = np.array([-360.0, 0.0, 360.0])
    cand = t[:, None] + ks[None]                                     # (F, 3)
    unary = w_range * np.maximum(0.0, np.abs(cand) - free) ** 2
    cost = unary[0].copy()
    back = np.zeros((F, 3), int)
    for i in range(1, F):
        trans = (cand[i][None, :] - cand[i - 1][:, None]) ** 2       # (prev, cur)
        tot = cost[:, None] + trans
        back[i] = np.argmin(tot, axis=0)
        cost = tot[back[i], np.arange(3)] + unary[i]
    k = int(np.argmin(cost))
    out = np.empty(F)
    for i in range(F - 1, -1, -1):
        out[i] = cand[i, k]
        k = back[i, k]
    return out, np.abs(out) > bound


def distribute_twist(o_raw, side, G_world, body_pose):
    """合并前把手腕扭转按比例（跨帧连续）分给小臂：改 FSB 的肘部局部旋转，合并脚本据此算出的手腕扭转就小了。
    合并脚本自带的 --forearm_twist 每帧单独算扭转角，跨过 ±180° 时会翻符号，一帧把小臂甩 160–200°（t999 左手 737/746–750 实测）。
    返回新的 body_pose 与统计。手的世界朝向不变（只转小臂自身轴，腕关节位置也不动）。"""
    ej = 18 if side == "L" else 19                                   # SMPL-X 肘关节
    E = G_world[:, ej]
    H = merge_world(o_raw, side)
    W = np.einsum("fji,fjk->fik", E, H)                              # 手腕局部（相对小臂）
    tau, _ = unwrap_deg(twist_deg(W, AXIS[side]))
    target = np.clip(TWIST_SHARE * tau, -TWIST_MAX, TWIST_MAX)
    phi = target.copy()
    for i in range(1, len(phi)):                                     # 限速：HaMeR 一帧把手翻 ~180° 时小臂不跟着甩，残余留给手腕
        phi[i] = phi[i - 1] + np.clip(target[i] - phi[i - 1], -TWIST_RATE, TWIST_RATE)
    bp = body_pose.copy().reshape(-1, 21, 3)
    Rel = rot.rotvec_to_mat(bp[:, ej - 1])
    T = rot.rotvec_to_mat(np.radians(phi)[:, None] * AXIS[side][None])
    bp[:, ej - 1] = rot.mat_to_rotvec(np.einsum("fij,fjk->fik", Rel, T))
    return bp.reshape(-1, 63), {"tau_p95_abs": float(np.percentile(np.abs(tau), 95)), "tau_range": [float(tau.min()), float(tau.max())],
                                "phi_range": [float(phi.min()), float(phi.max())], "phi_rate_limited_frames": int((np.abs(phi - target) > 1e-6).sum()),
                                "tau_jump_max": float(np.abs(np.diff(tau)).max())}


def build(take, variant):
    cfg = TAKES[take]
    h = load(f"{cfg['ref']}/hamer/{cfg['hamer']}")
    person_p = f"{cfg['fsb']}/gvhmr/hmr4d_results.pt_person-0.pkl"
    person = load(person_p)["smpl_params_global"]
    G = rot.chain_global_rotations(np.asarray(person["global_orient"], float), np.asarray(person["body_pose"], float))
    F = G.shape[0]
    base = variant.rstrip("f")
    R_wc = R_wc_of(take) if base != "H0" else RZ180
    yk = np.nan_to_num(np.load(f"{cfg['raw']}/main_track.npz")["kpts"], nan=0.0)
    out = {k: np.array(v, copy=True) for k, v in h.items()}
    meta = {"take": take, "variant": variant, "R_wc_vs_level_deg": float(np.degrees(np.arccos(np.clip(
        (np.trace(RZ180.T @ R_wc) - 1) / 2, -1, 1)))), "gated": {}, "filled": {}}
    for side, hi in SIDES.items():
        o = np.asarray(h["global_orient"], np.float64)[:, hi].copy()
        hp = np.asarray(h["hand_pose"], np.float64)[:, hi].copy()
        n = min(len(o), F)
        valid = ~np.isclose(o, 0).all(axis=(1, 2))
        if base != "H0":                                                 # 俯仰补偿：合并后的世界朝向 = R_wc·(相机系手朝向)
            C = RZ180 @ M @ R_wc @ M if side == "L" else RZ180 @ R_wc
            o[valid] = np.einsum("ij,fjk->fik", C, o[valid])
        if base in ("H2", "H3", "H4"):
            Hw = world_from_raw(np.where(valid[:, None, None], h["global_orient"][:, hi], np.eye(3))[:n], side, R_wc)
            Fw = G[:n, 20 if side == "L" else 21]
            sg = gate_signals(Hw, Fw, valid[:n], yk[:n, 9 if side == "L" else 10, 2])
            bad = sg["disagree"] | sg["occluded"]
            if base in ("H3", "H4"):
                bad |= sg["speed"]
            if base == "H4":                                         # "暴力一点"：手腕相对小臂的扭转一帧变化 >30° = 翻转帧
                Wl = np.einsum("fji,fjk->fik", G[:n, 18 if side == "L" else 19], Hw)
                tu, implausible = unwrap_deg(twist_deg(Wl, AXIS[side]))
                tj = np.abs(np.diff(tu))
                thr = max(40.0, float(np.percentile(tj[valid[1:n] & valid[:n - 1]], 98)))   # 本视频自适应：快舞手转得快
                jump = (np.r_[False, tj > thr] | np.r_[tj > thr, False]) | implausible
                bad |= valid[:n] & jump
                meta.setdefault("twist_jump_thr", {})[side] = round(thr, 1)
                meta.setdefault("twist_jump_frames", {})[side] = int(jump.sum())
                bad = close_dilate(bad, gap=6, dil=2)
            else:
                bad = close_dilate(bad)
            keep = valid[:n] & ~bad
            o_new, hp_new, filled = fill_gaps(o[:n], hp[:n], keep, side, R_wc, G[:n, 18 if side == "L" else 19])
            o[:n], hp[:n] = o_new, hp_new
            meta["gated"][side] = [[int(a), int(b)] for a, b in runs(bad)]
            meta["filled"][side] = {"frames": int(filled), "pct": round(100 * filled / n, 1),
                                    "by_signal_pct": {k: round(100 * float(v.mean()), 1) for k, v in sg.items() if not k.startswith("_")}}
        out["global_orient"][:, hi] = o.astype(np.float32)
        out["hand_pose"][:, hi] = hp.astype(np.float32)
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
    merged = os.path.join(d, "hamer", "merged.pkl")
    merge_body = os.path.join(d, "gvhmr", "hmr4d_results.pt_person-0.pkl")
    if base == "H4":                                                    # 自己做跨帧连续的小臂扭转分配，合并时不开 --forearm_twist
        pp = load(person_p)
        bp = np.asarray(pp["smpl_params_global"]["body_pose"], np.float64)
        for side, hi in SIDES.items():
            o_side = np.asarray(out["global_orient"], np.float64)[:F, hi]
            bp, st = distribute_twist(o_side, side, G, bp)
            meta.setdefault("twist", {})[side] = st
        for br in ("smpl_params_global", "smpl_params_incam"):
            pp[br]["body_pose"] = bp.astype(np.float32)
        merge_body = os.path.join(d, "hamer", "_merge_body_twisted.pkl")
        with open(merge_body, "wb") as fh:
            pickle.dump(pp, fh, protocol=4)
    log = run_merge(merge_body, hp_path, merged, fps=30.0,
                    script_dir=STAGE1_HAMER if "hand_keypoint_conf" in h else HANDOFF_HAMER, python=HAMERLIKE_PY,
                    extra=("--forearm_twist",) if variant.endswith("f") else ())
    m = load(merged)
    m.setdefault("hamer_merge_info", {})["fsb_hybrid"] = {k: v for k, v in meta.items() if k in ("variant", "R_wc_vs_level_deg")}
    with open(merged, "wb") as fh:                                       # 协议 4：Blender 自带 numpy 1.26 不靠垫片也能读
        pickle.dump(m, fh, protocol=4)
    top = os.path.join(d, os.path.basename(d) + ".pkl")
    shutil.copyfile(merged, top)
    meta["merge_log_tail"] = [l for l in log.splitlines() if "[HandBodyMerge]" in l][-8:]
    with open(os.path.join(d, "hybrid_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=1)
    rep = contract.validate_take(d)
    return d, rep, meta


if __name__ == "__main__":
    want = sys.argv[1:] or [f"{t}_{v}" for t in TAKES for v in VARIANTS]
    for tv in want:
        take, variant = tv.rsplit("_", 1)
        d, rep, meta = build(take, variant)
        print(f"{tv}: 合同 {'OK' if rep['ok'] else 'FAIL ' + '; '.join(rep['errors'][:2])}  俯仰差 {meta['R_wc_vs_level_deg']:.2f}°  "
              f"门控 {json.dumps(meta['filled'], ensure_ascii=False)}", flush=True)
