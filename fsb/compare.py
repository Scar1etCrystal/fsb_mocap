"""R3 新旧对比：同一段视频，旧链路（GVHMR+HaMeR）与新链路（FastSAM3DBody）的 take 逐帧对比。

输出：每关节 / 每组的误差分布（均值 / P95 / 最大）、跳帧检查（相邻帧突变计数、单帧尖刺），
世界空间脚踝俯仰角（需求文档 §2.5 的脚部验证口径）、根轨迹，JSON + Markdown 报告。

注意：旧链路不是真值。"新旧差异"里有一部分是新模型更好（比如脚），所以报告按部位分组，
通过线（均值 ≤5°、P95 ≤15°、无单帧跳变）逐组给结论，最终以 Blender 里肉眼终审为准。

用法：python -m fsb.compare <旧 take 目录> <新 take 目录> --out <报告目录> [--offset auto|N]
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

from . import rot
from .contract import load_pkl, take_paths

GROUPS = {   # SMPL-X 关节号（1..21 = body_pose）
    "legs": (1, 2, 4, 5), "feet": (7, 8, 10, 11), "spine_head": (3, 6, 9, 12, 15),
    "arms": (13, 14, 16, 17, 18, 19), "wrists": (20, 21),
}
JUMP_DEG = {"body": 30.0, "fingers": 45.0}       # 相邻帧单关节转角上限（30fps 下 30°/帧 = 900°/s）
PASS = {"mean": 5.0, "p95": 15.0}
FOOT_FWD = np.array([0.0, 0.0, 1.0])              # SMPL-X 静止姿态：人朝 +Z


def _load(take_or_pkl, branch):
    if os.path.isdir(take_or_pkl):
        p = take_paths(take_or_pkl)
        path = p["merged"] if os.path.isfile(p["merged"]) else p["top"]
    else:
        path = take_or_pkl
    data = load_pkl(path)
    return data[branch], path


def _stats(x):
    x = np.asarray(x, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    if not x.size:
        return {"n": 0}
    return {"n": int(x.size), "mean": round(float(x.mean()), 2), "p50": round(float(np.percentile(x, 50)), 2),
            "p95": round(float(np.percentile(x, 95)), 2), "max": round(float(x.max()), 2)}


def _yaw(R):
    f = R @ FOOT_FWD
    return np.arctan2(f[..., 0], f[..., 2])


def _rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _mean_rot(R):
    """旋转的近似均值（四元数同号后平均）。"""
    q = rot.mat_to_quat(R)
    q = q * np.where(q @ q[0] < 0, -1.0, 1.0)[:, None]
    m = q.mean(0)
    return rot.quat_to_mat(m / np.linalg.norm(m))


def _jumps(R, thr):
    """R:(F,J,3,3) → (每帧最大相邻转角, 超阈帧, 单帧尖刺帧)。尖刺 = 出去又回来：
    d(t−1,t) > thr 且 d(t,t+1) > thr 且 d(t−1,t+1) < thr/2。"""
    d = rot.geodesic_deg(R[1:], R[:-1])                       # (F-1,J)
    dmax = np.concatenate([[0.0], d.max(1)])
    over = np.nonzero(dmax > thr)[0]
    spikes = []
    if R.shape[0] >= 3:
        d2 = rot.geodesic_deg(R[2:], R[:-2])                  # (F-2,J)
        sp = (d[:-1] > thr) & (d[1:] > thr) & (d2 < thr / 2)  # 对 t=1..F-2
        spikes = (np.nonzero(sp.any(1))[0] + 1).tolist()
    return dmax, over.tolist(), spikes


def _foot_pitch(G, j):
    """世界空间脚的俯仰角（度）：脚的前向（静止 +Z）与水平面的夹角，抬脚尖为正。"""
    f = G[:, j] @ FOOT_FWD
    return np.degrees(np.arctan2(f[:, 1], np.linalg.norm(f[:, [0, 2]], axis=1)))


def compare(old_take, new_take, *, branch="smpl_params_global", offset="auto", max_offset=5) -> dict:
    old, p_old = _load(old_take, branch)
    new, p_new = _load(new_take, branch)
    Fo, Fn = old["body_pose"].shape[0], new["body_pose"].shape[0]
    bo = rot.rotvec_to_mat(old["body_pose"].reshape(Fo, 21, 3))
    bn = rot.rotvec_to_mat(new["body_pose"].reshape(Fn, 21, 3))

    def body_err(k):                       # new[t+k] vs old[t]
        a0, b0 = max(0, -k), max(0, k)
        n = min(Fo - a0, Fn - b0)
        return rot.geodesic_deg(bo[a0:a0 + n], bn[b0:b0 + n]).mean() if n > 10 else np.inf
    if offset == "auto":
        cand = {k: body_err(k) for k in range(-max_offset, max_offset + 1)}
        k = min(cand, key=cand.get)
        offset_info = {"chosen": int(k), "mean_body_err_by_offset": {str(a): round(float(b), 2) for a, b in cand.items()}}
    else:
        k = int(offset)
        offset_info = {"chosen": k}
    a0, b0 = max(0, -k), max(0, k)
    F = min(Fo - a0, Fn - b0)
    so, sn = slice(a0, a0 + F), slice(b0, b0 + F)

    out = {"old": p_old, "new": p_new, "branch": branch, "frames": {"old": Fo, "new": Fn, "compared": F},
           "offset": offset_info}
    # ---- 身体局部旋转
    e_local = rot.geodesic_deg(bo[so], bn[sn])                                    # (F,21)
    out["body_local"] = {"all": _stats(e_local),
                         "groups": {g: _stats(e_local[:, [j - 1 for j in js]]) for g, js in GROUPS.items()},
                         "joints": {rot.SMPLX_JOINT_NAMES[j]: _stats(e_local[:, j - 1]) for j in range(1, 22)}}
    # 常值偏差（约定/零位不同的诊断）：每关节平均相对旋转的角度 + 去掉后的残差
    bias, resid = {}, {}
    for j in range(21):
        rel = np.swapaxes(bo[so, j], -1, -2) @ bn[sn, j]
        B = _mean_rot(rel)
        bias[rot.SMPLX_JOINT_NAMES[j + 1]] = round(float(rot.geodesic_deg(np.eye(3), B)), 2)
        resid[rot.SMPLX_JOINT_NAMES[j + 1]] = _stats(rot.geodesic_deg(bo[so, j] @ B, bn[sn, j]))["mean"]
    out["body_local"]["bias_deg"] = bias
    out["body_local"]["mean_after_bias_removal"] = resid

    # ---- 世界旋转（去掉常值朝向差后）+ 根 + 脚
    Go = rot.chain_global_rotations(old["global_orient"][so], old["body_pose"][so])
    Gn = rot.chain_global_rotations(new["global_orient"][sn], new["body_pose"][sn])
    dyaw = _yaw(Gn[:, 0]) - _yaw(Go[:, 0])
    yaw_off = float(np.arctan2(np.sin(dyaw).mean(), np.cos(dyaw).mean()))
    Gn_al = np.einsum("ij,fkjl->fkil", _rot_y(-yaw_off), Gn)
    e_glob = rot.geodesic_deg(Go, Gn_al)                                           # (F,22)
    out["world"] = {"yaw_offset_deg": round(float(np.degrees(yaw_off)), 2),
                    "root_orient": _stats(e_glob[:, 0]),
                    "segments": {g: _stats(e_glob[:, list(js)]) for g, js in GROUPS.items()}}
    pitch = {}
    for side, j in (("L_Ankle", 7), ("R_Ankle", 8), ("L_Foot", 10), ("R_Foot", 11)):
        po, pn = _foot_pitch(Go, j), _foot_pitch(Gn_al, j)
        pitch[side] = {"err": _stats(np.abs(po - pn)), "old_range": round(float(np.ptp(po)), 1),
                       "new_range": round(float(np.ptp(pn)), 1),
                       "old_activity_deg_per_frame": round(float(np.abs(np.diff(po)).mean()), 3),
                       "new_activity_deg_per_frame": round(float(np.abs(np.diff(pn)).mean()), 3)}
    out["foot_pitch_world"] = pitch
    to, tn = old["transl"][so].astype(np.float64), new["transl"][sn].astype(np.float64)
    tn_al = (tn - tn[0]) @ _rot_y(-yaw_off).T + to[0]
    dist = np.linalg.norm(to - tn_al, axis=1)
    path_o = np.linalg.norm(np.diff(to[:, [0, 2]], axis=0), axis=1).sum()
    path_n = np.linalg.norm(np.diff(tn_al[:, [0, 2]], axis=0), axis=1).sum()
    out["root_transl"] = {"dist_m": _stats(dist), "path_len_xz_m": {"old": round(float(path_o), 2), "new": round(float(path_n), 2)},
                          "y_corr": round(float(np.corrcoef(to[:, 1], tn[:, 1])[0, 1]), 3) if F > 2 else None,
                          "note": "新序列平移到首帧对齐并转掉常值朝向差后比较"}

    # ---- 手指（两边都检测到的帧）
    hands = {}
    for side in ("left", "right"):
        key = f"{side}_hand_pose"
        if key not in old or key not in new:
            continue
        ho = rot.rotvec_to_mat(old[key][so].reshape(F, 15, 3))
        hn = rot.rotvec_to_mat(new[key][sn].reshape(F, 15, 3))
        both = np.ones(F, bool)
        for d in (old, new):
            if f"{side}_hand_is_detected" in d:
                both &= np.asarray(d[f"{side}_hand_is_detected"])[so if d is old else sn].astype(bool)
        e = rot.geodesic_deg(ho[both], hn[both])
        rel = np.swapaxes(ho[both], -1, -2) @ hn[both]
        bias_j = [round(float(rot.geodesic_deg(np.eye(3), _mean_rot(rel[:, j]))), 1) for j in range(15)] if both.any() else []
        hands[side] = {"frames_both_detected": int(both.sum()), "fingers": _stats(e), "bias_deg_per_joint": bias_j}
    out["hands"] = hands

    # ---- 跳帧检查（各自独立）
    jumps = {}
    for name, d, sl in (("old", old, so), ("new", new, sn)):
        Gb = rot.chain_global_rotations(d["global_orient"][sl], d["body_pose"][sl])
        _, over, spikes = _jumps(Gb, JUMP_DEG["body"])
        j = {"body_over_thr_frames": len(over), "body_spike_frames": spikes[:50], "body_spikes": len(spikes)}
        for side in ("left", "right"):
            key = f"{side}_hand_pose"
            if key in d:
                Hm = rot.rotvec_to_mat(d[key][sl].reshape(F, 15, 3))
                _, ov, sp = _jumps(Hm, JUMP_DEG["fingers"])
                j[f"{side}_fingers_spikes"] = len(sp)
                j[f"{side}_fingers_over_thr_frames"] = len(ov)
        t = d["transl"][sl].astype(np.float64)
        v = np.linalg.norm(np.diff(t, axis=0), axis=1)
        j["root_step_m"] = _stats(v)
        j["root_teleports"] = int((v > 0.15).sum())                              # >0.15 m/帧 = 4.5 m/s
        jumps[name] = j
    out["jumps"] = jumps

    # ---- 通过线（建议值）
    verdict = {}
    for g in GROUPS:
        s = out["body_local"]["groups"][g]
        verdict[g] = {"mean_ok": s["mean"] <= PASS["mean"], "p95_ok": s["p95"] <= PASS["p95"]}
    for side, h in hands.items():
        s = h["fingers"]
        if s.get("n"):
            verdict[f"{side}_fingers"] = {"mean_ok": s["mean"] <= PASS["mean"], "p95_ok": s["p95"] <= PASS["p95"]}
    verdict["new_no_body_spikes"] = jumps["new"]["body_spikes"] == 0
    out["verdict"] = verdict
    return out


def to_markdown(r: dict) -> str:
    L = [f"# 新旧对比报告", "", f"- 旧：`{r['old']}`", f"- 新：`{r['new']}`",
         f"- 分支：{r['branch']}；帧数 旧 {r['frames']['old']} / 新 {r['frames']['new']} / 对比 {r['frames']['compared']}；"
         f"帧偏移 {r['offset']['chosen']}（新[t+k] 对 旧[t]）", "",
         "## 身体局部旋转误差（度）", "", "| 组 | 均值 | P50 | P95 | 最大 | 通过线 |", "|---|---|---|---|---|---|"]
    for g, s in r["body_local"]["groups"].items():
        v = r["verdict"][g]
        L.append(f"| {g} | {s['mean']} | {s['p50']} | {s['p95']} | {s['max']} | {'✓' if v['mean_ok'] and v['p95_ok'] else '✗'} |")
    s = r["body_local"]["all"]
    L += [f"| **全部** | {s['mean']} | {s['p50']} | {s['p95']} | {s['max']} | |", "",
          "## 世界空间（转掉常值朝向差 %.1f° 后）" % r["world"]["yaw_offset_deg"], "",
          "| 段 | 均值 | P95 | 最大 |", "|---|---|---|---|",
          f"| 根朝向 | {r['world']['root_orient']['mean']} | {r['world']['root_orient']['p95']} | {r['world']['root_orient']['max']} |"]
    for g, s in r["world"]["segments"].items():
        L.append(f"| {g} | {s['mean']} | {s['p95']} | {s['max']} |")
    L += ["", "## 脚：世界空间俯仰角（需求 §2.5 口径）", "",
          "| 关节 | 误差均值 | P95 | 旧 动作幅度 | 新 动作幅度 | 旧 活动量°/帧 | 新 活动量°/帧 |", "|---|---|---|---|---|---|---|"]
    for k, v in r["foot_pitch_world"].items():
        L.append(f"| {k} | {v['err']['mean']} | {v['err']['p95']} | {v['old_range']} | {v['new_range']} | "
                 f"{v['old_activity_deg_per_frame']} | {v['new_activity_deg_per_frame']} |")
    L += ["", "（活动量 = 俯仰角逐帧变化的平均值；新 > 旧 说明新链路捕到了更多脚部动作）", "", "## 手指（两边都检测到的帧）", "",
          "| 手 | 帧数 | 均值 | P95 | 最大 | 每关节常值偏差（度） |", "|---|---|---|---|---|---|"]
    for side, h in r["hands"].items():
        s = h["fingers"]
        L.append(f"| {side} | {h['frames_both_detected']} | {s.get('mean')} | {s.get('p95')} | {s.get('max')} | {h['bias_deg_per_joint']} |")
    t = r["root_transl"]
    L += ["", "## 根位移", "", f"- 对齐后逐帧距离：均值 {t['dist_m'].get('mean')} m，P95 {t['dist_m'].get('p95')} m，最大 {t['dist_m'].get('max')} m",
          f"- 水平路径长：旧 {t['path_len_xz_m']['old']} m / 新 {t['path_len_xz_m']['new']} m；高度相关 {t['y_corr']}", "",
          "## 跳帧检查（各自独立）", "", "| | 身体 >%g°/帧 帧数 | 身体单帧尖刺 | 左手指尖刺 | 右手指尖刺 | 根瞬移(>0.15m/帧) |" % JUMP_DEG["body"],
          "|---|---|---|---|---|---|"]
    for name in ("old", "new"):
        j = r["jumps"][name]
        L.append(f"| {name} | {j['body_over_thr_frames']} | {j['body_spikes']} | {j.get('left_fingers_spikes')} | "
                 f"{j.get('right_fingers_spikes')} | {j['root_teleports']} |")
    sp = r["jumps"]["new"]["body_spike_frames"]
    if sp:
        L.append(f"\n新链路身体尖刺帧（0 基，前 50 个）：{sp}")
    L += ["", "## 每关节（局部）误差均值 / 常值偏差 / 去偏差后均值", "", "| 关节 | 均值 | P95 | 常值偏差 | 去偏差后均值 |", "|---|---|---|---|---|"]
    for j, s in r["body_local"]["joints"].items():
        L.append(f"| {j} | {s['mean']} | {s['p95']} | {r['body_local']['bias_deg'][j]} | {r['body_local']['mean_after_bias_removal'][j]} |")
    L += ["", "说明：旧链路不是真值；常值偏差大而去偏差后均值小 = 零位/约定不同（可校正），不是动作不同。"]
    return "\n".join(L) + "\n"


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--out", required=True)
    ap.add_argument("--branch", default="smpl_params_global")
    ap.add_argument("--offset", default="auto")
    a = ap.parse_args(argv)
    r = compare(a.old, a.new, branch=a.branch, offset=a.offset)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "compare.json"), "w", encoding="utf-8") as fh:
        json.dump(r, fh, ensure_ascii=False, indent=1)
    with open(os.path.join(a.out, "compare.md"), "w", encoding="utf-8") as fh:
        fh.write(to_markdown(r))
    print(to_markdown(r))
    return 0


if __name__ == "__main__":
    sys.exit(main())
