"""阶段 A 评估：旧链路（GVHMR+HaMeR）/ FSB 自带手 / 混合 H0–H3，在用户标注坏段内外的手腕、手指抖动。"""
from __future__ import annotations

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
from signals import TAKES, mask  # noqa: E402

OUT = os.path.join(ROOT, "out_hybrid")
REP = os.path.join(ROOT, "reports", "hybrid")
SEG_NAMES = {"t999": {"R": ["A 532–556", "B1 688–696", "D1 786–791"], "L": ["用户确认 87–127", "B2 687–701", "C+ 733–779（含 E~771）", "D1 786–791", "D2 800–817"]},
             "t1499": {"R": ["遮挡 512–554"], "L": ["遮挡 884–898"]}}
WRIST = {"L": slice(57, 60), "R": slice(60, 63)}
HAND = {"L": "left_hand_pose", "R": "right_hand_pose"}
TWISTED = ("H4", "B4", "B5", "B6", "B7")                                         # 合并前改了肘部（小臂扭转）的变体


def variants(take):
    """阶段 A 的全部变体 + 已构建的阶段 B 变体（B4 / B5）。"""
    vs = ["H0", "H1", "H2", "H3", "H1f", "H2f", "H4"]
    return vs + [v for v in ("B4", "B5", "B6", "B7") if os.path.isfile(f"{OUT}/{take}_{v}/hamer/merged.pkl")]


def series(pkl):
    g = load(pkl)["smpl_params_global"]
    bp = np.asarray(g["body_pose"], float)
    Gall = rot.chain_global_rotations(np.asarray(g["global_orient"], float), bp)   # 世界系：手腕全局 = 手的世界朝向
    out = {"Gw": {"L": Gall[:, 20], "R": Gall[:, 21]}}
    for s in ("L", "R"):
        W = rot.rotvec_to_mat(bp[:, WRIST[s]])
        H = rot.rotvec_to_mat(np.asarray(g[HAND[s]], float).reshape(-1, 15, 3))
        E = rot.rotvec_to_mat(bp[:, {"L": slice(51, 54), "R": slice(54, 57)}[s]])
        out[s] = {"W": W, "H": H, "wsp": np.r_[0, rot.geodesic_deg(W[1:], W[:-1])], "esp": np.r_[0, rot.geodesic_deg(E[1:], E[:-1])],
                  "fsp": np.r_[0, rot.geodesic_deg(H[1:], H[:-1]).mean(1)], "bend": rot.geodesic_deg(W, np.eye(3)[None])}
    out["body"] = bp
    return out


def main():
    os.makedirs(REP, exist_ok=True)
    summary = {}
    md = ["# 阶段 A：FastSAM3DBody 身体 + HaMeR 手（混合）评估", "",
          "手腕 = 合并后 body_pose 的手腕局部旋转（相对前臂）；抖动阈值 T = 旧链路该手手腕角速度的 P97（同一阈值比较所有版本）。",
          "坏段 = 用户标注（999：A–E 问题全集；1499：两处遮挡），pkl 帧号。", ""]
    for take, cfg in TAKES.items():
        vers = {"旧(GVHMR+HaMeR)": f"{cfg['ref']}/hamer/merged.pkl", "FSB自带手": f"{cfg['fsb']}/hamer/merged.pkl"}
        VS = variants(take)
        for v in VS:
            vers[f"混合{v}"] = f"{OUT}/{take}_{v}/hamer/merged.pkl"
        S = {k: series(p) for k, p in vers.items()}
        F = min(len(S[k]["body"]) for k in S)
        summary[take] = {}
        md += [f"## {take}", ""]
        # 身体（除手腕）应与 FSB 自带手版本逐位相同
        fsb_body = S["FSB自带手"]["body"][:F].reshape(F, 21, 3)
        for v in VS:
            hb = S[f"混合{v}"]["body"][:F].reshape(F, 21, 3)
            others = [j for j in range(21) if j not in ((17, 18, 19, 20) if (v.endswith("f") or v in TWISTED) else (19, 20))]
            d = float(np.abs(hb[:, others] - fsb_body[:, others]).max())
            summary[take][f"body_nonwrist_maxdiff_{v}"] = d
        md.append(f"- 混合版除手腕外的身体与 FSB 版逐位相同：最大差 "
                  + "，".join(f"{v} {summary[take][f'body_nonwrist_maxdiff_{v}']:.1e}" for v in VS)
                + "（f 版和 H4/B 不比肘部：小臂扭转本来就改肘）")
        old = S["旧(GVHMR+HaMeR)"]
        h0 = S["混合H0"]
        for s in ("L", "R"):
            fd = rot.geodesic_deg(h0[s]["H"][:F], old[s]["H"][:F]).mean(1)
            md.append(f"- {s} 手指：混合 H0 vs 旧 每帧平均差 中位 {np.median(fd):.2f}°、P95 {np.percentile(fd, 95):.2f}°"
                      "（同一份 HaMeR、同一合并脚本 → 应≈0）")
        md.append("")
        for s in ("L", "R"):
            bad = mask(F, cfg["bad"][s])
            T = float(np.percentile(old[s]["wsp"][:F], 97))
            md += [f"### {take} {'左' if s == 'L' else '右'}手（T = {T:.1f}°/帧）", "",
                   "| 版本 | 手腕角速度 P50 / P95 | 超 T 帧：坏段内 / 坏段外 | 坏段内手腕最大角速度 | 手腕弯曲 P95 | 手指角速度 P95 | 手的世界朝向 vs 旧：均值 / P95 | 小臂 >60°/帧 帧数 |",
                   "|---|---|---|---|---|---|---|---|"]
            for k, X in S.items():
                w, f = X[s]["wsp"][:F], X[s]["fsp"][:F]
                dd = rot.geodesic_deg(X["Gw"][s][:F], old["Gw"][s][:F])         # 手腕局部角随前臂变，世界朝向才是"还是不是那只手"
                row = {"wsp_p50": float(np.median(w)), "wsp_p95": float(np.percentile(w, 95)),
                       "over_T_in": int((w[bad] > T).sum()), "over_T_out": int((w[~bad] > T).sum()),
                       "max_in": float(w[bad].max()), "bend_p95": float(np.percentile(X[s]["bend"][:F], 95)),
                       "fsp_p95": float(np.percentile(f, 95)), "vs_old_mean": float(dd.mean()), "vs_old_p95": float(np.percentile(dd, 95)),
                       "elbow_spikes": int((X[s]["esp"][:F] > 60).sum())}
                summary[take].setdefault(s, {})[k] = row
                md.append(f"| {k} | {row['wsp_p50']:.1f} / {row['wsp_p95']:.1f} | {row['over_T_in']} / {row['over_T_out']} | "
                          f"{row['max_in']:.1f} | {row['bend_p95']:.0f}° | {row['fsp_p95']:.1f} | {row['vs_old_mean']:.1f} / {row['vs_old_p95']:.1f} | {row['elbow_spikes']} |")
            md += ["", f"逐段（手腕最大角速度 °/帧，旧 → {' / '.join(VS)}）：", ""]
            for (a, b), nm in zip(cfg["bad"][s], SEG_NAMES[take][s]):
                vals = [S[k][s]["wsp"][a:b + 1].max() for k in ["旧(GVHMR+HaMeR)"] + [f"混合{v}" for v in VS]]
                md.append(f"- {nm}：{vals[0]:.1f} → " + " / ".join(f"{x:.1f}" for x in vals[1:]))
            md.append("")
        for v in [v for v in VS if v in ("H2", "H3", "H4", "B4", "B5", "B6", "B7")]:
            meta = json.load(open(f"{OUT}/{take}_{v}/hybrid_meta.json", encoding="utf-8"))
            md.append(f"- {v} 门控补帧：左 {meta['filled']['L']['pct']}%、右 {meta['filled']['R']['pct']}%；区间（左）"
                      f"{meta['gated']['L'][:12]}{' …' if len(meta['gated']['L']) > 12 else ''}")
        md.append("")
    with open(os.path.join(REP, "evaluate.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")
    with open(os.path.join(REP, "evaluate.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1)
    print("\n".join(md))


if __name__ == "__main__":
    main()
