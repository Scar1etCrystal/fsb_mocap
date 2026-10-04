"""各版本最终手的世界朝向 vs HaMeR 原始输出（只看 HaMeR 有效帧；门控版去掉被门控的帧）。衡量合并后的手还剩多少"HaMeR 原样"。"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
from fsb import rot  # noqa: E402
from merge_tools import load  # noqa: E402
from signals import RZ180, SIDES, TAKES, R_wc_of  # noqa: E402
from build_hybrid import world_from_raw  # noqa: E402

lines = ["| 视频 手 | 旧 | H0 | H1 | H1f | H2f（非门控帧） |", "|---|---|---|---|---|---|"]
for take, cfg in TAKES.items():
    h = load(f"{cfg['ref']}/hamer/{cfg['hamer']}")
    Rwc = R_wc_of(take)
    vers = {"旧": (f"{cfg['ref']}/hamer/merged.pkl", RZ180), "H0": (f"{ROOT}/out_hybrid/{take}_H0/hamer/merged.pkl", RZ180)}
    for v in ("H1", "H1f", "H2f"):
        vers[v] = (f"{ROOT}/out_hybrid/{take}_{v}/hamer/merged.pkl", Rwc)
    for side, hi in SIDES.items():
        o = np.asarray(h["global_orient"], float)[:, hi]
        valid = ~np.isclose(o, 0).all(axis=(1, 2))
        cells = []
        for k, (p, Rw) in vers.items():
            g = load(p)["smpl_params_global"]
            F = len(g["body_pose"])
            G = rot.chain_global_rotations(np.asarray(g["global_orient"], float), np.asarray(g["body_pose"], float))[:, 20 if side == "L" else 21]
            raw = world_from_raw(np.where(valid[:F, None, None], o[:F], np.eye(3)), side, Rw)
            m = valid[:F].copy()
            if k == "H2f":
                for a, b in json.load(open(f"{ROOT}/out_hybrid/{take}_H2f/hybrid_meta.json"))["gated"][side]:
                    m[a:b + 1] = False
            d = rot.geodesic_deg(G[m], raw[m])
            cells.append(f"{np.median(d):.1f} / {np.percentile(d, 95):.1f}")
        lines.append(f"| {take} {'左' if side == 'L' else '右'} | " + " | ".join(cells) + " |")
print("手的世界朝向 vs HaMeR 原始输出（中位 / P95，度）\n")
print("\n".join(lines))
