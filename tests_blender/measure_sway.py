"""只读测量：源骨架（SMPL-X f_avg）vs Teto 控制骨架，151–277 帧胯部左右摆动。"""
import json
import sys

import bpy
import numpy as np

A, B = 151, 277
src = bpy.data.objects["Armature"]
rig = bpy.data.objects["RIG-arue式重音テトver 2.01_arm"]
SB = {"pelvis": "f_avg_Pelvis", "hipL": "f_avg_L_Hip", "hipR": "f_avg_R_Hip", "kneeL": "f_avg_L_Knee", "kneeR": "f_avg_R_Knee",
      "ankL": "f_avg_L_Ankle", "ankR": "f_avg_R_Ankle", "footL": "f_avg_L_Foot", "footR": "f_avg_R_Foot", "neck": "f_avg_Neck"}
TB = {"pelvis": "ORG-spine", "hipL": "ORG-thigh.L", "hipR": "ORG-thigh.R", "kneeL": "ORG-shin.L", "kneeR": "ORG-shin.R",
      "ankL": "ORG-foot.L", "ankR": "ORG-foot.R", "footL": "ORG-toe.L", "footR": "ORG-toe.R", "neck": "ORG-spine.004",
      "torso": "torso", "torso_root": "torso_root"}


def heads(obj, names):
    M = obj.matrix_world
    return {k: np.array(M @ obj.pose.bones[n].head) for k, n in names.items() if n in obj.pose.bones}


rows = []
sc = bpy.context.scene
for f in range(1, 1000):
    sc.frame_set(f)
    rows.append({"s": heads(src, SB), "t": heads(rig, TB)})
print("MISSING src", [k for k in SB if k not in rows[0]["s"]], "tgt", [k for k in TB if k not in rows[0]["t"]])
res = {}
for side in ("s", "t"):
    P = {k: np.stack([r[side][k] for r in rows]) for k in rows[0][side]}
    thigh = np.linalg.norm(P["kneeL"] - P["hipL"], axis=1).mean()
    shin = np.linalg.norm(P["ankL"] - P["kneeL"], axis=1).mean()
    hip_w = np.linalg.norm(P["hipL"] - P["hipR"], axis=1).mean()
    trunk = np.linalg.norm(P["neck"] - P["pelvis"], axis=1).mean()
    seg = slice(A - 1, B)                                            # 帧 f 在 rows[f-1]
    lat = (P["hipR"] - P["hipL"])[seg].mean(0); lat[2] = 0; lat /= np.linalg.norm(lat)
    mid_hip = (P["hipL"] + P["hipR"]) / 2
    mid_ank = (P["ankL"] + P["ankR"]) / 2
    rel = mid_hip - mid_ank                                         # 两髋中点相对两踝中点
    d_lat = rel[seg] @ lat
    feet_move = max(np.ptp(P["ankL"][seg][:, :2], axis=0).max(), np.ptp(P["ankR"][seg][:, :2], axis=0).max())
    def knee_angle(h, k, a):
        u, v = h - k, a - k
        return np.degrees(np.arccos(np.clip(np.einsum("fi,fi->f", u, v) / np.linalg.norm(u, axis=1) / np.linalg.norm(v, axis=1), -1, 1)))
    kL = 180 - knee_angle(P["hipL"], P["kneeL"], P["ankL"])
    kR = 180 - knee_angle(P["hipR"], P["kneeR"], P["ankR"])
    leg = thigh + shin
    amp = float(np.percentile(d_lat, 97.5) - np.percentile(d_lat, 2.5))
    res[side] = {"thigh": thigh, "shin": shin, "leg": leg, "hip_width": hip_w, "trunk": trunk,
                 "sway_pp_m": amp, "sway_std_m": float(d_lat.std()),
                 "sway_angle_pp_deg": float(np.degrees(np.arcsin(np.clip(amp / 2 / leg, -1, 1))) * 2),
                 "feet_move_m": float(feet_move), "hip_height_rel_ankles_mean": float(rel[seg][:, 2].mean()),
                 "knee_flex_mean_seg": float((kL[seg].mean() + kR[seg].mean()) / 2),
                 "knee_flex_mean_all": float((kL.mean() + kR.mean()) / 2), "knee_flex_p95_all": float(np.percentile(np.r_[kL, kR], 95)),
                 "pelvis_xy": P["pelvis"][:, :2].tolist()}
# 整片骨盆水平位移的比例（ARP 根位移缩放）
ps, pt = np.array(res["s"]["pelvis_xy"]), np.array(res["t"]["pelvis_xy"])
ds, dt = ps - ps.mean(0), pt - pt.mean(0)
scale = float((ds * dt).sum() / (ds * ds).sum())
for side in ("s", "t"):
    res[side].pop("pelvis_xy")
res["root_motion_scale_fit"] = scale
res["root_motion_fit_resid_m"] = float(np.sqrt(((dt - scale * ds) ** 2).sum(1).mean()))
print("RESULT " + json.dumps(res, ensure_ascii=False))
