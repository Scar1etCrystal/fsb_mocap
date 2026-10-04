"""只读测量（续）：151–277 帧胯部摆动的几种口径 + 各关节位移缩放。"""
import json

import bpy
import numpy as np

A, B = 151, 277
src = bpy.data.objects["Armature"]
rig = bpy.data.objects["RIG-arue式重音テトver 2.01_arm"]
SB = {"pelvis": "f_avg_Pelvis", "hipL": "f_avg_L_Hip", "hipR": "f_avg_R_Hip", "ankL": "f_avg_L_Ankle", "ankR": "f_avg_R_Ankle",
      "neck": "f_avg_Neck", "shL": "f_avg_L_Shoulder", "shR": "f_avg_R_Shoulder", "head": "f_avg_Head", "sp3": "f_avg_Spine3"}
TB = {"pelvis": "ORG-spine", "hipL": "ORG-thigh.L", "hipR": "ORG-thigh.R", "ankL": "ORG-foot.L", "ankR": "ORG-foot.R",
      "neck": "ORG-spine.004", "shL": "ORG-upper_arm.L", "shR": "ORG-upper_arm.R", "head": "ORG-spine.006", "sp3": "ORG-spine.003"}


def grab(obj, names):
    M = obj.matrix_world
    return {k: np.array(M @ obj.pose.bones[n].head) for k, n in names.items()}


sc = bpy.context.scene
S, T = [], []
for f in range(1, 1000):
    sc.frame_set(f)
    S.append(grab(src, SB)); T.append(grab(rig, TB))
out = {}
seg = slice(A - 1, B)
for name, R in (("源", S), ("Teto", T)):
    P = {k: np.stack([r[k] for r in R]) for k in R[0]}
    lat = (P["hipR"] - P["hipL"])[seg].mean(0); lat[2] = 0; lat /= np.linalg.norm(lat)
    up = np.array([0, 0, 1.0])
    hip = (P["hipL"] + P["hipR"]) / 2
    sho = (P["shL"] + P["shR"]) / 2
    ank = (P["ankL"] + P["ankR"]) / 2
    v = P["hipR"] - P["hipL"]
    roll = np.degrees(np.arctan2(v @ up, np.linalg.norm(v[:, :2], axis=1)))     # 骨盆侧倾（右髋高为正）
    vs = P["shR"] - P["shL"]
    sroll = np.degrees(np.arctan2(vs @ up, np.linalg.norm(vs[:, :2], axis=1)))  # 肩线侧倾
    def pp(x):
        x = x[seg]; return float(np.percentile(x, 97.5) - np.percentile(x, 2.5))
    out[name] = {"胯横移_相对双踝_pp": pp((hip - ank) @ lat), "胯横移_相对肩_pp": pp((hip - sho) @ lat),
                 "肩横移_相对双踝_pp": pp((sho - ank) @ lat), "头横移_相对双踝_pp": pp((P["head"] - ank) @ lat),
                 "骨盆侧倾_pp_deg": pp(roll), "肩线侧倾_pp_deg": pp(sroll),
                 "骨盆侧倾_相对肩线_pp_deg": pp(roll - sroll),
                 "_series": {"hip_ank": ((hip - ank) @ lat)[seg].tolist(), "hip_sho": ((hip - sho) @ lat)[seg].tolist(), "roll": roll[seg].tolist()}}
    out[name]["_P"] = {k: P[k] for k in ("pelvis", "ankL", "ankR", "neck", "head")}
# 时序相关
for k in ("hip_ank", "hip_sho", "roll"):
    a, b = np.array(out["源"]["_series"][k]), np.array(out["Teto"]["_series"][k])
    out.setdefault("相关", {})[k] = float(np.corrcoef(a, b)[0, 1])
# 各关节整片水平位移缩放
sc_ = {}
for k in ("pelvis", "ankL", "ankR", "neck", "head"):
    ps, pt = out["源"]["_P"][k][:, :2], out["Teto"]["_P"][k][:, :2]
    ds, dt = ps - ps.mean(0), pt - pt.mean(0)
    sc_[k] = round(float((ds * dt).sum() / (ds * ds).sum()), 3)
out["位移缩放"] = sc_
for n in ("源", "Teto"):
    out[n].pop("_series"); out[n].pop("_P")
print("RESULT " + json.dumps(out, ensure_ascii=False))
