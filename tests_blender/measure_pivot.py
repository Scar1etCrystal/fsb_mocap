"""只读：骨盆旋转支点相对髋关节的位置；Teto 髋中横移 = 支点平移 + 绕支点旋转 的拆分。"""
import json

import bpy
import numpy as np

A, B = 151, 277
src = bpy.data.objects["Armature"]
rig = bpy.data.objects["RIG-arue式重音テトver 2.01_arm"]
sc = bpy.context.scene


def W(obj, n):
    return np.array(obj.matrix_world @ obj.pose.bones[n].head)


rec = []
for f in range(1, 1000):
    sc.frame_set(f)
    rec.append({"s_piv": W(src, "f_avg_Pelvis"), "s_hl": W(src, "f_avg_L_Hip"), "s_hr": W(src, "f_avg_R_Hip"),
                "t_root": W(rig, "torso_root"), "t_torso": W(rig, "torso"), "t_hips": W(rig, "hips"),
                "t_org": W(rig, "ORG-spine"), "t_hl": W(rig, "ORG-thigh.L"), "t_hr": W(rig, "ORG-thigh.R")})
R = {k: np.stack([r[k] for r in rec]) for k in rec[0]}
seg = slice(A - 1, B)
out = {}
for side, piv_keys, hl, hr in (("源", ["s_piv"], "s_hl", "s_hr"), ("Teto", ["t_root", "t_torso", "t_hips", "t_org"], "t_hl", "t_hr")):
    hip = (R[hl] + R[hr]) / 2
    lat = (R[hr] - R[hl])[seg].mean(0); lat[2] = 0; lat /= np.linalg.norm(lat)
    for pk in piv_keys:
        off = (R[pk] - hip)
        out[f"{side}:{pk}"] = {"支点高出两髋中点_m": round(float(off[seg][:, 2].mean()), 3),
                                "支点横移_pp": round(float(np.ptp((R[pk] @ lat)[seg])), 3),
                                "两髋中点横移_pp": round(float(np.ptp((hip @ lat)[seg])), 3),
                                "两髋相对支点横移_pp(旋转贡献)": round(float(np.ptp(((hip - R[pk]) @ lat)[seg])), 3),
                                "支点横移与旋转贡献的相关": round(float(np.corrcoef((R[pk] @ lat)[seg], ((hip - R[pk]) @ lat)[seg])[0, 1]), 3)}
print("RESULT " + json.dumps(out, ensure_ascii=False))
