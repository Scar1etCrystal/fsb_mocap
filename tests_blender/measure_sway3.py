"""只读测量（续二）：躯干侧倾、脊柱各段长度、竖直方向缩放、151–277 双脚轨迹。"""
import json

import bpy
import numpy as np

A, B = 151, 277
src = bpy.data.objects["Armature"]
rig = bpy.data.objects["RIG-arue式重音テトver 2.01_arm"]
SCH = ["f_avg_Pelvis", "f_avg_Spine1", "f_avg_Spine2", "f_avg_Spine3", "f_avg_Neck", "f_avg_Head"]
TCH = ["ORG-spine", "ORG-spine.001", "ORG-spine.003", "ORG-spine.004", "ORG-spine.006"]
sc = bpy.context.scene
data = {"s": [], "t": []}
for f in range(1, 1000):
    sc.frame_set(f)
    for key, obj, ch, extra in (("s", src, SCH, ["f_avg_L_Hip", "f_avg_R_Hip", "f_avg_L_Ankle", "f_avg_R_Ankle"]),
                                ("t", rig, TCH, ["ORG-thigh.L", "ORG-thigh.R", "ORG-foot.L", "ORG-foot.R"])):
        M = obj.matrix_world
        data[key].append([list(M @ obj.pose.bones[n].head) for n in ch + extra])
out = {}
for key in ("s", "t"):
    P = np.array(data[key])                                           # (F, 链+4, 3)
    n = len(SCH) if key == "s" else len(TCH)
    iL, iR, aL, aR = n, n + 1, n + 2, n + 3
    ineck = n - 2
    seg = slice(A - 1, B)
    lat = (P[:, iR] - P[:, iL])[seg].mean(0); lat[2] = 0; lat /= np.linalg.norm(lat)
    hip = (P[:, iL] + P[:, iR]) / 2
    neck = P[:, ineck]
    tv = neck - hip
    lean = np.degrees(np.arctan2(tv @ lat, tv[:, 2]))
    seglen = [float(np.linalg.norm(P[:, i + 1] - P[:, i], axis=1).mean()) for i in range(n - 1)]
    ank = (P[:, aL] + P[:, aR]) / 2
    def pp(x):
        x = x[seg]; return float(np.percentile(x, 97.5) - np.percentile(x, 2.5))
    out[key] = {"躯干侧倾(髋中→颈)_pp_deg": pp(lean), "脊柱各段长(盆起，逐段到头)": [round(x, 3) for x in seglen],
                "髋高(相对踝)_均值": float((hip - ank)[seg][:, 2].mean()),
                "左踝横向轨迹(每10帧)": [round(float(x), 3) for x in (P[:, aL] @ lat)[seg][::10]],
                "右踝横向轨迹(每10帧)": [round(float(x), 3) for x in (P[:, aR] @ lat)[seg][::10]],
                "髋中横向轨迹(每10帧)": [round(float(x), 3) for x in (hip @ lat)[seg][::10]],
                "_pelvis": P[:, 0], "_ank": P[:, aL]}
for comp, i in (("水平", slice(0, 2)), ("竖直", slice(2, 3))):
    for name, k in (("骨盆", "_pelvis"), ("左踝", "_ank")):
        ps, pt = out["s"][k][:, i], out["t"][k][:, i]
        ds, dt = ps - ps.mean(0), pt - pt.mean(0)
        out.setdefault("位移缩放", {})[f"{name}{comp}"] = round(float((ds * dt).sum() / (ds * ds).sum()), 3)
for key in ("s", "t"):
    out[key].pop("_pelvis"); out[key].pop("_ank")
print("RESULT " + json.dumps(out, ensure_ascii=False))
