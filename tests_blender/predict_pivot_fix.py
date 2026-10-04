"""只读推演（不改文件、不保存）：把骨盆旋转的支点从 torso_root（膝高）挪到 hips 骨头（两髋上方 ~0.09 m，与源一致）后，
躯干整体多出一个平移 c(t) = (I − D(t))·d，D = torso_root 世界旋转相对静止的变化，d = hips 头 − torso_root 头（静止）。
脚是 IK 目标（世界位置不变），所以预测：髋/肩/头 = 现位置 + c(t)；再查髋到同侧踝目标的距离是否超过腿长。"""
import json

import bpy
import numpy as np
from mathutils import Matrix

A, B = 151, 277
src = bpy.data.objects["Armature"]
rig = bpy.data.objects["RIG-arue式重音テトver 2.01_arm"]
sc = bpy.context.scene
MW = rig.matrix_world
b_root, b_hips = rig.data.bones["torso_root"], rig.data.bones["hips"]
R0 = (MW @ b_root.matrix_local).to_3x3().normalized()
d = np.array((MW @ b_hips.head_local) - (MW @ b_root.head_local))
leg = {s: rig.data.bones[f"ORG-thigh.{s}"].length + rig.data.bones[f"ORG-shin.{s}"].length for s in "LR"}
rows = []
for f in range(1, 1000):
    sc.frame_set(f)
    pb = rig.pose.bones["torso_root"]
    Rw = (MW @ pb.matrix).to_3x3().normalized()
    D = np.array(Rw @ R0.inverted())
    c = (np.eye(3) - D) @ d
    g = lambda o, n: np.array(o.matrix_world @ o.pose.bones[n].head)
    rows.append({"c": c, "hl": g(rig, "ORG-thigh.L"), "hr": g(rig, "ORG-thigh.R"), "al": g(rig, "ORG-foot.L"), "ar": g(rig, "ORG-foot.R"),
                 "shl": g(rig, "ORG-upper_arm.L"), "shr": g(rig, "ORG-upper_arm.R"), "head": g(rig, "ORG-spine.006"),
                 "s_hl": g(src, "f_avg_L_Hip"), "s_hr": g(src, "f_avg_R_Hip"), "s_shl": g(src, "f_avg_L_Shoulder"),
                 "s_shr": g(src, "f_avg_R_Shoulder"), "s_head": g(src, "f_avg_Head")})
R = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
seg = slice(A - 1, B)
lat = (R["hr"] - R["hl"])[seg].mean(0); lat[2] = 0; lat /= np.linalg.norm(lat)
slat = (R["s_hr"] - R["s_hl"])[seg].mean(0); slat[2] = 0; slat /= np.linalg.norm(slat)
pp = lambda x: float(np.percentile(x[seg], 97.5) - np.percentile(x[seg], 2.5))
hip, sho, head = (R["hl"] + R["hr"]) / 2, (R["shl"] + R["shr"]) / 2, R["head"]
s_hip, s_sho = (R["s_hl"] + R["s_hr"]) / 2, (R["s_shl"] + R["s_shr"]) / 2
out = {"源": {"髋中横移": pp(s_hip @ slat), "肩中横移": pp(s_sho @ slat), "髋相对肩": pp((s_hip - s_sho) @ slat), "头横移": pp(R["s_head"] @ slat)},
       "Teto 现状": {"髋中横移": pp(hip @ lat), "肩中横移": pp(sho @ lat), "髋相对肩": pp((hip - sho) @ lat), "头横移": pp(head @ lat)},
       "Teto 改支点后(预测)": {"髋中横移": pp((hip + R["c"]) @ lat), "肩中横移": pp((sho + R["c"]) @ lat),
                          "髋相对肩": pp((hip - sho) @ lat), "头横移": pp((head + R["c"]) @ lat)}}
# 与源的时序相关
for k, (t, s) in {"髋中横移": ((hip + R["c"]) @ lat, s_hip @ slat), "现状髋中横移": (hip @ lat, s_hip @ slat)}.items():
    out.setdefault("与源时序相关", {})[k] = round(float(np.corrcoef(t[seg], s[seg])[0, 1]), 3)
# IK 可达性：髋关节到同侧踝目标的距离 / 腿长（全片）
reach = {}
for s, hk, ak in (("L", "hl", "al"), ("R", "hr", "ar")):
    now = np.linalg.norm(R[hk] - R[ak], axis=1) / leg[s]
    new = np.linalg.norm(R[hk] + R["c"] - R[ak], axis=1) / leg[s]
    reach[s] = {"现状_max": round(float(now.max()), 3), "改后_max": round(float(new.max()), 3),
                "改后_超过0.995的帧": int((new > 0.995).sum()), "现状_超过0.995的帧": int((now > 0.995).sum())}
out["髋到踝距离/腿长"] = reach
out["支点偏移d(m)"] = [round(float(x), 3) for x in d]
out["修正平移c 全片幅度(m)"] = round(float(np.linalg.norm(R["c"], axis=1).max()), 3)
print("RESULT " + json.dumps(out, ensure_ascii=False))
