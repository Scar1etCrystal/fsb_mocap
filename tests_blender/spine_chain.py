import bpy, numpy as np
rig = bpy.data.objects["RIG-arue式重音テトver 2.01_arm"]
for n in ["torso_root", "torso", "hips", "chest", "spine_fk", "spine_fk.001", "spine_fk.002", "spine_fk.003", "spine_fk.004", "spine_fk.005",
          "tweak_spine", "ORG-spine", "ORG-spine.001", "ORG-spine.003", "ORG-spine.004", "neck", "head", "MCH-spine", "MCH-spine.001"]:
    b = rig.data.bones.get(n)
    if b:
        pb = rig.pose.bones[n]
        print(f"BONE {n:14s} head_z={b.head_local.z:.3f} tail_z={b.tail_local.z:.3f} parent={b.parent.name if b.parent else None} rot={pb.rotation_mode}")
props = {k: rig.pose.bones["torso"].get(k) for k in ("neck_follow", "head_follow", "IK_FK", "fk_ik", "IK/FK")}
print("TORSO PROPS", {k: v for k, v in props.items() if v is not None}, dict(rig.pose.bones["torso"].items()) if rig.pose.bones["torso"].keys() else {})
for n in ["spine_fk", "spine_fk.001", "spine_fk.002", "spine_fk.003", "chest", "hips", "torso", "torso_root"]:
    pb = rig.pose.bones.get(n)
    if pb:
        fc = [f.data_path for f in rig.animation_data.action.fcurves if f'"{n}"' in f.data_path]
        print("ANIM", n, sorted(set(p.split('.')[-1] for p in fc)))
for pb in rig.pose.bones:
    if pb.name.startswith(("ORG-spine", "DEF-spine", "MCH-spine")):
        for c in pb.constraints:
            print("CON", pb.name, c.type, getattr(c, "subtarget", ""), round(c.influence, 2))
