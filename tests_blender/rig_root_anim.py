import bpy
rig = bpy.data.objects["RIG-arue式重音テトver 2.01_arm"]
act = rig.animation_data.action
for n in ("root", "torso_root", "torso", "hips"):
    fc = sorted({(f.data_path.split('.')[-1], f.array_index) for f in act.fcurves if f'pose.bones["{n}"]' in f.data_path})
    b = rig.data.bones[n]
    print("BONE", n, "parent", b.parent.name if b.parent else None, "anim", fc[:12], "inherit_rot", b.use_inherit_rotation, "local_loc", b.use_local_location)
f = [f for f in act.fcurves if f.data_path == 'pose.bones["torso_root"].location'][0]
print("KEYS", len(f.keyframe_points), f.keyframe_points[0].co[:], f.keyframe_points[-1].co[:], f.keyframe_points[0].interpolation)
