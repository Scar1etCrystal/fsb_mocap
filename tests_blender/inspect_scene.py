"""只读：列出场景里的骨架、动作、NLA、关键骨骼约束和自定义属性（不保存）。"""
import bpy
sc = bpy.context.scene
print("SCENE", sc.name, "frames", sc.frame_start, sc.frame_end, "fps", sc.render.fps, "unit_scale", sc.unit_settings.scale_length)
for o in bpy.data.objects:
    if o.type != "ARMATURE":
        continue
    ad = o.animation_data
    act = ad.action.name if ad and ad.action else None
    nla = [(t.name, [(s.name, s.action.name if s.action else None, s.blend_type, round(s.influence, 3), s.mute) for s in t.strips], t.mute) for t in ad.nla_tracks] if ad else []
    print(f"\nARM {o.name!r} bones={len(o.data.bones)} scale={tuple(round(x,4) for x in o.scale)} loc={tuple(round(x,3) for x in o.location)} parent={o.parent.name if o.parent else None} hide={o.hide_get()} action={act}")
    print("   NLA", nla)
    print("   props", {k: (v if isinstance(v, (int, float, str)) else str(type(v))) for k, v in o.items()})
    names = [b.name for b in o.data.bones]
    print("   bones sample", names[:40])
    for b in o.pose.bones:
        if b.constraints:
            for c in b.constraints:
                tgt = getattr(c, "target", None); sub = getattr(c, "subtarget", "")
                if tgt is not None and tgt.type == "ARMATURE" and tgt != o:
                    print(f"   CON {b.name}: {c.type} -> {tgt.name}:{sub} infl={c.influence:.2f}" + (f" mix={getattr(c,'mix_mode','')}" if hasattr(c,'mix_mode') else ""))
for a in bpy.data.actions:
    fr = a.frame_range
    print("ACTION", repr(a.name), "range", tuple(fr), "fcurves", len(a.fcurves), "users", a.users)
for t in bpy.data.texts:
    print("TEXT", t.name, len(t.as_string()))
