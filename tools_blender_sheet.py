"""Blender（后台）：关键帧对照图——视频帧 | 旧（GVHMR+HaMeR）| 新（FastSAM3DBody）。世界系网格从原相机方向看。"""
import os
import sys

import bpy
import numpy as np
from mathutils import Matrix, Vector

args = sys.argv[sys.argv.index("--") + 1:]
mesh_npz, frames_dir, out_prefix = args[0], args[1], args[2]
data = np.load(mesh_npz)
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)
sc = bpy.context.scene
sc.render.engine = "BLENDER_WORKBENCH"
sc.display.shading.light = "STUDIO"; sc.display.shading.color_type = "OBJECT"; sc.display.shading.show_cavity = True
S = 540
sc.render.resolution_x = sc.render.resolution_y = S
sc.render.image_settings.file_format = "PNG"
cam_data = bpy.data.cameras.new("cam"); cam = bpy.data.objects.new("cam", cam_data); sc.collection.objects.link(cam); sc.camera = cam
cam_data.type = "ORTHO"; cam_data.ortho_scale = 2.3
# 原相机在世界 z = −D、看 +z；Blender 坐标 (x, −z, y) → 相机在 +Y、看 −Y，画面右 = −X（与视频同向）
cam.matrix_world = Matrix.Translation(Vector((0, 8, 0))) @ Matrix.Rotation(np.pi / 2, 4, "X") @ Matrix.Rotation(np.pi, 4, "Y")


def render_mesh(verts, color, path):
    for o in [o for o in bpy.data.objects if o.type == "MESH"]:
        bpy.data.objects.remove(o, do_unlink=True)
    v = verts.astype(np.float64)
    v = v - (v.max(0) + v.min(0)) / 2
    vb = np.stack([v[:, 0], -v[:, 2], v[:, 1]], 1)
    m = bpy.data.meshes.new("m"); m.from_pydata(vb.tolist(), [], data["faces"].tolist()); m.update()
    o = bpy.data.objects.new("m", m); sc.collection.objects.link(o); o.color = (*color, 1.0)
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)


def pixels(path, w, h):
    img = bpy.data.images.load(path)
    if img.size[0] != w or img.size[1] != h:
        img.scale(w, h)
    a = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, 4)
    bpy.data.images.remove(img)
    return a


rows = []
tmp = out_prefix + "_tmp.png"
for f in data["frames"]:
    f = int(f)
    panels = [pixels(os.path.join(frames_dir, f"{f:05d}.jpg"), 960, S)]
    for name, color in (("old", (0.55, 0.62, 0.75)), ("new", (0.95, 0.6, 0.25))):
        render_mesh(data[f"{name}_{f}"], color, tmp)
        panels.append(pixels(tmp, S, S))
    rows.append(np.concatenate(panels, axis=1))
    print("row", f)
os.remove(tmp)
for k in range(0, len(rows), 5):
    block = rows[k:k + 5][::-1]                     # Blender 像素自下而上
    img = np.concatenate(block, axis=0)
    h, w = img.shape[:2]
    out = bpy.data.images.new("sheet", w, h)
    out.pixels = img.ravel().tolist()
    out.filepath_raw = f"{out_prefix}_{k // 5 + 1}.png"; out.file_format = "PNG"; out.save()
    print("SAVED", out.filepath_raw)
