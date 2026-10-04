"""Blender（后台）：手部对照图。每行一帧，4 列：视频（人物 + 左右手特写）| 旧 | FSB 自带手 | 混合 H2f。
模型列 = 全身（上）+ 左右手特写（下，镜头对准手腕关节，和视频裁剪同方向）。画面左右与视频一致：主角的左手在画面右侧。

    blender -b x.blend --python blender_hand_sheet.py -- <make_sheet_data 输出目录> <take> <输出前缀>
"""
import json
import os
import sys

import bpy
import numpy as np
from mathutils import Matrix, Vector

args = sys.argv[sys.argv.index("--") + 1:]
src, take, out_prefix = args[0], args[1], args[2]
data = np.load(os.path.join(src, f"{take}_meshes.npz"))
labels = json.load(open(os.path.join(src, f"{take}_labels.json"), encoding="utf-8"))
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)
sc = bpy.context.scene
sc.render.engine = "BLENDER_WORKBENCH"
sc.display.shading.light = "STUDIO"; sc.display.shading.color_type = "OBJECT"; sc.display.shading.show_cavity = True
sc.render.image_settings.file_format = "PNG"
cam_data = bpy.data.cameras.new("cam"); cam = bpy.data.objects.new("cam", cam_data); sc.collection.objects.link(cam); sc.camera = cam
cam_data.type = "ORTHO"
ROT = Matrix.Rotation(np.pi / 2, 4, "X") @ Matrix.Rotation(np.pi, 4, "Y")   # 看 −Y，画面右 = −X（与视频同向）
COLORS = {"old": (0.55, 0.62, 0.75), "fsb": (0.95, 0.6, 0.25), "H2f": (0.45, 0.78, 0.45)}
TMP = out_prefix + "_tmp.png"


def to_b(v):
    return np.stack([v[:, 0], -v[:, 2], v[:, 1]], 1)


def set_mesh(verts, color):
    for o in [o for o in bpy.data.objects if o.type == "MESH"]:
        bpy.data.objects.remove(o, do_unlink=True)
    m = bpy.data.meshes.new("m"); m.from_pydata(to_b(verts.astype(np.float64)).tolist(), [], data["faces"].tolist()); m.update()
    o = bpy.data.objects.new("m", m); sc.collection.objects.link(o); o.color = (*color, 1.0)


def shot(center_b, scale, w, h):
    sc.render.resolution_x, sc.render.resolution_y = w, h
    cam_data.ortho_scale = scale
    cam.matrix_world = Matrix.Translation(Vector((center_b[0], center_b[1] + 8, center_b[2]))) @ ROT
    sc.render.filepath = TMP
    bpy.ops.render.render(write_still=True)
    return pixels(TMP, w, h)


def pixels(path, w, h):
    img = bpy.data.images.load(path)
    if img.size[0] != w or img.size[1] != h:
        img.scale(w, h)
    a = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, 4)
    bpy.data.images.remove(img)
    return a


def model_panel(name, f):
    v = data[f"v_{name}_{f}"]; j = data[f"j_{name}_{f}"]
    set_mesh(v, COLORS[name])
    vb = to_b(v.astype(np.float64)); c = (vb.max(0) + vb.min(0)) / 2
    body = shot(c, 2.3, 540, 360)
    hands = [shot(to_b(((j[w] + j[m]) / 2)[None].astype(np.float64))[0], 0.30, 270, 180)
             for w, m in ((20, 28), (21, 43))]                           # 对准 手腕—中指根 中点（左 20/28，右 21/43）
    # 画面左右与视频一致：主角左手在画面右 → 下排按 [右手, 左手] 排，和视频特写同序
    bottom = np.concatenate([hands[1], hands[0]], axis=1)
    return np.concatenate([bottom, body], axis=0)                        # Blender 像素自下而上：先放下排


def video_panel(f):
    person = pixels(os.path.join(src, f"{take}_{f:05d}_person.png"), 540, 360)
    hl = pixels(os.path.join(src, f"{take}_{f:05d}_handL.png"), 270, 180)
    hr = pixels(os.path.join(src, f"{take}_{f:05d}_handR.png"), 270, 180)
    return np.concatenate([np.concatenate([hr, hl], axis=1), person], axis=0)   # 视频里主角右手在画面左


rows = []
for f in data["frames"]:
    f = int(f)
    row = [video_panel(f)] + [model_panel(n, f) for n in ("old", "fsb", "H2f")]
    rows.append(np.concatenate(row, axis=1))
    print("row", f, labels[str(f)], flush=True)
if os.path.exists(TMP):
    os.remove(TMP)
for k in range(0, len(rows), 5):
    block = rows[k:k + 5][::-1]
    img = np.concatenate(block, axis=0)
    h, w = img.shape[:2]
    out = bpy.data.images.new("sheet", w, h)
    out.pixels = img.ravel().tolist()
    out.filepath_raw = f"{out_prefix}_{k // 5 + 1}.png"; out.file_format = "PNG"; out.save()
    print("SAVED", out.filepath_raw, [int(x) for x in data["frames"][k:k + 5]])
