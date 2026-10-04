"""Blender（后台）：MHR 网格（灰蓝）与重定向后的 SMPL-X 网格（橙）并排渲染：正面 + 侧面。不保存 .blend。"""
import sys
import bpy
import numpy as np
from mathutils import Matrix, Vector

args = sys.argv[sys.argv.index("--") + 1:]
data = np.load(args[0])
out_prefix = args[1]
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)
sc = bpy.context.scene
sc.render.engine = "BLENDER_WORKBENCH"
sc.display.shading.light = "STUDIO"; sc.display.shading.color_type = "OBJECT"; sc.display.shading.show_cavity = True
sc.render.resolution_x, sc.render.resolution_y = 1100, 800
sc.render.image_settings.file_format = "PNG"
cam_data = bpy.data.cameras.new("cam"); cam = bpy.data.objects.new("cam", cam_data); sc.collection.objects.link(cam); sc.camera = cam
cam_data.type = "ORTHO"; cam_data.ortho_scale = 2.6


def mesh(name, verts, faces, color, offset):
    m = bpy.data.meshes.new(name)
    v = verts.astype(np.float64) + offset
    # 模型系 (x 左, y 上, z 前) → Blender (x, y=−z_model, z=y_model)
    vb = np.stack([v[:, 0], -v[:, 2], v[:, 1]], 1)
    m.from_pydata(vb.tolist(), [], faces.tolist()); m.update()
    o = bpy.data.objects.new(name, m); sc.collection.objects.link(o); o.color = (*color, 1.0)
    return o


for f in data["frames"]:
    for o in [o for o in bpy.data.objects if o.type == "MESH"]:
        bpy.data.objects.remove(o, do_unlink=True)
    vm, vs = data[f"mhr_{f}"], data[f"smplx_{f}"]
    c = vm.mean(0)
    mesh("MHR", vm - c, data["mhr_faces"], (0.55, 0.62, 0.75), np.array([0.55, 0, 0]))
    mesh("SMPLX", vs - c, data["smplx_faces"], (0.95, 0.6, 0.25), np.array([-0.55, 0, 0]))
    for view, loc, rotm in (("front", Vector((0, -6, 0.0)), Matrix.Rotation(np.pi / 2, 4, "X")),
                            ("side", Vector((6, 0, 0.0)), Matrix.Rotation(np.pi / 2, 4, "Z") @ Matrix.Rotation(np.pi / 2, 4, "X"))):
        cam.matrix_world = Matrix.Translation(loc) @ rotm
        if view == "side":                      # 侧面两个人会重叠：上下错开
            for o in bpy.data.objects:
                if o.name == "SMPLX":
                    o.location = (1.10, 0, 0)    # 侧视时沿视线方向错开不起作用，改为前后（x）错开
        sc.render.filepath = f"{out_prefix}_f{int(f)}_{view}.png"
        bpy.ops.render.render(write_still=True)
        print("RENDERED", sc.render.filepath)
