"""Blender 里加载打过补丁的 PoseCapture 插件，走 FastSAM3DBody 前端完整跑一次捕捉（到存好捕捉文件夹为止；
导入动画要 GVHMR 目录里的 SMPL 许可模型，本机没有，不测）。

    blender -b <任意.blend> --python addon_fsb_e2e.py -- <补丁后插件.py> <FSB 整合包> <PKL 输出目录> <视频> <fsb 仓库>

FSB 整合包可以是"替身"：envs/bin/python + app/（build_pack.py --steps app）+ outputs/_work/<视频名>/ 里预置好
云端原始结果——这样 run_video.py 会跳过全部 GPU 阶段，只做转换+打包，整条插件链路照常走。
"""
import importlib.util
import json
import os
import pickle
import sys
import tempfile
import types

import bpy

args = sys.argv[sys.argv.index("--") + 1:]
addon_path, fsb_pack, save_dir, video, repo = args[:5]
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)


spec = importlib.util.spec_from_file_location("pose_capture_fsb_test", addon_path)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)
prefs = types.SimpleNamespace(gvhmr_dir="", hamer_dir="", wilor_dir="", pkl_save_dir=save_dir, ffmpeg_dir="",
                              better_fbx_dir="", fsb_dir=fsb_pack)
mod._prefs = lambda: prefs                                    # 模块不是以插件身份启用的：偏好设置用替身

try:
    mod.register()
    check("register()", True)
except Exception as exc:
    check("register()", False, repr(exc))

check("偏好设置有 fsb_dir", "fsb_dir" in mod.PoseCapturePreferences.__annotations__)
s = bpy.context.scene.pose_capture_settings
check("身体前端默认 GVHMR", s.body_frontend == "GVHMR")
items = [i.identifier for i in s.bl_rna.properties["body_frontend"].enum_items]
check("身体前端选项", items == ["GVHMR", "FASTSAM3DB"], str(items))


class FakeLayout:
    def __init__(self):
        self.props, self.labels = [], []

    def prop(self, data, name, **kw):
        self.props.append(name)

    def label(self, text="", **kw):
        self.labels.append(text)

    def separator(self, **kw):
        pass

    def row(self, **kw):
        return self

    def box(self, **kw):
        return self

    def column(self, **kw):
        return self


def draw(frontend):
    s.body_frontend, s.mode = frontend, "FULL"
    fake = types.SimpleNamespace(layout=FakeLayout())
    mod.POSE_PT_settings.draw(fake, bpy.context)
    return fake.layout


lg = draw("GVHMR")
check("面板 GVHMR：手部后端/相机模式照旧", {"body_frontend", "hand_backend", "camera_motion"} <= set(lg.props), str(lg.props[:8]))
lf = draw("FASTSAM3DB")
check("面板 FastSAM3DBody：隐藏手部后端/相机模式", "hand_backend" not in lf.props and "camera_motion" not in lf.props
      and any("FastSAM3DBody" in t for t in lf.labels), str(lf.props[:8]))

check("_fsb_pack_error() 替身整合包可用", mod._fsb_pack_error() == "", mod._fsb_pack_error())
prefs.fsb_dir = "/nonexistent_fsb"
check("_fsb_pack_error() 路径不对时报错", "FastSAM3DBody 路径" in mod._fsb_pack_error())
prefs.fsb_dir = fsb_pack

# 捕捉算子的新检查分支（都在启动线程之前返回）
class FakeOp:
    def __init__(self):
        self.msgs = []

    def report(self, kind, msg):
        self.msgs.append(msg)


tmp = tempfile.mkdtemp()
img = os.path.join(tmp, "x.jpg")
open(img, "wb").write(b"\xff\xd8\xff\xd9")
s.body_frontend, s.filepath, s.is_processing = "FASTSAM3DB", video, False
op = FakeOp()
r = mod.POSE_OT_capture.execute(op, bpy.context)
check("算子：没有 GVHMR 身体模型时拒绝", r == {"CANCELLED"} and any("身体模型" in m for m in op.msgs), str(op.msgs))
os.makedirs(os.path.join(tmp, "gvhmr", "inputs", "checkpoints", "body_models"))
prefs.gvhmr_dir = os.path.join(tmp, "gvhmr")
s.filepath = img
op = FakeOp()
r = mod.POSE_OT_capture.execute(op, bpy.context)
check("算子：FastSAM3DBody 拒绝图片", r == {"CANCELLED"} and any("只支持视频" in m for m in op.msgs), str(op.msgs))
prefs.gvhmr_dir = ""

check("流水线默认前端 GVHMR", mod.PoseCapturePipeline(filepath=video, gender="FEMALE", mode="FULL").body_frontend == "GVHMR")

# 完整捕捉（同步跑 run()；插件里是后台线程跑同一个方法）
logs, done = [], {}
p = mod.PoseCapturePipeline(filepath=video, gender="FEMALE", mode="FULL", body_frontend="FASTSAM3DB",
                            on_complete=lambda pkl, err: done.update(pkl=pkl, err=err), on_progress=logs.append)
p.run()
check("捕捉无错误", not done.get("err"), str(done.get("err"))[:300])
pkl = done.get("pkl")
check("捕捉产出 pkl", bool(pkl) and os.path.isfile(pkl), str(pkl))
check("进度行实时转发", sum(1 for m in logs if m.startswith("[FSB]")) >= 3, str([m for m in logs if m.startswith("[FSB]")][:3]))
if pkl and os.path.isfile(pkl):
    sub = os.path.dirname(pkl)
    stem = os.path.splitext(os.path.basename(video))[0]
    want = ["gvhmr/hmr4d_results.pt_person-0.pkl", "gvhmr/0_input_video.mp4", "hamer/merged.pkl",
            f"hamer/{stem}_hamer_data.pkl", "fsb_meta.json", os.path.basename(sub) + os.path.splitext(video)[1]]
    miss = [w for w in want if not os.path.isfile(os.path.join(sub, w))]
    check("捕捉文件夹结构同 GVHMR+HAMER", not miss, f"缺 {miss}" if miss else sub)
    check("整合包 outputs/<take> 已收走", not os.path.exists(os.path.join(fsb_pack, "outputs", stem)))
    check("续跑目录成功后已清理", not os.path.exists(os.path.join(fsb_pack, "outputs", "_work", stem)))
    with open(pkl, "rb") as fh:
        d = pickle.load(fh, encoding="latin1")
    F = d["smpl_params_global"]["body_pose"].shape[0]
    check("顶层 pkl 可读、帧数、src_fps", F == 999 and abs(float(d.get("src_fps", 0)) - 30.0) < 1e-6
          and d["hamer_merge_info"]["source"] == "fast_sam_3d_body", f"F={F} src_fps={d.get('src_fps')}")
    sys.path.insert(0, repo)
    from fsb import contract
    rep = contract.validate_take(sub)
    check("捕捉文件夹通过 R1 合同校验（Blender 自带 numpy）", rep["ok"], "; ".join(rep["errors"][:3]))
    print("WARNINGS", rep["warnings"])

print(f"==== {RESULTS.count(False)} FAIL / {len(RESULTS)} ====", flush=True)
