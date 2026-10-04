"""整合包环境自检：Python / 显卡 / 驱动 / 依赖 / 模型文件，可选 --smoke 真跑一帧。

    envs\\python.exe app\\winpack\\check_env.py            静态检查（几秒）
    envs\\python.exe app\\winpack\\check_env.py --smoke    再加载模型真跑：1 帧人体 + 内参 + 骨架重算，报速度和峰值显存
    envs\\python.exe app\\winpack\\check_env.py --hash     额外校验模型文件哈希（model.ckpt 2 GB，约 10 秒）

结果写到 <整合包>/check_env_report.json；退出码 0 = 全部通过。
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import runtime_env  # noqa: E402

# 推理链用到的模块（顶层 import 失败 = 环境坏了）
MODULES = ["numpy", "cv2", "PIL", "scipy", "torch", "torchvision", "roma", "einops", "timm", "yacs", "omegaconf",
           "pytorch_lightning", "braceexpand", "huggingface_hub", "ultralytics", "lap", "utils3d", "pipeline",
           "moge.model.v2", "sam_3d_body", "tools.build_fov_estimator", "fsb.pipeline"]


class Report:
    def __init__(self):
        self.rows, self.ok = [], True

    def add(self, item, ok, detail="", warn=False):
        mark = "OK  " if ok else ("WARN" if warn else "FAIL")
        if not ok and not warn:
            self.ok = False
        self.rows.append({"item": item, "status": mark.strip(), "detail": detail})
        print(f"[{mark}] {item:28s} {detail}", flush=True)


def nvidia_smi():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=15)
        return out.stdout.strip() or out.stderr.strip()
    except Exception as exc:
        return f"nvidia-smi 不可用：{exc!r}"


def check_gpu(rep):
    import torch
    rep.add("torch", True, f"{torch.__version__}（CUDA {torch.version.cuda}，cuDNN {torch.backends.cudnn.version()}）")
    if not torch.cuda.is_available():
        rep.add("CUDA 可用", False, "torch.cuda.is_available() = False：没有 NVIDIA 显卡、驱动太旧（cu124 需 ≥ 528），"
                                   "或装成了 CPU 版 torch。nvidia-smi：" + nvidia_smi())
        return None
    i = torch.cuda.current_device()
    name = torch.cuda.get_device_name(i)
    cap = torch.cuda.get_device_capability(i)
    vram = torch.cuda.get_device_properties(i).total_memory / 2 ** 30
    rep.add("显卡", True, f"{name}，sm_{cap[0]}{cap[1]}，{vram:.1f} GB；{nvidia_smi()}")
    archs = torch.cuda.get_arch_list()
    sm = f"sm_{cap[0]}{cap[1]}"
    supported = sm in archs or any(a.startswith("compute_") and int(a.split("_")[1]) <= cap[0] * 10 + cap[1] for a in archs)
    rep.add("torch 支持该显卡架构", supported,
            f"{sm} {'在' if supported else '不在'} {archs}" + ("" if supported else
            "：RTX 50 系（sm_120）需要 cu128 版整合包（build_pack.py --profile cu128）"))
    rep.add("显存", vram >= 7.5, f"{vram:.1f} GB（建议 ≥ 8 GB；--smoke 会报实测峰值）", warn=vram < 7.5)
    return vram


def check_files(rep, L, deep):
    import download_weights as dw
    items = json.load(open(dw.MANIFEST, encoding="utf-8"))["files"]
    for it in items:
        st, why = dw.local_status(L["pack"], it, deep=deep)
        rep.add(f"文件 {it['name']}", st == "ok", f"{why or st}  {it['dest']}")
    rest = os.path.join(L["app"], "fsb", "assets", "mhr_rest.npz")
    rep.add("文件 mhr_rest.npz", os.path.isfile(rest), rest)


def smoke(rep, L):
    import numpy as np
    import torch
    from fsb import pipeline
    runtime_env.patch_torch_hub(L["dinov3_hub"])
    paths = pipeline.Paths(L["fsb_src"], ckpt=L["ckpt"])
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    est, model = pipeline.build_model(paths)
    rep.add("加载 SAM 3D Body", True, f"{time.time() - t0:.1f}s")
    W, H = 1280, 720                                          # 合成画面：灰底 + 人形框（不需要真的有人，验证整条 GPU 链路）
    img = np.full((H, W, 3), 128, np.uint8)
    box = np.array([[W * 0.4, H * 0.1, W * 0.6, H * 0.95]], np.float32)
    K = torch.tensor(pipeline.K_from_f_mm(W, H, 35.0), dtype=torch.float32)[None]
    times = []
    for _ in range(6):
        t = time.time()
        outs = est.process_one_image(img, bboxes=box, cam_int=K.clone())
        torch.cuda.synchronize()
        times.append(time.time() - t)
    o = outs[0]
    shapes = {k: tuple(np.shape(o[k])) for k in ("pred_cam_t", "pred_keypoints_3d", "global_rot", "body_pose_params",
                                                 "hand_pose_params", "scale_params", "shape_params")}
    rep.add("单帧推理", True, f"{np.mean(times[2:]) * 1000:.0f} ms/帧（预热后；云端 RTX 4080 实测 407–466）；{shapes}")
    a = {k: np.asarray(o[k].detach().cpu().numpy() if hasattr(o[k], "detach") else o[k], np.float32)[None]
         for k in ("global_rot", "body_pose_params", "hand_pose_params", "scale_params", "shape_params")}
    jc, jr = pipeline._recompute_skeleton(model, a)
    rep.add("骨架重算", jc.shape[1:] == (127, 3) and jr.shape[-2:] == (3, 3), f"关节 {jc.shape}，旋转 {jr.shape}")
    del est, model
    try:
        fov = pipeline.build_fov(paths)
        t = time.time()
        Kf = fov.get_cam_intrinsics(img)
        Kf = Kf.cpu().numpy() if hasattr(Kf, "cpu") else np.asarray(Kf)
        rep.add("MoGe-2 内参", np.isfinite(Kf).all(), f"fx={float(np.asarray(Kf).reshape(-1, 3, 3)[0, 0, 0]):.1f}，{time.time() - t:.1f}s")
    except Exception as exc:
        rep.add("MoGe-2 内参", False, f"{exc!r}（不致命：会退回默认 60° 视角，只影响深度尺度）", warn=True)
    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    rep.add("峰值显存", True, f"{peak:.2f} GB（PyTorch 分配量；任务管理器里会再多 ~0.5 GB 上下文）")


def main(argv=None):
    ap = argparse.ArgumentParser(description="整合包环境自检")
    ap.add_argument("--smoke", action="store_true", help="加载模型真跑一帧")
    ap.add_argument("--hash", action="store_true", help="校验模型文件哈希")
    ap.add_argument("--pack", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    L = runtime_env.setup(a.pack)
    rep = Report()
    rep.add("Python", sys.version_info[:2] == (3, 10), f"{sys.version.split()[0]} @ {sys.executable}",
            warn=sys.version_info[:2] != (3, 10))
    rep.add("系统", True, f"{platform.platform()}；整合包 {L['pack']}")
    try:
        L["pack"].encode("ascii")
        rep.add("整合包路径为纯 ASCII", True, "")
    except UnicodeEncodeError:
        rep.add("整合包路径为纯 ASCII", False, "路径含中文等字符：部分 C 扩展/OpenCV 读文件会失败，请放到纯英文路径", warn=True)
    if os.name == "nt" and len(L["pack"]) > 80:
        rep.add("整合包路径长度", False, f"{len(L['pack'])} 字符；site-packages 深层文件可能超 260 字符限制，建议放浅一点（如 D:\\PoseCapture_Pack）", warn=True)
    for m in MODULES:
        try:
            t = time.time()
            mod = importlib.import_module(m)
            rep.add(f"import {m}", True, f"{getattr(mod, '__version__', '')} {time.time() - t:.1f}s".strip())
        except Exception as exc:
            rep.add(f"import {m}", False, f"{exc!r}"[:300])
    try:
        import cv2
        ff = "FFMPEG:" in cv2.getBuildInformation() and "YES" in cv2.getBuildInformation().split("FFMPEG:")[1][:40]
        rep.add("OpenCV 带 FFmpeg", ff, "读 mp4/mkv 需要")
    except Exception:
        pass
    vram = None
    try:
        vram = check_gpu(rep)
    except Exception as exc:
        hint = "（DLL 加载失败多半是缺 VC++ 运行库：运行整合包里的 vc_redist.x64.exe）" if ("126" in str(exc) or "DLL" in str(exc)) else ""
        rep.add("torch / CUDA", False, f"{exc!r}"[:300] + hint)
    hub = os.path.join(L["dinov3_hub"], "hubconf.py")
    rep.add("DINOv3 结构代码", os.path.isfile(hub), hub)
    check_files(rep, L, deep=a.hash)
    if a.smoke:
        if vram is None:
            rep.add("--smoke", False, "没有可用 GPU，跳过")
        elif not rep.ok:
            rep.add("--smoke", False, "前面有 FAIL，先修好再跑 smoke")
        else:
            try:
                smoke(rep, L)
            except Exception as exc:
                traceback.print_exc()
                rep.add("--smoke", False, f"{exc!r}"[:400])
    out = os.path.join(L["pack"], "check_env_report.json")
    try:
        with open(out, "w", encoding="utf-8") as fh:
            json.dump({"ok": rep.ok, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "rows": rep.rows}, fh,
                      ensure_ascii=False, indent=1)
    except OSError:
        pass
    print(("\n全部通过。" if rep.ok else "\n有 FAIL 项（见上）。") + f" 报告：{out}")
    return 0 if rep.ok else 1


if __name__ == "__main__":
    sys.exit(main())
