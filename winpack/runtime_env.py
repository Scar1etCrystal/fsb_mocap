"""整合包运行时环境：目录约定、环境变量、离线补丁。run_video.py / check_env.py 第一件事就是 setup()。

整合包目录（= PoseCapture 插件偏好里的「FastSAM3DBody 路径」）：
    FastSAM3DBody/
      envs/python.exe                 嵌入式 Python 3.10 + site-packages（torch cu124 …）
      app/Fast-SAM-3D-Body/           FastSAM3DBody 源码快照（d72aa369）
      app/fsb/                        我们的转换/打包代码（含 assets/mhr_rest.npz）
      app/winpack/                    本目录：入口、自检、下载、构建
      app/third_party/MoGe/           MoGe 源码（只用 moge.model.v2）
      checkpoints/sam-3d-body-dinov3/ model.ckpt  model_config.yaml  assets/mhr_model.pt
      checkpoints/yolo/               yolo11m-pose.pt
      checkpoints/torch_hub/facebookresearch_dinov3_main/   DINOv3 结构代码（权重在 model.ckpt 里）
      checkpoints/hf/                 HF_HOME（MoGe-2 ViT-S 权重）
      outputs/                        每次运行的 take 目录；outputs/_work/ 是可续跑的中间结果
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
PACK = os.path.dirname(APP)

# 与 2026-10-04 云端验证（407/466 ms/帧，RTX 4080）完全相同的推理开关（cloud/env.sh）
VALIDATED_ENV = {
    "IMG_SIZE": "512", "LAYER_DTYPE": "fp32", "SKIP_KEYPOINT_PROMPT": "1", "GPU_HAND_PREP": "1",
    "BODY_INTERM_PRED_LAYERS": "0,1,2", "HAND_INTERM_PRED_LAYERS": "0,1", "KEYPOINT_PROMPT_INTERM_INTERVAL": "999",
    "MHR_USE_CUDA_GRAPH": "0", "FOV_TRT": "0", "FOV_FAST": "1", "FOV_MODEL": "s", "FOV_LEVEL": "0",
    # Windows 上没有官方 triton → torch.compile 不可用；TensorRT 引擎需在目标卡上另建，默认都关
    "USE_COMPILE": "0", "USE_COMPILE_BACKBONE": "0", "DECODER_COMPILE": "0", "USE_TRT_BACKBONE": "0",
}


def layout(pack=None):
    """代码位置跟着本文件走（app = winpack 的上一级；开发仓库里就是仓库根），数据位置跟着 pack。"""
    pack = os.path.abspath(pack or PACK)
    # 云端等非整合包布局用环境变量指过去：FSB_ROOT=代码目录，FSB_CKPT=权重目录，FSB_DINOV3_HUB=DINOv3 结构代码
    ck = os.path.abspath(os.environ.get("FSB_CKPT") or os.path.join(pack, "checkpoints"))
    return {
        "pack": pack, "app": APP, "fsb_src": os.path.abspath(os.environ.get("FSB_ROOT") or os.path.join(APP, "Fast-SAM-3D-Body")),
        "moge_src": os.path.join(APP, "third_party", "MoGe"), "ckpt": ck,
        "dinov3_hub": os.environ.get("FSB_DINOV3_HUB") or os.path.join(ck, "torch_hub", "facebookresearch_dinov3_main"),
        "hf_home": os.path.join(ck, "hf"), "outputs": os.path.join(pack, "outputs"),
        "yolo_cfg": os.path.join(pack, "_ultralytics_config"),
    }


def safe_stdio():
    """中文 Windows 上输出被重定向时 stdout 是 cp936，打印 ✓/✗ 会抛 UnicodeEncodeError（命令就此中断）：改成替换字符。"""
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(errors="replace")
        except Exception:
            pass


def setup(pack=None, offline=True):
    """设环境变量 + sys.path（必须在 import torch / sam_3d_body 之前调用）。返回 layout 字典。"""
    safe_stdio()
    L = layout(pack)
    for k, v in VALIDATED_ENV.items():
        os.environ.setdefault(k, v)
    # 包内的 HF 缓存优先（用户自己全局设过 HF_HOME 时，setdefault 会让 MoGe 在离线模式下找不到权重、静默退回 60° 视角）
    if os.path.isdir(L["hf_home"]):
        os.environ["HF_HOME"] = L["hf_home"]
    else:
        os.environ.setdefault("HF_HOME", L["hf_home"])
    os.environ.setdefault("TORCH_HOME", os.path.join(L["ckpt"], "torch_hub"))
    os.environ.setdefault("YOLO_CONFIG_DIR", L["yolo_cfg"])
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")   # torch≥2.6（cu128 版）默认 weights_only=True
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")            # Windows 上 numpy/torch 各带一份 OpenMP 时防崩
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    if offline:                                                       # 整合包不联网：缺文件就报错，不去下载
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("YOLO_OFFLINE", "True")
    os.makedirs(L["yolo_cfg"], exist_ok=True)
    for p in (L["moge_src"], L["fsb_src"], L["app"]):
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)
    return L


def patch_torch_hub(local_dir):
    """FastSAM3DBody 只认 ~/.cache/torch/hub/facebookresearch_dinov3_main，否则去 GitHub 拉（还要先联网查默认分支，
    云端就卡在这）。整合包把结构代码放在包内，这里把对 facebookresearch/dinov3 的 github 加载改成包内 local 加载。"""
    import torch.hub
    if getattr(torch.hub.load, "_fsb_patched", False):
        return
    orig = torch.hub.load

    def load(repo_or_dir, model, *args, source="github", trust_repo=None, force_reload=False, verbose=True,
             skip_validation=False, **kwargs):
        repo = str(repo_or_dir).split(":")[0].rstrip("/")
        if source == "github" and repo == "facebookresearch/dinov3" and os.path.isfile(os.path.join(local_dir, "hubconf.py")):
            return orig(local_dir, model, *args, source="local", **kwargs)
        return orig(repo_or_dir, model, *args, source=source, trust_repo=trust_repo, force_reload=force_reload,
                    verbose=verbose, skip_validation=skip_validation, **kwargs)

    load._fsb_patched = True
    torch.hub.load = load


def ascii_safe_copy(path, work_dir):
    """OpenCV 在 Windows 上打不开含中文等非 ASCII 字符的路径：复制到 work 目录下的 ASCII 文件名。"""
    try:
        path.encode("ascii")
        return path, False
    except UnicodeEncodeError:
        import shutil
        os.makedirs(work_dir, exist_ok=True)
        dst = os.path.join(work_dir, "input" + os.path.splitext(path)[1].lower())
        try:
            dst.encode("ascii")
        except UnicodeEncodeError:                                    # work 目录本身含非 ASCII：退到系统临时目录
            import hashlib
            import tempfile
            name = "fsb_" + hashlib.md5(path.encode("utf-8")).hexdigest()[:12] + os.path.splitext(path)[1].lower()
            dst = os.path.join(tempfile.gettempdir(), name)
            try:
                dst.encode("ascii")
            except UnicodeEncodeError:                                # 用户名是中文时临时目录也不行：退到公共目录
                base = os.environ.get("PUBLIC") or os.path.splitdrive(os.path.abspath(path))[0] + os.sep
                os.makedirs(os.path.join(base, "fsb_tmp"), exist_ok=True)
                dst = os.path.join(base, "fsb_tmp", name)
        if not (os.path.isfile(dst) and os.path.getsize(dst) == os.path.getsize(path)):
            shutil.copyfile(path, dst)
        return dst, True
