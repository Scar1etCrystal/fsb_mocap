"""组装 FastSAM3DBody Windows 一键整合包（在 Linux 或 Windows 上都能跑：Windows 的包是"交叉组装"出来的——
嵌入式 Python 解压即用，依赖用 pip 的 --platform win_amd64 --target 直接装进 envs/Lib/site-packages，不需要在 Windows 上编译任何东西）。

    python build_pack.py --out D:/PoseCapture_Pack/FastSAM3DBody                     全部步骤（下载约 5.5 GB）
    python build_pack.py --out OUT --dry-run                                          只打印计划，不下载
    python build_pack.py --out OUT --estimate                                         查询每个 wheel / 模型的下载大小（只查元数据）
    python build_pack.py --out OUT --steps app,launchers,check                        只更新代码和启动脚本（改代码后用）
    python build_pack.py --out OUT --profile cu128                                    RTX 50 系（sm_120）版：torch 2.7.1+cu128（未经云端验证）

步骤（--steps 逗号分隔，默认全部，按此顺序）：
    python     下载 python-3.10.11-embed-amd64.zip → envs/，改 python310._pth 打开 site
    wheels     pip download 锁文件里的全部 win_amd64/cp310 wheel → wheelhouse/（git 依赖和纯源码包现场打成 wheel）
    install    wheelhouse 里的 wheel 全部 --no-deps 装进 envs/Lib/site-packages
    app        代码：Fast-SAM-3D-Body 快照、fsb、winpack、MoGe 源码（moge/）
    weights    模型文件（download_weights.py；SAM 3D Body 权重是 gated，需 --hf-token）
    vcredist   附带 vc_redist.x64.exe（VC++ 运行库安装包，缺 DLL 时用户自己点）
    launchers  run_video.bat / check_env.bat / download_weights.bat / 使用说明.txt（CRLF）
    check      静态核对目录与版本，写 BUILD_INFO.json
    archive    7z 打包（需要 7z）
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
STEPS = ["python", "wheels", "install", "app", "weights", "vcredist", "launchers", "check", "archive"]
PY_VERSION = "3.10.11"
PY_EMBED = f"python-{PY_VERSION}-embed-amd64.zip"
PY_URLS = [f"https://www.python.org/ftp/python/{PY_VERSION}/{PY_EMBED}",
           f"https://mirrors.huaweicloud.com/python/{PY_VERSION}/{PY_EMBED}"]
PY_SIZE = 8629277
VCREDIST_URL = "https://aka.ms/vs/17/release/vc_redist.x64.exe"
PROFILES = {
    "cu124": {"lock": "requirements_win.lock.txt", "torch_index": "https://download.pytorch.org/whl/cu124",
              "torch_index_cn": "https://mirrors.aliyun.com/pytorch-wheels/cu124/", "validated": True,
              "note": "torch 2.5.1+cu124：RTX 20/30/40 系（sm_75–sm_89），驱动 ≥ 528；= 2026-10-04 云端验证环境"},
    "cu128": {"lock": "requirements_win_cu128.lock.txt", "torch_index": "https://download.pytorch.org/whl/cu128",
              "torch_index_cn": "https://mirrors.aliyun.com/pytorch-wheels/cu128/", "validated": False,
              "note": "torch 2.7.1+cu128：含 RTX 50 系（sm_120），驱动 ≥ 570；未经云端验证，先跑 check_env --smoke"},
}
MOGE_COMMIT = "74fbce054ebed49800de42d0ad0e83495065719a"
FSB_COMMIT = "d72aa36913a1673ace029d345f437561a85ec9e2"
PLATFORM_ARGS = ["--platform", "win_amd64", "--python-version", "3.10", "--implementation", "cp", "--abi", "cp310",
                 "--only-binary=:all:"]


UA = {"User-Agent": "pip/24.0 fsb-winpack"}         # 默认 Python-urllib UA 会被 PyTorch 的 CDN 拒（403）


def urlopen(url, method=None, timeout=60):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA, method=method), timeout=timeout)


def say(msg):
    print(msg, flush=True)


def human(n):
    return f"{n / 2 ** 30:.2f} GB" if n >= 2 ** 30 else f"{n / 2 ** 20:.1f} MB"


def run(cmd, dry, **kw):
    say("  $ " + " ".join(str(c) for c in cmd))
    if not dry:
        subprocess.run([str(c) for c in cmd], check=True, **kw)


def fetch(url, dest, dry, size=None):
    if os.path.isfile(dest) and (size is None or os.path.getsize(dest) == size):
        say(f"  已有 {dest}")
        return
    say(f"  下载 {url} → {dest}")
    if dry:
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with urlopen(url, timeout=120) as r, open(dest + ".part", "wb") as fh:
        shutil.copyfileobj(r, fh, 8 << 20)
    os.replace(dest + ".part", dest)


def read_lock(profile):
    """锁文件 → (二进制 wheel 规格, 需要现场打 wheel 的规格)。"""
    binary, build = [], []
    for line in open(os.path.join(HERE, PROFILES[profile]["lock"]), encoding="utf-8"):
        spec = line.split("#")[0].strip()
        if not spec:
            continue
        if " @ git+" in spec or spec.startswith("antlr4-python3-runtime=="):
            build.append(spec.split(" @ ")[1] if " @ " in spec else spec)
        else:
            binary.append(spec)
    return binary, build


# ------------------------------------------------------------------ 步骤

def step_python(a):
    envs = os.path.join(a.out, "envs")
    zp = os.path.join(a.cache, PY_EMBED)
    for url in PY_URLS[::-1] if a.cn else PY_URLS:
        try:
            fetch(url, zp, a.dry_run, PY_SIZE)
            break
        except Exception as exc:
            say(f"  {url} 失败：{exc!r}")
    else:
        raise RuntimeError("嵌入式 Python 下载失败")
    if a.dry_run:
        say(f"  解压到 {envs}，改 python310._pth（打开 import site，加 Lib\\site-packages）")
        return
    os.makedirs(envs, exist_ok=True)
    with zipfile.ZipFile(zp) as zf:
        zf.extractall(envs)
    os.makedirs(os.path.join(envs, "Lib", "site-packages"), exist_ok=True)
    pth = os.path.join(envs, "python310._pth")
    with open(pth, "w", encoding="utf-8", newline="\r\n") as fh:
        fh.write("python310.zip\n.\nLib\\site-packages\n# app/ code paths are added at runtime by app/winpack/runtime_env.py\nimport site\n")  # ._pth 只放 ASCII
    say(f"  ✓ envs/python.exe（{PY_VERSION} 嵌入式），{pth} 已改")


def step_wheels(a):
    wh = a.wheelhouse
    os.makedirs(wh, exist_ok=True)
    binary, build = read_lock(a.profile)
    prof = PROFILES[a.profile]
    torch_specs = [s for s in binary if "+cu" in s]
    other = [s for s in binary if "+cu" not in s]
    idx = ["-i", a.pip_index] if a.pip_index else []
    req = os.path.join(a.cache, f"wheels_{a.profile}.txt")
    if not a.dry_run:
        os.makedirs(a.cache, exist_ok=True)
        open(req, "w", encoding="utf-8").write("\n".join(other) + "\n")
    run([sys.executable, "-m", "pip", "download", "--no-deps", "-d", wh, *PLATFORM_ARGS, *idx, "-r", req], a.dry_run)
    tidx = prof["torch_index_cn"] if a.cn else prof["torch_index"]
    # 阿里云镜像是平铺文件列表（-f）；官方是 PEP 503 索引（--index-url）
    tsrc = ["-f", tidx] if "aliyun" in tidx else ["--index-url", tidx]
    run([sys.executable, "-m", "pip", "download", "--no-deps", "-d", wh, *PLATFORM_ARGS, *tsrc, *torch_specs], a.dry_run)
    for spec in build:                                          # 纯 Python：在本机打成 py3-none-any wheel
        run([sys.executable, "-m", "pip", "wheel", "--no-deps", "-w", wh, *idx, spec], a.dry_run)
    if not a.dry_run:
        whls = glob.glob(os.path.join(wh, "*.whl"))
        bad = [w for w in whls if not re.search(r"(cp310-win_amd64|abi3-win_amd64|none-any|py3-none-win_amd64)\.whl$", w)]
        if bad:
            raise RuntimeError(f"wheelhouse 里有非 Windows/cp310 的 wheel：{bad}")
        say(f"  ✓ {len(whls)} 个 wheel，{human(sum(os.path.getsize(w) for w in whls))}")


def step_install(a):
    sp = os.path.join(a.out, "envs", "Lib", "site-packages")
    binary, build = read_lock(a.profile)
    names = {re.split(r"[=@ ]", s)[0].lower().replace("-", "_") for s in binary} | {"utils3d", "pipeline", "antlr4_python3_runtime"}
    whls = sorted(glob.glob(os.path.join(a.wheelhouse, "*.whl")))
    pick = [w for w in whls if os.path.basename(w).split("-")[0].lower() in names]
    if not a.dry_run and len(pick) < len(names):
        have = {os.path.basename(w).split("-")[0].lower() for w in pick}
        raise RuntimeError(f"wheelhouse 缺：{sorted(names - have)}（先跑 wheels 步骤）")
    say(f"  安装 {len(pick) or len(names)} 个 wheel → {sp}")
    run([sys.executable, "-m", "pip", "install", "--no-deps", "--no-index", "--target", sp, "--upgrade",
         *PLATFORM_ARGS, *pick], a.dry_run)
    if not a.dry_run:
        for junk in ("bin", "Scripts"):                         # 宿主平台生成的命令行启动器，Windows 包里用不到
            shutil.rmtree(os.path.join(sp, junk), ignore_errors=True)
        if not a.no_slim:                                       # 只有编译 C++ 扩展才用得到：cu124 省 0.78 GB，cu128 省 0.84 GB
            freed = 0
            for f in glob.glob(os.path.join(sp, "torch", "lib", "*.lib")):
                freed += os.path.getsize(f)
                os.remove(f)
            inc = os.path.join(sp, "torch", "include")
            if os.path.isdir(inc):
                freed += dir_size(inc)
                shutil.rmtree(inc)
            say(f"  瘦身：删 torch/lib/*.lib + torch/include，{human(freed)}（不影响推理；torch.compile/编译扩展用不了，本来也关着）")
        say(f"  ✓ site-packages {human(dir_size(sp))}")


def _copy_tree(src, dst, dry, ignore=("__pycache__", "*.pyc")):
    if os.path.abspath(src) == os.path.abspath(dst):          # 在整合包里面跑本脚本时源 = 目标：先删后拷会把代码删光
        say(f"  {dst} 就是源目录，跳过")
        return
    say(f"  复制 {src} → {dst}")
    if not dry:
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns(*ignore))


def step_app(a):
    app = os.path.join(a.out, "app")
    # 1) FastSAM3DBody 快照：--fsb-src 目录，或仓库里云端打包回来的源码 tgz
    dst = os.path.join(app, "Fast-SAM-3D-Body")
    tgz = os.path.join(REPO, "data", "cloud_meta", "fastsam3dbody_src.tgz")
    if a.fsb_src:
        src_root = a.fsb_src
    elif not os.path.isfile(tgz) and os.path.isfile(os.path.join(dst, "sam_3d_body", "__init__.py")):
        say(f"  没有源码包 {tgz}，{dst} 已有 FastSAM3DBody 代码，保留")
        src_root = None
    elif not os.path.isfile(tgz):
        raise RuntimeError(f"找不到 FastSAM3DBody 源码：{tgz}；用 --fsb-src 指定源码目录")
    else:
        src_root = os.path.join(a.cache, "fsb_src")
        say(f"  解包 {tgz} → {src_root}")
        if not a.dry_run:
            shutil.rmtree(src_root, ignore_errors=True)
            os.makedirs(src_root)
            with tarfile.open(tgz) as tf:
                tf.extractall(src_root)
            subs = [d for d in os.listdir(src_root) if os.path.isdir(os.path.join(src_root, d))]
            if "sam_3d_body" not in subs and len(subs) == 1:
                src_root = os.path.join(src_root, subs[0])
    for sub in ("sam_3d_body", "tools") if src_root else ():
        _copy_tree(os.path.join(src_root, sub), os.path.join(dst, sub), a.dry_run)
    if not a.dry_run and src_root:
        for f in ("LICENSE", "README.md"):
            if os.path.isfile(os.path.join(src_root, f)):
                shutil.copy2(os.path.join(src_root, f), dst)
        open(os.path.join(dst, "FSB_COMMIT.txt"), "w", encoding="utf-8").write(FSB_COMMIT + "\n")
    # 2) 我们的代码
    _copy_tree(os.path.join(REPO, "fsb"), os.path.join(app, "fsb"), a.dry_run)
    _copy_tree(HERE, os.path.join(app, "winpack"), a.dry_run, ignore=("__pycache__", "*.pyc", "_cache", "wheelhouse"))
    # 3) MoGe 源码（只要 moge/ 包；不经 pip，理由见 requirements_win.in）
    moge_dst = os.path.join(app, "third_party", "MoGe")
    stamp = os.path.join(moge_dst, "MOGE_COMMIT.txt")
    if a.moge_src:
        _copy_tree(os.path.join(a.moge_src, "moge"), os.path.join(moge_dst, "moge"), a.dry_run)
    elif os.path.isfile(stamp) and open(stamp, encoding="utf-8").read().strip() == MOGE_COMMIT:
        say(f"  MoGe {MOGE_COMMIT[:12]} 已在 {moge_dst}，保留")
    else:
        zp = os.path.join(a.cache, f"MoGe_{MOGE_COMMIT[:12]}.zip")
        fetch((a.github_proxy or "") + f"https://github.com/microsoft/MoGe/archive/{MOGE_COMMIT}.zip", zp, a.dry_run)
        if not a.dry_run:
            shutil.rmtree(moge_dst, ignore_errors=True)
            with zipfile.ZipFile(zp) as zf:
                top = zf.namelist()[0].split("/")[0]
                for n in zf.namelist():
                    rel = n[len(top) + 1:]
                    if (rel.startswith("moge/") or rel == "LICENSE") and not n.endswith("/"):
                        p = os.path.join(moge_dst, *rel.split("/"))
                        os.makedirs(os.path.dirname(p), exist_ok=True)
                        with zf.open(n) as fi, open(p, "wb") as fo:
                            shutil.copyfileobj(fi, fo)
            open(os.path.join(moge_dst, "MOGE_COMMIT.txt"), "w", encoding="utf-8").write(MOGE_COMMIT + "\n")
    if not a.dry_run:
        say(f"  ✓ app/ {human(dir_size(app))}")


def step_weights(a):
    cmd = [sys.executable, os.path.join(HERE, "download_weights.py"), "--pack", a.out]
    if a.hf_token:
        cmd += ["--hf-token", a.hf_token]
    if a.cn:
        cmd += ["--hf-endpoint", "https://hf-mirror.com"]
    if a.github_proxy:
        cmd += ["--github-proxy", a.github_proxy]
    if a.dry_run:
        cmd.append("--dry-run")
    say("  $ " + " ".join(c if not c.startswith("hf_") else "hf_***" for c in cmd))
    subprocess.run(cmd, check=not a.dry_run)


def step_vcredist(a):
    fetch(VCREDIST_URL, os.path.join(a.out, "vc_redist.x64.exe"), a.dry_run)


LAUNCHERS = {
    "run_video.bat": r"""@echo off
chcp 65001 >nul
setlocal
set "PACK=%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
if "%~1"=="" (
  echo 用法：把视频文件拖到本文件上，或  run_video.bat 视频路径 [其他参数]
  echo 结果在 %PACK%outputs\视频名\ ；中间结果在 outputs\_work\ ，中断后重跑会续跑
  pause
  exit /b 2
)
"%PACK%envs\python.exe" "%PACK%app\winpack\run_video.py" --video %*
set RC=%ERRORLEVEL%
if not "%RC%"=="0" echo 失败（退出码 %RC%），上面有原因；环境问题先运行 check_env.bat
pause
exit /b %RC%
""",
    "check_env.bat": r"""@echo off
chcp 65001 >nul
set PYTHONUTF8=1
"%~dp0envs\python.exe" "%~dp0app\winpack\check_env.py" --smoke %*
pause
""",
    "download_weights.bat": r"""@echo off
chcp 65001 >nul
set PYTHONUTF8=1
echo SAM 3D Body 权重是 gated 仓库：先在 https://huggingface.co/facebook/sam-3d-body-dinov3 申请访问，
echo 通过后在 https://huggingface.co/settings/tokens 建一个 read token。
echo 本脚本直连 huggingface.co / github.com；国内连不上请改用 download_weights_cn.bat（经第三方镜像）。
set /p TOKEN=粘贴 HF token 后回车（已下载过可直接回车只做校验）：
if "%TOKEN%"=="" (
  "%~dp0envs\python.exe" "%~dp0app\winpack\download_weights.py" --verify-only
) else (
  "%~dp0envs\python.exe" "%~dp0app\winpack\download_weights.py" --hf-token %TOKEN%
)
pause
""",
    "download_weights_cn.bat": r"""@echo off
chcp 65001 >nul
set PYTHONUTF8=1
echo 国内网络版：HF 走 hf-mirror.com、GitHub 走 ghfast.top（都是第三方镜像）。
echo 注意：你的 HF token 会发给 hf-mirror.com（下载 gated 权重必须带 token）。不放心就用 download_weights.bat 直连。
set /p TOKEN=粘贴 HF token 后回车（已下载过可直接回车只做校验）：
if "%TOKEN%"=="" (
  "%~dp0envs\python.exe" "%~dp0app\winpack\download_weights.py" --verify-only
) else (
  "%~dp0envs\python.exe" "%~dp0app\winpack\download_weights.py" --hf-token %TOKEN% --hf-endpoint https://hf-mirror.com --github-proxy https://ghfast.top/ --allow-token-to-mirror
)
pause
""",
}

README_TXT = """FastSAM3DBody 动捕整合包（PoseCapture 插件的第二个身体前端，与 GVHMR 并存）
================================================================================

一、第一次使用
  1. 放到纯英文、尽量短的路径，例如 D:\\PoseCapture_Pack\\FastSAM3DBody
     （和 GVHMR、HAMER、ffmpeg 放在同一个 PoseCapture_Pack 下，插件会自动找到）
  2. 双击 check_env.bat：检查显卡/驱动/依赖/模型，并真跑一帧，最后一行是"全部通过"即可
     - 如果报 DLL 加载失败：先运行 vc_redist.x64.exe（VC++ 运行库）
     - 如果报模型文件缺失：运行 download_weights.bat（需要 HuggingFace token，见窗口提示）
  3. Blender → 编辑 → 偏好设置 → 插件 → PoseCapture → "FastSAM3DBody 路径" 指到本目录
     （本目录在 PoseCapture_Pack 下面时可留空，自动识别）

二、在 Blender 里用
  侧栏「姿态捕捉」→ 设置 → 身体前端 选 "FastSAM3DBody" → 捕捉。
  身体和手一次出（不再跑 HAMER/WiLoR）；结果文件夹结构与 GVHMR+HAMER 完全相同，之后的重定向/修复照常用。

三、不开 Blender 单独跑
  把视频拖到 run_video.bat 上。结果：outputs\\<视频名>\\（gvhmr\\、hamer\\、merged.pkl、<视频名>.pkl）。
  中途关掉窗口，再拖一次同一个视频会从断点续跑（outputs\\_work\\ 里是中间结果，成功后可删）。

四、要求
  NVIDIA 显卡，建议 ≥ 8 GB 显存；驱动 ≥ 528（RTX 50 系要用 cu128 版整合包，驱动 ≥ 570）；内存 ≥ 16 GB。
  速度参考：RTX 4080 约 0.4–0.47 秒/帧（1000 帧约 7–8 分钟）。

五、说明
  - 画面里有路人时自动锁定主角（最大、最居中、持续时间最长的那条轨迹，路人片段会被排除）。
  - 机位需固定（与 GVHMR 的"静态相机"相同）；地面由脚的位置自动拟合。
  - 模型许可：SAM 3D Body 权重为 Meta SAM License（gated），整合包不附带权重，由使用者自行申请下载。
"""


def step_launchers(a):
    if a.dry_run:
        say(f"  写 {', '.join(LAUNCHERS)}、使用说明.txt → {a.out}")
        return
    os.makedirs(a.out, exist_ok=True)
    for name, body in LAUNCHERS.items():
        with open(os.path.join(a.out, name), "w", encoding="utf-8", newline="\r\n") as fh:
            fh.write(body)
    with open(os.path.join(a.out, "使用说明.txt"), "w", encoding="utf-8-sig", newline="\r\n") as fh:
        fh.write(README_TXT)
    os.makedirs(os.path.join(a.out, "outputs"), exist_ok=True)
    say(f"  ✓ {len(LAUNCHERS)} 个 .bat + 使用说明.txt")


def dir_size(p):
    return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(p) for f in fs
               if not os.path.islink(os.path.join(r, f)))


def step_check(a):
    out, problems = a.out, []
    need = ["envs/python.exe", "envs/python310._pth", "app/Fast-SAM-3D-Body/sam_3d_body/__init__.py",
            "app/Fast-SAM-3D-Body/tools/build_fov_estimator.py", "app/fsb/pipeline.py", "app/fsb/assets/mhr_rest.npz",
            "app/winpack/run_video.py", "app/third_party/MoGe/moge/model/v2.py", "run_video.bat"]
    for rel in need:
        if not os.path.isfile(os.path.join(out, rel)):
            problems.append(f"缺 {rel}")
    pth = os.path.join(out, "envs", "python310._pth")
    if os.path.isfile(pth) and "\nimport site" not in "\n" + open(pth, encoding="utf-8").read():
        problems.append("python310._pth 没打开 import site")
    sp = os.path.join(out, "envs", "Lib", "site-packages")
    installed = {}
    for d in glob.glob(os.path.join(sp, "*.dist-info")):
        n, v = os.path.basename(d)[:-10].rsplit("-", 1)
        installed[n.lower().replace("-", "_")] = v
    binary, build = read_lock(a.profile)
    for spec in binary:
        n, v = spec.split("==")
        got = installed.get(n.lower().replace("-", "_").replace(".", "_"))
        if got != v:
            problems.append(f"{n}: 锁定 {v}，装的是 {got}")
    sizes = {k: dir_size(os.path.join(out, k)) for k in ("envs", "app", "checkpoints") if os.path.isdir(os.path.join(out, k))}
    info = {"built": time.strftime("%Y-%m-%d %H:%M:%S"), "profile": a.profile, "profile_note": PROFILES[a.profile]["note"],
            "python": PY_VERSION, "fsb_commit": FSB_COMMIT, "moge_commit": MOGE_COMMIT, "packages": installed,
            "sizes_bytes": sizes, "problems": problems}
    if not a.dry_run and os.path.isdir(out):
        json.dump(info, open(os.path.join(out, "BUILD_INFO.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    say("  " + "，".join(f"{k} {human(v)}" for k, v in sizes.items()))
    say("  ✓ 核对通过" if not problems else "  ✗ " + "\n  ✗ ".join(problems[:30]))
    return not problems


def step_archive(a):
    z = shutil.which("7z") or shutil.which("7za") or r"C:\Program Files\7-Zip\7z.exe"
    dst = os.path.join(os.path.dirname(os.path.abspath(a.out)), f"FastSAM3DBody_{a.profile}_{time.strftime('%Y%m%d')}.7z")
    run([z, "a", "-t7z", "-mx=5", "-mmt=on", dst, a.out, "-xr!_cache", "-xr!wheelhouse", "-xr!outputs",
         "-xr!__pycache__"], a.dry_run)


# ------------------------------------------------------------------ 大小估算（只查元数据）

def estimate(a):
    import urllib.parse
    binary, build = read_lock(a.profile)
    total, rows = 0, []
    for spec in binary:
        n, v = spec.split("==")
        size, fname = None, None
        try:
            if "+cu" in v:
                url = PROFILES[a.profile]["torch_index"] + f"/{n}/"
                html = urlopen(url, timeout=30).read().decode()
                cand = [m for m in re.findall(r'href="([^"]+\.whl)[^"]*"', html)
                        if f"{n}-{urllib.parse.quote(v)}-cp310-cp310-win_amd64" in m or f"{n}-{v}-cp310-cp310-win_amd64" in m]
                if cand:
                    u = urllib.parse.urljoin(url, cand[0])
                    size = int(urlopen(u, method="HEAD", timeout=30).headers["Content-Length"])
                    fname = os.path.basename(urllib.parse.unquote(cand[0].split("#")[0]))
            else:
                d = json.load(urlopen(f"https://pypi.org/pypi/{n}/{v}/json", timeout=30))
                for f in d["urls"]:
                    fn = f["filename"]
                    if fn.endswith(".whl") and (("cp310" in fn and "win_amd64" in fn) or "abi3-win_amd64" in fn or "none-any" in fn):
                        size, fname = f["size"], fn
                        break
        except Exception as exc:
            fname = f"查询失败 {exc!r}"[:80]
        rows.append((n, v, size, fname))
        total += size or 0
    rows.sort(key=lambda r: -(r[2] or 0))
    say(f"wheel 下载合计 {human(total)}（{len(rows)} 个；另有 {len(build)} 个纯 Python 包现场打 wheel，各 < 1 MB）")
    for n, v, s, f in rows[:12]:
        say(f"  {human(s) if s else '?':>10s}  {n}=={v}  {f}")
    say(f"嵌入式 Python {human(PY_SIZE)}；模型文件见 download_weights.py --dry-run（约 2.78 GB）；VC++ 运行库 24.4 MB")
    return total


def main(argv=None):
    ap = argparse.ArgumentParser(description="组装 FastSAM3DBody Windows 整合包")
    ap.add_argument("--out", required=True, help="整合包目录（建议 <PoseCapture_Pack>/FastSAM3DBody）")
    ap.add_argument("--profile", choices=sorted(PROFILES), default="cu124")
    ap.add_argument("--steps", default=",".join(STEPS))
    ap.add_argument("--cache", default=None, help="下载缓存（默认 <out>/../_fsb_build_cache）")
    ap.add_argument("--wheelhouse", default=None, help="wheel 目录（默认 <cache>/wheelhouse_<profile>）")
    ap.add_argument("--fsb-src", default=None, help="Fast-SAM-3D-Body 源码目录（默认用仓库里的云端快照 tgz）")
    ap.add_argument("--moge-src", default=None, help="MoGe 源码目录（默认按固定提交从 GitHub 下载 zip）")
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""))
    ap.add_argument("--pip-index", default="", help="PyPI 镜像，如 https://pypi.tuna.tsinghua.edu.cn/simple")
    ap.add_argument("--github-proxy", default="", help="GitHub 下载前缀代理")
    ap.add_argument("--cn", action="store_true", help="国内网络：torch 走阿里云镜像、HF 走 hf-mirror、Python 走华为云")
    ap.add_argument("--no-slim", action="store_true", help="保留 torch 的 .lib 和头文件（默认删，省 ~0.8 GB）")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--estimate", action="store_true", help="只估算下载大小（查元数据）")
    a = ap.parse_args(argv)
    a.out = os.path.abspath(a.out)
    a.cache = os.path.abspath(a.cache or os.path.join(os.path.dirname(a.out), "_fsb_build_cache"))
    a.wheelhouse = os.path.abspath(a.wheelhouse or os.path.join(a.cache, f"wheelhouse_{a.profile}"))
    if a.cn and not a.pip_index:
        a.pip_index = "https://pypi.tuna.tsinghua.edu.cn/simple"
    say(f"整合包 {a.out}（{a.profile}：{PROFILES[a.profile]['note']}）")
    if a.estimate:
        estimate(a)
        return 0
    steps = [s.strip() for s in a.steps.split(",") if s.strip()]
    bad = [s for s in steps if s not in STEPS]
    if bad:
        say(f"未知步骤 {bad}；可选 {STEPS}")
        return 2
    ok = True
    for s in STEPS:
        if s in steps:
            say(f"\n== {s}")
            r = globals()[f"step_{s}"](a)
            ok = ok and (r is not False)
    say("\n完成。" if ok else "\n有问题（见上）。")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
