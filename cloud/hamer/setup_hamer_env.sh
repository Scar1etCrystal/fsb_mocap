#!/usr/bin/env bash
# 云端（Linux）搭一个和用户 HAMER 环境同版本的环境：py3.11 / torch 2.5.1 / numpy 2.4.4 / scipy 1.17.1。
# 不装 detectron2（检人改用锁定的主角框）。用法：bash setup_hamer_env.sh  （日志 /root/autodl-tmp/logs/hamer_env.log）
set -euo pipefail
ENV=/root/autodl-tmp/envs/hamer311
CODE=/root/autodl-tmp/HAMER_code
if [ ! -x "$ENV/bin/python" ]; then
    # 只用清华镜像：.condarc 里的 defaults（repo.anaconda.com）从云端连得极慢，会卡在 Collecting package metadata
    /root/miniconda3/bin/conda create -y -q -p "$ENV" --override-channels -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main python=3.11
fi
PY="$ENV/bin/python"
# pip 用清华源：/etc/pip.conf 里的阿里云源在云端实测只有 ~1.8 MB/s（清华 ~12 MB/s）
export PIP_INDEX_URL=${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}
"$PY" -m pip install -q --upgrade pip
"$PY" -m pip install -q torch==2.5.1 torchvision==0.20.1
"$PY" -m pip install -q -r "$(dirname "$0")/requirements_hamer_linux.txt"
# mmcv / mmengine 的元数据会把 opencv-python 5.x 拉进来、盖掉 cv2 → 换回与用户环境同版本的 4.13.0.92（headless，同一个 cv2）
"$PY" -m pip uninstall -y -q opencv-python || true
"$PY" -m pip install -q --force-reinstall --no-deps opencv-python-headless==4.13.0.92
# xtcocotools 1.14.3 的 Linux 预编译包是按 numpy 1.x 编的（import 报 numpy.dtype size changed）→ 按 numpy 2.4.4 从源码重编
"$PY" -m pip install -q --force-reinstall --no-deps --no-binary xtcocotools --no-build-isolation xtcocotools==1.14.3
# mmcv 1.7.2（用户包里打过补丁的源码）：用户环境是 -e 可编辑安装；这里同样直接挂源码目录（.pth），不编译 CUDA 算子
# （ViTPose 推理用不到）。不用 pip install：它的 setup.py 要 pkg_resources，新版 setuptools 已经没有了
SP=$("$PY" -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')
echo "$CODE/third-party/mmcv1-src-patched" > "$SP/mmcv1_src_patched.pth"
# chumpy 0.70（MANO 的 pkl 要靠它反序列化）：setup.py 在新 pip 下装不上，import 在 numpy 2 / py3.11 下也失败
# （numpy 别名、inspect.getargspec）→ 取源码包、打两处补丁、直接放进 site-packages（纯 Python）
if [ ! -d "$SP/chumpy" ]; then
    # pip download 也会跑它的 setup.py（同样失败）→ 直接从镜像索引取源码包
    T=$(mktemp -d)
    "$PY" - "$T" <<'PY'
import re, sys, urllib.parse, urllib.request
idx = "https://pypi.tuna.tsinghua.edu.cn/simple/chumpy/"
href = re.search(r'href="([^"#]*chumpy-0\.70\.tar\.gz)', urllib.request.urlopen(idx, timeout=60).read().decode()).group(1)
urllib.request.urlretrieve(urllib.parse.urljoin(idx, href), sys.argv[1] + "/chumpy-0.70.tar.gz")
PY
    tar xzf "$T"/chumpy-0.70.tar.gz -C "$T"; cp -r "$T"/chumpy-0.70/chumpy "$SP/"; rm -rf "$T"
fi
sed -i 's/^from numpy import bool, int, float, complex, object, unicode, str, nan, inf$/from numpy import nan, inf/' "$SP/chumpy/__init__.py"
sed -i 's/inspect.getargspec(/inspect.getfullargspec(/' "$SP/chumpy/ch.py"
"$PY" - <<'PY'
import sys, torch, numpy, scipy, cv2, chumpy, mmcv, smplx, timm, pytorch_lightning
print(sys.version.split()[0], torch.__version__, torch.version.cuda, torch.cuda.is_available(), numpy.__version__, scipy.__version__, cv2.__version__, mmcv.__version__)
PY
echo HAMER_ENV_OK
