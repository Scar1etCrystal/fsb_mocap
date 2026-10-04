#!/usr/bin/env bash
# 阶段 B ②：FastSAM3DBody 借用 HAMER 环境（py3.11 / torch 2.5.1 / numpy 2.4.4）。
# HAMER 环境里一个包都不装、不升级；FastSAM3DBody 缺的包装进独立目录 $OV，运行时放到 PYTHONPATH 最前面。
# 用法：bash setup_fsb_overlay.sh   （先跑过 setup_hamer_env.sh）
set -euo pipefail
PY=/root/autodl-tmp/envs/hamer311/bin/python
OV=/root/autodl-tmp/fsb_overlay_py311
LOGD=/root/autodl-tmp/logs
export PIP_INDEX_URL=${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}
# 与 Windows 整合包 requirements_win.in 同版本；HAMER 环境已有的包（torch、numpy、opencv、timm、lightning…）不重装
WANT="ultralytics==8.4.172 ultralytics-thop==2.2.2 lap==0.5.13 roma==1.6.1 polars==1.44.2 psutil==5.9.8"
# 正常解析（不用 --ignore-installed：那样会按最新 torch 解析，把 CUDA 13 的 nvidia-* 全拉进来，3.4 GB）：
# HAMER 环境已满足的依赖不动，报告里只剩真正缺的包；要求升级已装包的条目由下面过滤掉并记日志
"$PY" -m pip install --dry-run --quiet --report "$LOGD/fsb_overlay_report.json" $WANT
"$PY" - "$LOGD/fsb_overlay_report.json" > "$LOGD/fsb_overlay_list.txt" <<'PY'
import json, sys
from importlib import metadata
have = {d.metadata["Name"].lower().replace("_", "-") for d in metadata.distributions()}
skip = {"opencv-python"}          # HAMER 环境（Windows）装的是 opencv-python；云端是 headless，同一个 cv2，不重复装
for it in json.load(open(sys.argv[1]))["install"]:
    n, v = it["metadata"]["name"], it["metadata"]["version"]
    key = n.lower().replace("_", "-")
    if key in have or key in skip:
        print(f"# 已在 HAMER 环境（不装）: {n}=={v}", file=sys.stderr)
        continue
    print(f"{n}=={v}")
PY
cat "$LOGD/fsb_overlay_list.txt"
rm -rf "$OV" && mkdir -p "$OV"
"$PY" -m pip install --quiet --no-deps --target "$OV" -r "$LOGD/fsb_overlay_list.txt"
# MoGe-2 本体和它的两个纯 Python 依赖（GitHub 提交，云端 pip 拉不动）：从已验证的 FastSAM3DBody 环境原样拷
SP310=/root/autodl-tmp/envs/fast_sam_3d_body/lib/python3.10/site-packages
cp -r "$SP310/utils3d" "$SP310/pipeline" "$SP310/moge" "$OV/"
du -sh "$OV"
echo FSB_OVERLAY_OK
