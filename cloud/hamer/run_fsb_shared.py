"""阶段 B ②：FastSAM3DBody 借用 HAMER 同版本环境（py3.11 / torch 2.5.1 / numpy 2.4.4 + 补包目录）跑全流程，
与已验证环境（py3.10 / numpy 1.26.4，/root/autodl-tmp/fsb_work/<take>）的中间结果逐项比，报告写 compare.json。

    source fsb_frontend/cloud/env.sh; export PYTHONPATH=<补包目录>:$PYTHONPATH
    /root/autodl-tmp/envs/hamer311/bin/python run_fsb_shared.py [t999 t1499]
"""
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/root/autodl-tmp/fsb_frontend")
from fsb import pipeline  # noqa: E402

FSB_ROOT = "/root/autodl-tmp/Fast-SAM-3D-Body"
OLD = "/root/autodl-tmp/fsb_work"
NEW = "/root/autodl-tmp/fsb_work_py311"
OUT = "/root/autodl-tmp/fsb_out_py311"
VIDEOS = {"t999": "/root/autodl-tmp/videos/0001-0999_f_full_20260925_162525.mkv",
          "t1499": "/root/autodl-tmp/videos/0001-1499_f_full_20260925_165539.mkv"}


def maxdiff(a, b):
    if a.shape != b.shape:
        return f"shape {a.shape} vs {b.shape}"
    if a.dtype.kind not in "fiub":
        return None
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return float(np.nanmax(d)) if d.size else 0.0


def compare(old, new):
    rep = {}
    for name in ("track_raw.npz", "main_track.npz"):
        a, b = np.load(f"{old}/{name}"), np.load(f"{new}/{name}")
        rep[name] = {k: maxdiff(a[k], b[k]) for k in a.files if k in b.files}
    rep["cam.json"] = {"old": json.load(open(f"{old}/cam.json")), "new": json.load(open(f"{new}/cam.json"))}
    for pat in ("sam3d_part_*.npz", "skel_part_*.npz"):
        agg = {}
        for f in sorted(glob.glob(f"{old}/{pat}")):
            g = os.path.join(new, os.path.basename(f))
            if not os.path.isfile(g):
                agg.setdefault("_missing", []).append(os.path.basename(f))
                continue
            a, b = np.load(f), np.load(g)
            for k in a.files:
                if k in b.files:
                    d = maxdiff(a[k], b[k])
                    if isinstance(d, float):
                        agg[k] = max(agg.get(k, 0.0), d)
                    elif d is not None:
                        agg[k] = d
        rep[pat] = agg
    return rep


def main():
    takes = sys.argv[1:] or list(VIDEOS)
    paths = pipeline.Paths(FSB_ROOT)
    # 与 Windows 整合包同一套离线补丁：DINOv3 结构代码从本地目录载入（否则 torch.hub 先联网问 GitHub 默认分支，云端常被断开）
    sys.path.insert(0, "/root/autodl-tmp/fsb_frontend/winpack")
    import runtime_env
    runtime_env.patch_torch_hub("/root/autodl-tmp/torch-hub/hub/facebookresearch_dinov3_main")
    report = {"python": sys.version.split()[0], "numpy": np.__version__}
    import torch
    report["torch"] = torch.__version__
    for take in takes:
        t0 = time.time()
        stem = os.path.splitext(os.path.basename(VIDEOS[take]))[0]
        files, info = pipeline.run(VIDEOS[take], os.path.join(OUT, take), paths, work=os.path.join(NEW, take),
                                   take_name=f"{stem}_fsb_py311", hamer_stem=stem, copy_video=False)
        report[take] = {"seconds": round(time.time() - t0, 1), "frames": info["frames"],
                        "invalid_frames": len(info["invalid_frames"]), "infer_ms_mean": info.get("infer_ms_mean"),
                        "vs_py310": compare(os.path.join(OLD, take), os.path.join(NEW, take))}
        print(json.dumps({take: report[take]}, ensure_ascii=False)[:3000], flush=True)
    with open(os.path.join(NEW, "compare.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)
    print("FSB_SHARED_DONE", flush=True)


if __name__ == "__main__":
    main()
