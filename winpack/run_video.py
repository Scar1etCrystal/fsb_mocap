"""FastSAM3DBody 整合包入口：一段视频 → PoseCapture 合同目录（gvhmr/ + hamer/ + merged.pkl + <take>.pkl）。

    envs\\python.exe app\\winpack\\run_video.py --video D:\\clips\\dance.mp4
    envs\\python.exe app\\winpack\\run_video.py --video in.mp4 --out D:\\out\\dance --take-name dance --hamer-stem dance
    envs\\python.exe app\\winpack\\run_video.py --from-raw <云端/旧 work 目录> --out <take 目录>   （只转换+打包，不用 GPU）

进度行以 "[FSB] " 开头（PoseCapture 插件逐行转发）；最后一行 "[FSB] RESULT {json}" 给插件取结果路径。
退出码：0 成功；2 参数/文件问题；3 运行失败。中间结果在 work 目录（默认 outputs/_work/<take>），中断后重跑同一命令会续跑。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import runtime_env  # noqa: E402


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", help="输入视频（mp4/mkv/mov/avi/webm…）")
    src.add_argument("--from-raw", help="已有的原始结果目录（sam3d_part_* + skel_part_* + main_track + cam.json）：只做转换+打包")
    ap.add_argument("--out", help="输出 take 目录（默认 <整合包>/outputs/<视频名>）")
    ap.add_argument("--take-name", help="顶层 pkl 名（默认 = 输出目录名）")
    ap.add_argument("--hamer-stem", help="hamer/<stem>_hamer_data.pkl 的 stem（默认 = 视频名）")
    ap.add_argument("--work", help="中间结果目录（默认 <整合包>/outputs/_work/<take>）")
    ap.add_argument("--stages", default="track,select,fov,infer,package", help="逗号分隔，默认全部")
    ap.add_argument("--f-mm", type=float, default=0.0, help="35mm 等效焦距；>0 时不跑 MoGe（与 GVHMR --f_mm 同换算）")
    ap.add_argument("--fps", type=float, default=None, help="--from-raw 时覆盖帧率（默认读 track_raw.npz）")
    ap.add_argument("--no-copy-video", action="store_true", help="不把视频复制成 gvhmr/0_input_video.mp4")
    ap.add_argument("--keep-vertices", action="store_true", help="保存逐帧网格顶点（调试用，~110 KB/帧）")
    ap.add_argument("--clean-work", action="store_true", help="成功后删除 work 目录")
    ap.add_argument("--pack", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--online", action="store_true", help=argparse.SUPPRESS)
    return ap.parse_args(argv)


def result_line(**kw):
    print("[FSB] RESULT " + json.dumps(kw, ensure_ascii=False), flush=True)


def main(argv=None):
    a = parse(argv)
    L = runtime_env.setup(a.pack, offline=not a.online)
    from fsb import pipeline                                      # noqa: E402（setup 之后才能导入）
    log = pipeline.log

    name_src = a.video or a.from_raw.rstrip("/\\")
    stem = os.path.splitext(os.path.basename(name_src))[0]
    out = os.path.abspath(a.out or os.path.join(L["outputs"], stem))
    take = a.take_name or os.path.basename(out.rstrip("/\\"))
    work = os.path.abspath(a.work or (a.from_raw if a.from_raw else os.path.join(L["outputs"], "_work", take)))
    t0 = time.time()
    try:
        if a.from_raw:
            if not os.path.isdir(a.from_raw):
                log(f"找不到原始结果目录：{a.from_raw}")
                return 2
            tr = os.path.join(work, "track_raw.npz")
            if a.fps:
                fps = a.fps
            else:
                import numpy as np
                fps = float(np.load(tr)["fps"]) if os.path.isfile(tr) else 30.0
            files, info = pipeline.stage_package(work, out, os.path.join(L["app"], "fsb", "assets", "mhr_rest.npz"),
                                                 take_name=take, hamer_stem=a.hamer_stem or take, fps=fps)
        else:
            if not os.path.isfile(a.video):
                log(f"找不到视频：{a.video}")
                return 2
            stages = tuple(s.strip() for s in a.stages.split(",") if s.strip())
            bad = [s for s in stages if s not in pipeline.STAGES]
            if bad:
                log(f"未知阶段：{bad}（可选 {','.join(pipeline.STAGES)}）")
                return 2
            video, copied = runtime_env.ascii_safe_copy(os.path.abspath(a.video), work)
            if copied:
                log(f"路径含非 ASCII 字符，已复制到 {video}")
            paths = pipeline.Paths(L["fsb_src"], ckpt=L["ckpt"])
            meta = pipeline.video_meta(video)
            fps = meta[2]
            os.makedirs(work, exist_ok=True)
            pipeline._guard_work(work, video, meta)               # 先清掉别的视频留下的中间结果，再判断还要干哪些活
            miss = paths.missing(pipeline.pending_stages(work, stages, meta[3], a.f_mm))
            if miss:
                log("缺模型文件，先运行 download_weights.bat：\n  " + "\n  ".join(miss))
                return 2
            if pipeline.needs_gpu(work, stages, meta[3], a.f_mm):
                import torch
                if not torch.cuda.is_available():
                    log("没有可用的 NVIDIA GPU（torch.cuda.is_available() = False）；先运行 check_env.bat 查原因")
                    return 3
                runtime_env.patch_torch_hub(L["dinov3_hub"])
            files, info = pipeline.run(video, out, paths, work=work, stages=stages, take_name=take,
                                       hamer_stem=a.hamer_stem or stem, f_mm=a.f_mm, copy_video=not a.no_copy_video,
                                       keep_vertices=a.keep_vertices)
            if files is None:                                      # 没跑 package 阶段
                result_line(ok=True, take_dir=None, work=work, seconds=round(time.time() - t0, 1))
                return 0
        if a.clean_work and not a.from_raw:
            import shutil
            shutil.rmtree(work, ignore_errors=True)
        result_line(ok=True, take_dir=out, work=None if a.clean_work else work, seconds=round(time.time() - t0, 1),
                    frames=info["frames"], fps=fps, invalid_frames=len(info["invalid_frames"]),
                    infer_ms_mean=info.get("infer_ms_mean"), **{k: v for k, v in files.items()})
        return 0
    except Exception as exc:
        traceback.print_exc()
        log(f"失败：{exc}")
        result_line(ok=False, error=str(exc)[:2000], work=work)
        return 3


if __name__ == "__main__":
    sys.exit(main())
