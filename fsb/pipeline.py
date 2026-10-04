"""FastSAM3DBody 视频动捕流水线（平台无关：云端 Linux 与 Windows 整合包共用这一份）。

    视频 → YOLO 跟踪 → 锁主角 → 固定内参（MoGe-2）→ 逐帧 FastSAM3DBody → 用最终参数重算 MHR 骨架
         → 骨骼重定向 + 时间处理 + 落地（fsb.convert）→ 合同目录（fsb.package）→ 校验（fsb.contract）

各阶段产物落在 work 目录、可续跑（已完成的阶段/分块跳过）。torch / ultralytics / sam_3d_body 只在用到时导入，
所以 package 阶段（--from-raw）在没有 GPU 环境的机器上也能跑。
阶段与 2026-10-04 云端验证过的 cloud/run_take.py + cloud/recompute_skel.py 逐步对应（同参数、同文件格式）。
"""
from __future__ import annotations

import contextlib
import glob
import io
import json
import os
import sys
import time

import numpy as np

from . import contract, convert, package, retarget, track

PART = 200                                                     # 每块帧数（续跑粒度）
KEEP = {"pred_cam_t": np.float32, "pred_joint_coords": np.float32, "focal_length": np.float32, "global_rot": np.float32,
        "body_pose_params": np.float32, "hand_pose_params": np.float32, "scale_params": np.float32,
        "shape_params": np.float32, "pred_keypoints_3d": np.float32, "pred_keypoints_2d": np.float32,
        "lhand_bbox": np.float32, "rhand_bbox": np.float32, "bbox": np.float32}
J0_SMPLX_NEUTRAL = np.array([0.003120000008493662, -0.35141000151634216, 0.012040000408887863])  # SMPL-X 中性静止骨盆
VIDEO_EXTS = (".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".wmv", ".flv")


class Paths:
    """模型文件位置。fsb_root = Fast-SAM-3D-Body 代码目录；ckpt = 权重目录（默认 fsb_root/checkpoints）。"""

    def __init__(self, fsb_root, ckpt=None, rest_npz=None, yolo=None):
        self.fsb_root = os.path.abspath(fsb_root)
        self.ckpt = os.path.abspath(ckpt or os.path.join(self.fsb_root, "checkpoints"))
        self.model_dir = os.path.join(self.ckpt, "sam-3d-body-dinov3")
        self.model_ckpt = os.path.join(self.model_dir, "model.ckpt")
        self.model_cfg = os.path.join(self.model_dir, "model_config.yaml")
        self.mhr_model = os.path.join(self.model_dir, "assets", "mhr_model.pt")
        self.yolo = yolo or os.path.join(self.ckpt, "yolo", "yolo11m-pose.pt")
        self.rest_npz = rest_npz or os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "mhr_rest.npz")

    def missing(self, stages):
        need = [self.rest_npz]
        if "track" in stages:
            need.append(self.yolo)
        if "fov" in stages or "infer" in stages:
            need += [self.model_ckpt, self.model_cfg, self.mhr_model]
        return [p for p in need if not os.path.isfile(p)]


def log(msg, progress=None):
    """进度行。插件按 '[FSB]' 前缀转发到界面；progress = 0..100。"""
    tag = f"[FSB] {progress:5.1f}% " if progress is not None else "[FSB] "
    print(tag + msg, flush=True)


def video_frames(path, start=0, stop=None):
    import cv2
    cap = cv2.VideoCapture(path)
    i = 0
    while True:
        ok, bgr = cap.read()
        if not ok or (stop is not None and i >= stop):
            break
        if i >= start:
            yield i, bgr
        i += 1
    cap.release()


def video_meta(path):
    """(W, H, fps, 帧数)。帧数逐帧 grab 数出来（容器里记的帧数对 VFR/mkv 不可靠）。"""
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"打不开视频：{path}")
    W, H, fps = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), float(cap.get(cv2.CAP_PROP_FPS))
    n = 0
    while cap.grab():
        n += 1
    cap.release()
    if n == 0:
        raise RuntimeError(f"视频里读不出帧：{path}")
    return W, H, (fps if fps > 1e-3 else 30.0), n


# ---------------------------------------------------------------- 阶段 1：跟踪（YOLO11-Pose + ByteTrack）

def stage_track(video, work, paths, meta, device=0):
    out = os.path.join(work, "track_raw.npz")
    if os.path.isfile(out):
        log("跟踪：已有结果，跳过", 15)
        return
    from ultralytics import YOLO
    model = YOLO(paths.yolo)
    rows = {"frame": [], "xyxy": [], "conf": [], "tid": [], "kpts": []}
    t0 = time.time()
    for fi, r in enumerate(model.track(source=video, stream=True, classes=[0], conf=0.25, iou=0.5,
                                       tracker="bytetrack.yaml", verbose=False, device=device)):
        if fi % 100 == 0:
            log(f"跟踪 {fi}/{meta[3]}", 2 + 13 * fi / max(1, meta[3]))
        b = r.boxes
        if b is None or len(b) == 0:
            continue
        ids = b.id.int().cpu().numpy() if b.id is not None else -np.ones(len(b), int)
        kp = r.keypoints.data.cpu().numpy() if r.keypoints is not None else np.zeros((len(b), 17, 3))
        for k in range(len(b)):
            rows["frame"].append(fi); rows["xyxy"].append(b.xyxy[k].cpu().numpy()); rows["conf"].append(float(b.conf[k]))
            rows["tid"].append(int(ids[k])); rows["kpts"].append(kp[k])
    arrays = {"frame": np.asarray(rows["frame"], int), "xyxy": np.asarray(rows["xyxy"], np.float32).reshape(-1, 4),
              "conf": np.asarray(rows["conf"], np.float32), "tid": np.asarray(rows["tid"], int),
              "kpts": np.asarray(rows["kpts"], np.float32).reshape(-1, 17, 3)}
    np.savez_compressed(out, **arrays, W=meta[0], H=meta[1], fps=meta[2], F=meta[3])
    log(f"跟踪完成：{len(arrays['frame'])} 个检测，{len(set(arrays['tid'].tolist()))} 条轨迹，{time.time() - t0:.0f}s", 15)


# ---------------------------------------------------------------- 阶段 2：锁主角（路人排除）

def stage_select(work):
    out = os.path.join(work, "main_track.npz")
    if os.path.isfile(out):
        return json.load(open(os.path.join(work, "main_track.json"), encoding="utf-8"))
    z = np.load(os.path.join(work, "track_raw.npz"))
    W, H, F = int(z["W"]), int(z["H"]), int(z["F"])
    if len(z["frame"]) == 0:
        raise RuntimeError("整段视频没有检测到人")
    dets, ids, kps = [], [], []
    for t in range(F):
        m = z["frame"] == t
        dets.append(np.concatenate([z["xyxy"][m].reshape(-1, 4), z["conf"][m].reshape(-1, 1)], 1))
        ids.append(z["tid"][m]); kps.append(z["kpts"][m].reshape(-1, 17, 3))
    r = track.select_main_subject(dets, W, H, ids)
    kp_main = np.full((F, 17, 3), np.nan)                      # 每帧与主角框最匹配那个检测的 17 点（手腕置信度判手可见）
    det_iou = np.zeros(F)
    for t in range(F):
        if len(dets[t]):
            iou = track.iou_xyxy(r["bbox_xyxy"][t], dets[t][:, :4])[0]
            k = int(np.argmax(iou)); det_iou[t] = iou[k]
            if iou[k] > 0.5:
                kp_main[t] = kps[t][k]
    np.savez_compressed(out, bbox_xyxy=r["bbox_xyxy"], detected=r["detected"], source_tracklet=r["source_tracklet"],
                        kpts=kp_main, det_iou=det_iou)
    with open(os.path.join(work, "main_track.json"), "w", encoding="utf-8") as fh:
        json.dump(r["report"], fh, indent=1)
    rep = r["report"]
    log(f"锁主角：覆盖率 {rep['coverage']}，接段 {len(rep['links'])}，其他人轨迹 {len(rep['others_top'])}，"
        f"交叉风险段 {len(rep['risk_frames'])}", 17)
    return rep


# ---------------------------------------------------------------- 模型

def _patch_utils3d():
    """MoGe 新版调 utils3d.pt.*；云端验证的 utils3d（ca3e9ac）同名函数在 utils3d.torch（两个用到的函数已核对）。"""
    try:
        import utils3d
        if not hasattr(utils3d, "pt"):
            import utils3d.torch as _t
            sys.modules.setdefault("utils3d.pt", _t)
            utils3d.pt = _t
    except Exception:
        pass


def build_model(paths):
    if paths.fsb_root not in sys.path:
        sys.path.insert(0, paths.fsb_root)
    from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body
    with contextlib.redirect_stdout(io.StringIO()):
        model, cfg = load_sam_3d_body(paths.model_ckpt, device="cuda", mhr_path=paths.mhr_model)
    est = SAM3DBodyEstimator(sam_3d_body_model=model, model_cfg=cfg, human_detector=None, human_segmentor=None,
                             fov_estimator=None)
    return est, model


def build_fov(paths):
    if paths.fsb_root not in sys.path:
        sys.path.insert(0, paths.fsb_root)
    _patch_utils3d()
    from tools.build_fov_estimator import FOVEstimator
    with contextlib.redirect_stdout(io.StringIO()):
        return FOVEstimator(name="moge2", device="cuda")


# ---------------------------------------------------------------- 阶段 3：固定内参

def K_from_f_mm(W, H, f_mm):
    """与 GVHMR create_camera_sensor 相同的换算：按对角线把 35mm 等效焦距换成像素焦距。"""
    f = (W ** 2 + H ** 2) ** 0.5 / (24 ** 2 + 36 ** 2) ** 0.5 * f_mm
    return np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])


def stage_fov(video, work, paths, meta, f_mm=0.0, default_hfov_deg=60.0):
    out = os.path.join(work, "cam.json")
    if os.path.isfile(out):
        return json.load(open(out, encoding="utf-8"))
    import cv2
    W, H, F = meta[0], meta[1], meta[3]
    Ks, err = [], None
    if f_mm and f_mm > 0:                                       # 用户给了焦距：不跑 MoGe
        K, src = K_from_f_mm(W, H, f_mm), f"f_mm={f_mm:g}"
    else:
        try:
            fov = build_fov(paths)
            pick = set(np.linspace(0, F - 1, 15).astype(int).tolist())
            for fi, bgr in video_frames(video):
                if fi in pick:
                    with contextlib.redirect_stdout(io.StringIO()):
                        Kf = fov.get_cam_intrinsics(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                    Kf = Kf.cpu().numpy() if hasattr(Kf, "cpu") else np.asarray(Kf)
                    Ks.append(Kf.reshape(3, 3))
            del fov
        except Exception as exc:                                # MoGe 不可用：退回默认视角（只影响深度尺度/落地）
            err = repr(exc)
        if Ks:
            K, src = np.median(np.array(Ks), 0), "moge2"
        else:
            f = (W / 2) / np.tan(np.radians(default_hfov_deg) / 2)
            K, src = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]]), f"default_hfov_{default_hfov_deg:g}deg"
    res = {"K": np.asarray(K).tolist(), "source": src, "W": W, "H": H, "error": err,
           "fx_samples": [round(float(k[0, 0]), 1) for k in Ks]}
    if Ks:
        res["fx_spread_pct"] = round(float(np.std([k[0, 0] for k in Ks]) / K[0, 0] * 100), 2)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    log(f"相机内参：fx={K[0, 0]:.1f}（{src}）" + (f"；MoGe 失败已退回默认：{err[:120]}" if err else ""), 20)
    return res


# ---------------------------------------------------------------- 阶段 4：逐帧推理 + 重算骨架

def _recompute_skeleton(model, a, chunk=50):
    """用最终参数重算 MHR 骨架（pred_global_rots 的手部是融合前旧值，2026-10-04 实测差到 115°）。"""
    import torch
    jc_all, jr_all = [], []
    n = len(a["global_rot"])
    for s in range(0, n, chunk):
        t = lambda k: torch.from_numpy(a[k][s:s + chunk].astype(np.float32)).cuda()
        with torch.no_grad():
            _, jc, jr = model.head_pose.mhr_forward(
                global_trans=t("global_rot") * 0, global_rot=t("global_rot"), body_pose_params=t("body_pose_params"),
                hand_pose_params=t("hand_pose_params"), scale_params=t("scale_params"), shape_params=t("shape_params"),
                expr_params=torch.zeros(min(chunk, n - s), 72).cuda(), return_joint_rotations=True,
                return_joint_coords=True)
        jc_all.append(jc.cpu().numpy().astype(np.float32)); jr_all.append(jr.cpu().numpy().astype(np.float32))
    return np.concatenate(jc_all), np.concatenate(jr_all)


def pending_parts(work, F):
    """还没完成（sam3d 块 + 重算骨架块都在才算完成）的分块号。"""
    done = {int(os.path.basename(p)[11:15]) for p in glob.glob(os.path.join(work, "sam3d_part_*.npz"))
            if os.path.isfile(p.replace("sam3d_part_", "skel_part_"))}
    return [p for p in range((F + PART - 1) // PART) if p not in done], done


def pending_stages(work, stages, F, f_mm=0.0):
    """stages 里还真要干活（因而需要模型文件）的那些；已有结果、可续跑跳过的不算。"""
    out = []
    if "track" in stages and not os.path.isfile(os.path.join(work, "track_raw.npz")):
        out.append("track")
    if "fov" in stages and not os.path.isfile(os.path.join(work, "cam.json")) and not (f_mm and f_mm > 0):
        out.append("fov")
    if "infer" in stages and pending_parts(work, F)[0]:
        out.append("infer")
    return out


def needs_gpu(work, stages, F, f_mm=0.0):
    return bool({"fov", "infer"} & set(pending_stages(work, stages, F, f_mm)))


def stage_infer(video, work, paths, meta, keep_vertices=False):
    F = meta[3]
    todo, done = pending_parts(work, F)
    if not todo:
        log("推理：已有结果，跳过", 93)
        return
    import cv2
    import torch
    t_load = time.time()
    est, model = build_model(paths)
    log(f"模型加载 {time.time() - t_load:.0f}s；推理 {F} 帧（续跑：已完成 {len(done)} 块）", 21)
    K = torch.tensor(json.load(open(os.path.join(work, "cam.json"), encoding="utf-8"))["K"], dtype=torch.float32)[None]
    boxes = np.load(os.path.join(work, "main_track.npz"))["bbox_xyxy"]
    keep = dict(KEEP, **({"pred_vertices": np.float16} if keep_vertices else {}))
    buf, times = {k: [] for k in keep}, []
    for fi, bgr in video_frames(video, start=todo[0] * PART, stop=F):
        part = fi // PART
        if part in done:
            continue
        t0 = time.time()
        with contextlib.redirect_stdout(io.StringIO()):
            outs = est.process_one_image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                                         bboxes=boxes[fi][None].astype(np.float32), cam_int=K.clone())
        torch.cuda.synchronize()
        times.append(time.time() - t0)
        o = outs[0]
        for k, dt in keep.items():
            v = o.get(k)
            v = v.detach().cpu().numpy() if hasattr(v, "detach") else (np.zeros(4) if v is None and k.endswith("bbox") else np.asarray(v))
            buf[k].append(np.asarray(v, dtype=dt))
        if fi % 25 == 0:
            log(f"推理 {fi + 1}/{F}（{np.mean(times[-25:]) * 1000:.0f} ms/帧）", 21 + 72 * (fi + 1) / F)
        if (fi + 1) % PART == 0 or fi + 1 == F:
            arrays = {k: np.stack(v) for k, v in buf.items()}
            n = len(arrays["bbox"])
            np.savez(os.path.join(work, f"sam3d_part_{part:04d}.npz"), first=part * PART, last=fi,
                     times=np.array(times[-n:]), **arrays)
            jc, jr = _recompute_skeleton(model, arrays)
            np.savez(os.path.join(work, f"skel_part_{part:04d}.npz"), joint_coords_model=jc, joint_rots_model=jr)
            buf = {k: [] for k in keep}
    if times:
        peak = torch.cuda.max_memory_allocated() / 2 ** 30
        log(f"推理完成：{len(times)} 帧，平均 {np.mean(times) * 1000:.0f} ms/帧，峰值显存 {peak:.1f} GB", 93)


# ---------------------------------------------------------------- 阶段 5：转换 + 打包 + 校验

def stage_package(work, out_take, rest_npz, *, take_name=None, hamer_stem=None, source_video=None, J0=None, fps=None):
    if fps is None:
        tr = os.path.join(work, "track_raw.npz")
        fps = float(np.load(tr)["fps"]) if os.path.isfile(tr) else 30.0
    rt = retarget.from_rest_npz(rest_npz)
    seq, info, _ = convert.convert_take(work, rt, J0=J0_SMPLX_NEUTRAL if J0 is None else J0, fps=fps)
    files = package.package_take(seq, out_take, take_name=take_name, hamer_stem=hamer_stem, source_video=source_video,
                                 overwrite=True, extra_meta={"convert": {k: v for k, v in info.items() if k != "selection"},
                                                             "selection": info.get("selection")})
    rep = contract.validate_take(out_take)
    if not rep["ok"]:
        raise RuntimeError("合同校验失败：" + "; ".join(rep["errors"][:5]))
    log(f"打包完成并通过合同校验：{out_take}", 99)
    return files, info


STAGES = ("track", "select", "fov", "infer", "package")
WORK_ARTIFACTS = ("track_raw.npz", "main_track.npz", "main_track.json", "cam.json", "sam3d_part_*.npz", "skel_part_*.npz")


def video_signature(path, meta):
    """续跑前确认 work 目录里的中间结果属于同一段视频（大小 + 首尾各 4 MB 的 sha1 + 分辨率/帧数）。"""
    import hashlib
    h, size = hashlib.sha1(), os.path.getsize(path)
    with open(path, "rb") as fh:
        h.update(fh.read(4 << 20))
        if size > 8 << 20:
            fh.seek(-(4 << 20), os.SEEK_END)
            h.update(fh.read())
    return {"size": size, "sha1_head_tail": h.hexdigest(), "W": meta[0], "H": meta[1], "frames": meta[3],
            "fps": round(meta[2], 4)}


def _guard_work(work, video, meta):
    sig, p = video_signature(video, meta), os.path.join(work, "video_sig.json")
    old = json.load(open(p, encoding="utf-8")) if os.path.isfile(p) else None
    stale = [f for pat in WORK_ARTIFACTS for f in glob.glob(os.path.join(work, pat))]
    if old != sig and stale:
        log(f"work 目录里是另一段视频（或未记录来源）的中间结果，清掉 {len(stale)} 个文件重算", 1)
        for f in stale:
            os.remove(f)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(sig, fh, indent=1)


def run(video, out_take, paths, *, work, stages=STAGES, take_name=None, hamer_stem=None, f_mm=0.0, copy_video=True,
        keep_vertices=False):
    os.makedirs(work, exist_ok=True)
    meta = video_meta(video)
    log(f"视频 {os.path.basename(video)}：{meta[0]}x{meta[1]}，{meta[2]:.3f} fps，{meta[3]} 帧", 1)
    _guard_work(work, video, meta)
    miss = paths.missing(pending_stages(work, stages, meta[3], f_mm))
    if miss:
        raise FileNotFoundError("缺模型文件（先运行 download_weights）：\n  " + "\n  ".join(miss))
    t0 = time.time()
    if "track" in stages:
        stage_track(video, work, paths, meta)
    if "select" in stages:
        stage_select(work)
    if "fov" in stages:
        stage_fov(video, work, paths, meta, f_mm=f_mm)
    if "infer" in stages:
        stage_infer(video, work, paths, meta, keep_vertices=keep_vertices)
    files = info = None
    if "package" in stages:
        files, info = stage_package(work, out_take, paths.rest_npz, take_name=take_name, hamer_stem=hamer_stem,
                                    source_video=video if copy_video else None, fps=meta[2])
    log(f"全部完成，用时 {time.time() - t0:.0f}s", 100)
    return files, info
