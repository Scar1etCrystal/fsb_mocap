"""云端：一段视频 → YOLO 跟踪 → 锁主角 → 估相机内参 → 逐帧 Fast SAM 3D Body → MHR 原始输出（npz）。

分阶段、可续跑：--stages track,select,sheet,fov,infer（默认全部）。产物都在 --out 目录：
  track_raw.npz     逐帧全部人体检测（框/置信度/跟踪 ID/17 点）
  main_track.npz/json  主角框（无缺口）+ 筛选报告（风险帧）
  sheet_*.jpg       抽查图：绿 = 主角，红 = 其他人（风险帧 + 每 ~100 帧）
  cam.json          固定内参（多帧中位数，视频里焦距不变）
  sam3d_part_XXXX.npz  逐帧 MHR 输出（每 200 帧一块，可续跑）
"""
import argparse
import contextlib
import glob
import io
import json
import os
import sys
import time

import cv2
import numpy as np

FSB = os.environ.get("FSB_ROOT", "/root/autodl-tmp/Fast-SAM-3D-Body")  # 新流程见 fsb/pipeline.py + winpack/run_video.py
sys.path.insert(0, FSB)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fsb import track  # noqa: E402

PART = 200


def video_meta(path):
    cap = cv2.VideoCapture(path)
    W, H, fps = int(cap.get(3)), int(cap.get(4)), float(cap.get(5))
    n = 0
    while cap.grab():
        n += 1
    return W, H, fps, n


def frames(path, start=0, stop=None):
    cap = cv2.VideoCapture(path)
    i = 0
    while True:
        ok, bgr = cap.read()
        if not ok or (stop is not None and i >= stop):
            break
        if i >= start:
            yield i, bgr
        i += 1


def stage_track(a, meta):
    from ultralytics import YOLO
    model = YOLO(os.path.join(FSB, "checkpoints/yolo/yolo11m-pose.pt"))
    rows = {"frame": [], "xyxy": [], "conf": [], "tid": [], "kpts": []}
    t0 = time.time()
    for fi, r in enumerate(model.track(source=a.video, stream=True, classes=[0], conf=0.25, iou=0.5,
                                       tracker="bytetrack.yaml", verbose=False, device=0)):
        b = r.boxes
        if b is None or len(b) == 0:
            continue
        ids = b.id.int().cpu().numpy() if b.id is not None else -np.ones(len(b), int)
        kp = r.keypoints.data.cpu().numpy() if r.keypoints is not None else np.zeros((len(b), 17, 3))
        for k in range(len(b)):
            rows["frame"].append(fi); rows["xyxy"].append(b.xyxy[k].cpu().numpy()); rows["conf"].append(float(b.conf[k]))
            rows["tid"].append(int(ids[k])); rows["kpts"].append(kp[k])
    np.savez_compressed(os.path.join(a.out, "track_raw.npz"), **{k: np.asarray(v) for k, v in rows.items()},
                        W=meta[0], H=meta[1], fps=meta[2], F=meta[3])
    print(f"[track] {len(rows['frame'])} detections, {len(set(rows['tid']))} ids, {time.time() - t0:.1f}s")


def stage_select(a, meta):
    z = np.load(os.path.join(a.out, "track_raw.npz"))
    W, H, F = int(z["W"]), int(z["H"]), int(z["F"])
    dets = [np.zeros((0, 5)) for _ in range(F)]
    ids = [np.zeros(0, int) for _ in range(F)]
    kps = [np.zeros((0, 17, 3)) for _ in range(F)]
    for t in range(F):
        m = z["frame"] == t
        dets[t] = np.concatenate([z["xyxy"][m], z["conf"][m][:, None]], 1)
        ids[t] = z["tid"][m]
        kps[t] = z["kpts"][m]
    r = track.select_main_subject(dets, W, H, ids)
    # 每帧与主角框最匹配的那个检测的 17 点（手腕置信度用来判手是否可见）
    kp_main = np.full((F, 17, 3), np.nan)
    det_iou = np.zeros(F)
    for t in range(F):
        if len(dets[t]):
            iou = track.iou_xyxy(r["bbox_xyxy"][t], dets[t][:, :4])[0]
            k = int(np.argmax(iou)); det_iou[t] = iou[k]
            if iou[k] > 0.5:
                kp_main[t] = kps[t][k]
    np.savez_compressed(os.path.join(a.out, "main_track.npz"), bbox_xyxy=r["bbox_xyxy"], detected=r["detected"],
                        source_tracklet=r["source_tracklet"], kpts=kp_main, det_iou=det_iou)
    with open(os.path.join(a.out, "main_track.json"), "w") as fh:
        json.dump(r["report"], fh, indent=1)
    rep = r["report"]
    print(f"[select] coverage {rep['coverage']}, gaps {rep['gaps'][:5]}, chain {rep['chain']}, links {len(rep['links'])}, "
          f"others {len(rep['others_top'])}, risk runs {len(rep['risk_frames'])}")


def stage_sheet(a, meta):
    z = np.load(os.path.join(a.out, "track_raw.npz"))
    mt = np.load(os.path.join(a.out, "main_track.npz"))
    rep = json.load(open(os.path.join(a.out, "main_track.json")))
    F = int(z["F"])
    pick = set(range(0, F, max(1, F // 12)))
    for s, e in rep["risk_frames"]:
        pick.update({s, (s + e) // 2, e})
    pick = sorted(p for p in pick if p < F)
    tiles = []
    for fi, bgr in frames(a.video):
        if fi not in pick:
            continue
        img = bgr.copy()
        m = z["frame"] == fi
        for b in z["xyxy"][m]:
            cv2.rectangle(img, tuple(int(v) for v in b[:2]), tuple(int(v) for v in b[2:4]), (0, 0, 255), 3)
        b = mt["bbox_xyxy"][fi]
        cv2.rectangle(img, tuple(int(v) for v in b[:2]), tuple(int(v) for v in b[2:4]), (0, 255, 0), 5)
        risk = any(s <= fi <= e for s, e in rep["risk_frames"])
        cv2.putText(img, f"{fi}{' RISK' if risk else ''}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 255), 6)
        tiles.append(cv2.resize(img, (480, 270)))
    for k in range(0, len(tiles), 16):
        grid = tiles[k:k + 16]
        while len(grid) % 4:
            grid.append(np.zeros_like(tiles[0]))
        rows = [np.concatenate(grid[i:i + 4], 1) for i in range(0, len(grid), 4)]
        cv2.imwrite(os.path.join(a.out, f"sheet_{k // 16:02d}.jpg"), np.concatenate(rows, 0), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"[sheet] {len(tiles)} frames → {(len(tiles) + 15) // 16} sheets")


def build_estimator(with_fov=False):
    import torch
    from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body
    model, cfg = load_sam_3d_body(os.path.join(FSB, "checkpoints/sam-3d-body-dinov3/model.ckpt"), device="cuda",
                                  mhr_path=os.path.join(FSB, "checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt"))
    fov = None
    if with_fov:
        # MoGe v2 调 utils3d.pt.*；环境里 utils3d 1.7 的同名函数在 utils3d.torch（已核对两个函数都在）
        import utils3d
        import utils3d.torch as _u3t
        sys.modules.setdefault("utils3d.pt", _u3t)
        utils3d.pt = _u3t
        from tools.build_fov_estimator import FOVEstimator
        fov = FOVEstimator(name="moge2", device="cuda")
    est = SAM3DBodyEstimator(sam_3d_body_model=model, model_cfg=cfg, human_detector=None, human_segmentor=None,
                             fov_estimator=fov)
    return est, fov


def stage_fov(a, meta):
    _, fov = build_estimator(with_fov=True)
    F = meta[3]
    pick = set(np.linspace(0, F - 1, 15).astype(int).tolist())
    Ks = []
    for fi, bgr in frames(a.video):
        if fi in pick:
            with contextlib.redirect_stdout(io.StringIO()):
                K = fov.get_cam_intrinsics(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            K = K.cpu().numpy() if hasattr(K, "cpu") else np.asarray(K)
            Ks.append(K.reshape(3, 3))
    Ks = np.array(Ks)
    K = np.median(Ks, 0)
    out = {"K": K.tolist(), "fx_samples": Ks[:, 0, 0].round(1).tolist(), "W": meta[0], "H": meta[1],
           "fx_spread_pct": round(float(Ks[:, 0, 0].std() / K[0, 0] * 100), 2)}
    with open(os.path.join(a.out, "cam.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"[fov] K fx={K[0, 0]:.1f} cx={K[0, 2]:.1f} cy={K[1, 2]:.1f}, spread {out['fx_spread_pct']}% over {len(Ks)} frames")


KEEP = {"pred_global_rots": np.float32, "pred_joint_coords": np.float32, "pred_cam_t": np.float32,
        "focal_length": np.float32, "global_rot": np.float32, "body_pose_params": np.float32,
        "hand_pose_params": np.float32, "scale_params": np.float32, "shape_params": np.float32,
        "pred_keypoints_3d": np.float32, "pred_keypoints_2d": np.float32, "pred_vertices": np.float16,
        "lhand_bbox": np.float32, "rhand_bbox": np.float32, "bbox": np.float32}


def stage_infer(a, meta):
    import torch
    est, _ = build_estimator(with_fov=False)
    K = torch.tensor(json.load(open(os.path.join(a.out, "cam.json")))["K"], dtype=torch.float32)[None]
    boxes = np.load(os.path.join(a.out, "main_track.npz"))["bbox_xyxy"]
    F = meta[3] if a.max_frames is None else min(meta[3], a.max_frames)
    done = {int(os.path.basename(p)[11:15]) for p in glob.glob(os.path.join(a.out, "sam3d_part_*.npz"))}
    buf, times, start = {k: [] for k in KEEP}, [], None
    t_all = time.time()
    for fi, bgr in frames(a.video, stop=F):
        part = fi // PART
        if part in done:
            continue
        if start is None:
            start = fi
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        t0 = time.time()
        with contextlib.redirect_stdout(io.StringIO()):
            outs = est.process_one_image(rgb, bboxes=boxes[fi][None].astype(np.float32), cam_int=K.clone())
        torch.cuda.synchronize()
        times.append(time.time() - t0)
        o = outs[0]
        for k, dt in KEEP.items():
            v = o.get(k)
            v = v.detach().cpu().numpy() if hasattr(v, "detach") else (np.zeros(4) if v is None and k.endswith("bbox") else np.asarray(v))
            buf[k].append(np.asarray(v, dtype=dt))
        if (fi + 1) % PART == 0 or fi + 1 == F:
            np.savez(os.path.join(a.out, f"sam3d_part_{part:04d}.npz"), first=part * PART, last=fi,
                     times=np.array(times[-len(buf['bbox']):]), **{k: np.stack(v) for k, v in buf.items()})
            print(f"[infer] part {part} frames {part * PART}..{fi}  {np.mean(times[-len(buf['bbox']):]) * 1000:.0f} ms/frame", flush=True)
            buf = {k: [] for k in KEEP}
    if times:
        print(f"[infer] {len(times)} frames, mean {np.mean(times) * 1000:.0f} ms, p95 {np.percentile(times, 95) * 1000:.0f} ms, wall {time.time() - t_all:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stages", default="track,select,sheet,fov,infer")
    ap.add_argument("--max-frames", type=int, default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    meta = video_meta(a.video)
    print(f"[meta] {os.path.basename(a.video)} {meta[0]}x{meta[1]} fps {meta[2]:.3f} frames {meta[3]}", flush=True)
    for st in a.stages.split(","):
        globals()[f"stage_{st}"](a, meta)


if __name__ == "__main__":
    main()
