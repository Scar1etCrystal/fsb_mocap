"""一段 take：云端 FastSAM3DBody 原始输出 → SMPL-X 序列（中间格式）→ 合同目录。

输入目录（云端 run_take.py + recompute_skel.py 的产物）：
  sam3d_part_*.npz  逐帧输出（pred_cam_t / pred_keypoints_3d … 在 OpenCV 相机系）
  skel_part_*.npz   用最终参数重算的 MHR 骨架（joint_rots_model / joint_coords_model，模型系）
  main_track.npz    主角框 + 每帧是否真检测到 + YOLO 17 点（手腕置信度）
  cam.json          固定内参
步骤：
  1. 无效帧（片头片尾没有人，如 999 段最后 5 帧黑场）→ 沿用最近有效帧（手标为未检测）
  2. 骨骼重定向（fsb.retarget）→ SMPL-X 全局/局部旋转（模型系）
  3. 根：相机系朝向 = F·G_root（模型系→OpenCV），骨盆位置 = cam_t + 根关节坐标
  4. 时间处理：局部旋转/根朝向 去尖刺 + 切空间高斯；根位置高斯（深度方向更强）
  5. 世界系：脚跟/脚尖关键点拟合地面定重力 → GVHMR 约定（fsb.world）
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np

from . import retarget, rot, smooth, world

KP = {"l_heel": 17, "l_big_toe": 15, "l_small_toe": 16, "r_heel": 20, "r_big_toe": 18, "r_small_toe": 19,
      "neck": 69, "l_hip": 9, "r_hip": 10}
SMPLX_J0 = np.array([0.001, -0.351, 0.012])        # 占位：真实值由 SMPL-X 模型给（convert_take 里覆盖）


def load_take(raw_dir):
    parts = sorted(glob.glob(os.path.join(glob.escape(raw_dir), "sam3d_part_*.npz")))   # 目录名里的 [..] 不能当通配
    skels = sorted(glob.glob(os.path.join(glob.escape(raw_dir), "skel_part_*.npz")))
    if len(parts) != len(skels):
        raise RuntimeError(f"{raw_dir}: sam3d 分块 {len(parts)} 个、重算骨架 {len(skels)} 个——先跑 recompute_skel.py")
    cat = lambda files, key: np.concatenate([np.load(f)[key] for f in files])
    d = {k: cat(parts, k) for k in ("pred_cam_t", "pred_keypoints_3d", "pred_keypoints_2d", "focal_length", "bbox",
                                     "lhand_bbox", "rhand_bbox", "times")}
    d["R_mhr"] = cat(skels, "joint_rots_model").astype(np.float64)
    d["P_model"] = cat(skels, "joint_coords_model").astype(np.float64)
    mt = np.load(os.path.join(raw_dir, "main_track.npz"))
    d["detected"] = mt["detected"].astype(bool)
    d["yolo_kpts"] = mt["kpts"]
    d["cam"] = json.load(open(os.path.join(raw_dir, "cam.json")))
    d["report"] = json.load(open(os.path.join(raw_dir, "main_track.json")))
    F = d["R_mhr"].shape[0]
    if len(d["detected"]) != F:
        raise RuntimeError(f"帧数不一致：骨架 {F}，主角轨迹 {len(d['detected'])}")
    return d


def invalid_frames(detected, max_inner_gap=15):
    """片头/片尾连续没检测到 → 无效；中间短缺口（≤ max_inner_gap，框已插值）视为有效。"""
    F = len(detected)
    bad = np.zeros(F, bool)
    idx = np.nonzero(detected)[0]
    if not len(idx):
        return ~bad
    bad[:idx[0]] = True
    bad[idx[-1] + 1:] = True
    t = idx[0]
    while t <= idx[-1]:
        if not detected[t]:
            s = t
            while not detected[t]:
                t += 1
            if t - s > max_inner_gap:
                bad[s:t] = True
        t += 1
    return bad


def _hold(arr, bad):
    """无效帧用最近的有效帧替换（沿时间轴 0）。"""
    out = np.array(arr, copy=True)
    good = np.nonzero(~bad)[0]
    for t in np.nonzero(bad)[0]:
        out[t] = arr[good[np.argmin(np.abs(good - t))]]
    return out


def convert_take(raw_dir, rt: retarget.Retargeter, *, J0=None, sigma_body=1.5, sigma_hand=2.0, sigma_root=1.5,
                 sigma_transl=(2.0, 2.0, 4.0), despike_deg=30.0, wrist_conf=0.3, fps=30.0):
    d = load_take(raw_dir)
    F = d["R_mhr"].shape[0]
    J0 = SMPLX_J0 if J0 is None else np.asarray(J0, dtype=np.float64)
    bad = invalid_frames(d["detected"])
    R_mhr, P_model = _hold(d["R_mhr"], bad), _hold(d["P_model"], bad)
    cam_t = _hold(d["pred_cam_t"].astype(np.float64), bad)
    kp3 = _hold(d["pred_keypoints_3d"].astype(np.float64), bad)

    # 2. 重定向（模型系）
    G = rt.global_rotations(R_mhr, P_model, swing_fix=True)
    L = rt.local_rotations(G)
    bone_err, bone_names = rt.bone_dir_errors(G, P_model)

    # 3. 根：相机系
    G0_cam = retarget.F_FLIP @ G[:, 0]
    root_cam = cam_t + (P_model[:, rt.idx["root"]] @ retarget.F_FLIP.T)
    # 4. 时间处理
    jitter_before = float(np.mean(rot.geodesic_deg(L[1:, 1:22], L[:-1, 1:22])))
    L_s = L.copy()
    L_s[:, 1:22], sp_b = smooth.despike(L[:, 1:22], despike_deg)
    L_s[:, 25:55], sp_h = smooth.despike(L[:, 25:55], 45.0)
    L_s[:, 1:22] = smooth.gaussian(L_s[:, 1:22], sigma=sigma_body)
    L_s[:, 25:55] = smooth.gaussian(L_s[:, 25:55], sigma=sigma_hand)
    G0_s, sp_r = smooth.despike(G0_cam, despike_deg)
    G0_s = smooth.gaussian(G0_s, sigma=sigma_root)
    root_s = np.stack([smooth.gaussian_vec(root_cam[:, k], sigma=s) for k, s in enumerate(sigma_transl)], 1)
    jitter_after = float(np.mean(rot.geodesic_deg(L_s[1:, 1:22], L_s[:-1, 1:22])))

    go_cam = rot.mat_to_rotvec(G0_s)
    transl_cam = root_s - J0
    # 5. 世界系：脚跟/脚尖关键点（相机系）拟合地面
    feet = kp3[:, [KP[k] for k in ("l_heel", "l_big_toe", "l_small_toe", "r_heel", "r_big_toe", "r_small_toe")]] + cam_t[:, None]
    torso = np.stack([root_cam, kp3[:, KP["neck"]] + cam_t], 1)
    pts = np.concatenate([torso, feet], 1)
    use = ~bad
    up, floor_d, ground = world.estimate_up(pts[use], foot_idx=range(2, 8), pelvis_idx=0, neck_idx=1, fps=fps)
    go_w, tr_w, R_wc = world.to_world(go_cam, transl_cam, J0, up, floor_d)

    rv = rot.mat_to_rotvec(L_s)
    # 手：YOLO 手腕置信度（COCO 9 左腕 / 10 右腕），无效帧标未检测
    yk = d["yolo_kpts"]
    conf = np.nan_to_num(np.stack([yk[:, 9, 2], yk[:, 10, 2]], 1), nan=0.0)
    hand_valid = (conf > wrist_conf) & ~bad[:, None]
    seq = {"fps": fps, "body_pose": rv[:, 1:22].reshape(F, 63), "left_hand_pose": rv[:, 25:40].reshape(F, 45),
           "right_hand_pose": rv[:, 40:55].reshape(F, 45), "betas": np.zeros(10),
           "world_global_orient": go_w, "world_transl": tr_w,
           "hand_valid": hand_valid, "hand_conf": conf.astype(np.float32)}
    info = {"frames": F, "invalid_frames": [int(t) for t in np.nonzero(bad)[0]],
            "infer_ms_mean": round(float(np.mean(d["times"][1:]) * 1000), 1),
            "bone_dir_err_deg": {"exact_bones_max": round(float(bone_err[:, [i for i, n in enumerate(bone_names)
                                                                         if not n.startswith(("Pelvis→", "Spine3→", "L_Wrist→", "R_Wrist→"))]].max()), 3)},
            "despiked": {"body_frames": len(sp_b), "hand_frames": len(sp_h), "root_frames": len(sp_r)},
            "body_jitter_deg_per_frame": {"before": round(jitter_before, 3), "after": round(jitter_after, 3)},
            "ground": dict(ground, **world.camera_tilt(up)), "hand_valid_frames": hand_valid.sum(0).tolist(),
            "camera_K": d["cam"]["K"], "selection": d["report"]}
    return seq, info, {"G": G, "L": L, "L_s": L_s, "R_wc": R_wc}
