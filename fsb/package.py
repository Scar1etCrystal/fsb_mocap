"""打包器：SMPL-X 序列（中间格式）→ R1 合同 take 目录（PoseCapture 插件 / MoCap Doctor 原样可读）。

中间格式 = dict（或 .npz，键同名），见 docs/格式_SMPLX序列.md：

  fps            float
  body_pose      (F,63)  SMPL-X 21 关节局部轴角（含手腕 19/20）
  left_hand_pose (F,45)  SMPL-X 左手 15 关节局部轴角（真左手表示，插件直接当骨旋转用）
  right_hand_pose(F,45)
  betas          (10,) 或 (F,10)
  world_global_orient (F,3)  世界系：Y 上、重力对齐、Z = 首帧相机视线水平投影（同 GVHMR global）
  world_transl        (F,3)  世界系根位移（米），首帧 XZ = 0
  incam_global_orient (F,3)  可选：相机系；缺省 = 世界系副本（与参考素材一致）
  incam_transl        (F,3)  可选
  hand_valid     (F,2) bool  可选（左, 右），缺省全 True
  hand_conf      (F,2) float 可选 → hand_keypoint_conf / *_hand_keypoint_conf
  hand_frames_original int   可选，写进 hamer_merge_info

手部原始 pkl 与插件"重新合并"（hand_body_merge.py）互逆：合并会做
  hand_global = RZ180 · raw（左手再 M·…·M），wrist_local = elbow_global⁻¹ · hand_global
所以 raw = RZ180 · G_wrist（左手 raw = RZ180 · M·G_wrist·M），G 用 incam 分支正运动学算。
"""
from __future__ import annotations

import json
import os
import pickle
import shutil
from typing import Any

import numpy as np

from . import rot
from .contract import BRANCHES, MERGED_REL, PERSON_REL

RZ180 = np.diag([-1.0, -1.0, 1.0])            # hand_body_merge.py 对 HaMeR 朝向的固定补偿
MERGE_INFO = {"source": "fast_sam_3d_body", "method": "build_hand_pose_arrays", "wrist_mode": "stitch",
              "reference_branch": "incam", "mirror_left_hand": True}


# 协议 4：numpy≥2 写协议 5 会引用 numpy._core.numeric，干净的 numpy 1.26（Blender 4.x 自带）读不了（实测）；
# 协议 4 只引用 numpy(._)core.multiarray，numpy 1.26 / 2.x 都能直接读。插件读 pkl 不在乎协议号。
PICKLE_PROTOCOL = 4


def _f32(a):
    return np.ascontiguousarray(np.asarray(a, dtype=np.float32))


def load_seq(path_or_dict) -> dict:
    if isinstance(path_or_dict, dict):
        return dict(path_or_dict)
    with np.load(path_or_dict, allow_pickle=False) as z:
        seq = {k: z[k] for k in z.files}
    if "fps" in seq:
        seq["fps"] = float(seq["fps"])
    return seq


def check_seq(seq: dict) -> int:
    F = int(np.asarray(seq["body_pose"]).shape[0])
    need = {"body_pose": (F, 63), "left_hand_pose": (F, 45), "right_hand_pose": (F, 45),
            "world_global_orient": (F, 3), "world_transl": (F, 3)}
    for k, shp in need.items():
        a = np.asarray(seq.get(k))
        if a.shape != shp:
            raise ValueError(f"seq.{k}: 形状 {a.shape} ≠ {shp}")
        if not np.isfinite(a).all():
            raise ValueError(f"seq.{k}: 含 NaN/Inf")
    b = np.asarray(seq["betas"])
    if b.shape not in ((10,), (F, 10)):
        raise ValueError(f"seq.betas: 形状 {b.shape} ∉ {{(10,), ({F},10)}}（SMPL-X 只取前 10 个主成分）")
    for k in ("incam_global_orient", "incam_transl"):
        if k in seq and np.asarray(seq[k]).shape != (F, 3):
            raise ValueError(f"seq.{k}: 形状 {np.asarray(seq[k]).shape} ≠ ({F},3)")
    if "hand_valid" in seq and np.asarray(seq["hand_valid"]).shape != (F, 2):
        raise ValueError(f"seq.hand_valid: 形状 ≠ ({F},2)")
    return F


def _branches(seq, F):
    betas = np.asarray(seq["betas"], dtype=np.float64)
    betas = np.tile(betas[None], (F, 1)) if betas.ndim == 1 else betas
    body = {"body_pose": _f32(seq["body_pose"]), "betas": _f32(betas)}
    world = dict(body, global_orient=_f32(seq["world_global_orient"]), transl=_f32(seq["world_transl"]))
    if "incam_global_orient" in seq:
        incam = dict(body, global_orient=_f32(seq["incam_global_orient"]),
                     transl=_f32(seq.get("incam_transl", seq["world_transl"])))
    else:
        incam = {k: v.copy() for k, v in world.items()}      # 参考素材：两分支逐位相同
    order = ("body_pose", "global_orient", "transl", "betas")
    return ({k: world[k] for k in order}, {k: incam[k] for k in order})


def hamer_raw_from_seq(seq: dict, incam: dict, valid: np.ndarray, conf=None) -> dict:
    """手部原始 pkl（MANO / HaMeR 布局），与 hand_body_merge.py 的重新合并互逆。"""
    F = valid.shape[0]
    G = rot.chain_global_rotations(incam["global_orient"], incam["body_pose"])      # (F,22,3,3)
    g_left, g_right = G[:, 20], G[:, 21]
    raw_go = np.zeros((F, 2, 3, 3))
    raw_go[:, 0] = RZ180 @ rot.mirror_lr(g_left)
    raw_go[:, 1] = RZ180 @ g_right
    fl = rot.rotvec_to_mat(np.asarray(seq["left_hand_pose"]).reshape(F, 15, 3))
    fr = rot.rotvec_to_mat(np.asarray(seq["right_hand_pose"]).reshape(F, 15, 3))
    raw_hp = np.zeros((F, 2, 15, 3, 3))
    raw_hp[:, 0] = rot.mirror_lr(fl)
    raw_hp[:, 1] = fr
    abs_lr = np.zeros((F, 2, 3, 3))
    abs_lr[:, 0] = rot.mirror_lr(raw_go[:, 0])          # = hamer_video.py: LEFT_MIRROR @ orient @ LEFT_MIRROR
    abs_lr[:, 1] = raw_go[:, 1]
    inval = ~valid
    raw_go[inval] = 0.0                                  # 参考素材：没检测到的手 = 全零矩阵 + valid 0
    raw_hp[inval] = 0.0
    abs_lr[inval] = 0.0
    out = {"global_orient": _f32(raw_go), "hand_pose": _f32(raw_hp),
           "global_orient_valid": valid.astype(np.uint8),
           "hand_pose_abs_flat": np.zeros((F, 2, 15, 3, 3), dtype=np.float32),   # 参考素材同为全零
           "global_orient_abs_lr": _f32(abs_lr)}
    if conf is not None:
        out["hand_keypoint_conf"] = _f32(conf)
    return out


def merged_from(world, incam, seq, raw, valid, conf=None, hand_frames_original=None) -> dict:
    F = valid.shape[0]
    # *_hand_global_orient：hand_body_merge.py 写的是 RZ180·raw 的轴角（左手不镜像）
    go_aa = rot.mat_to_rotvec(np.einsum("ij,fhjk->fhik", RZ180, np.asarray(raw["global_orient"], dtype=np.float64)))
    go_aa[~valid] = 0.0
    out = {}
    for name, br in (("smpl_params_global", world), ("smpl_params_incam", incam)):
        d = {k: v.copy() for k, v in br.items()}
        d["left_hand_pose"] = _f32(seq["left_hand_pose"])
        d["right_hand_pose"] = _f32(seq["right_hand_pose"])
        d["left_hand_global_orient"] = _f32(go_aa[:, 0])
        d["right_hand_global_orient"] = _f32(go_aa[:, 1])
        d["left_hand_is_detected"] = valid[:, 0].astype(bool).copy()
        d["right_hand_is_detected"] = valid[:, 1].astype(bool).copy()
        if conf is not None:
            d["left_hand_keypoint_conf"] = _f32(conf[:, 0])
            d["right_hand_keypoint_conf"] = _f32(conf[:, 1])
        out[name] = d
    info = dict(MERGE_INFO)
    info.update({"gvhmr_frames": int(F),
                 "hamer_frames_original": int(hand_frames_original if hand_frames_original else F),
                 "merged_frames": int(F)})
    out["hamer_merge_info"] = info
    return out


def package_take(seq, out_dir: str, *, take_name: str | None = None, hamer_stem: str | None = None,
                 source_video: str | None = None, overwrite: bool = False, extra_meta: dict | None = None) -> dict:
    """写出合同目录。返回写了哪些文件。out_dir = take 目录（目录名即 take 名）。"""
    seq = load_seq(seq)
    F = check_seq(seq)
    out_dir = os.path.abspath(out_dir)
    take = take_name or os.path.basename(out_dir.rstrip(os.sep))
    if os.path.exists(out_dir) and os.listdir(out_dir) and not overwrite:
        raise FileExistsError(f"{out_dir} 已存在且非空（overwrite=True 才覆盖）")
    if overwrite and os.path.isdir(out_dir):                     # 覆盖：先删上次写的合同文件（换了 stem / take 名时
        import glob                                                # 旧的 *_hamer_data.pkl 和顶层 pkl 会留下，合同校验看到两份）
        base = glob.escape(out_dir)
        for f in (glob.glob(os.path.join(base, "hamer", "*_hamer_data.pkl")) + glob.glob(os.path.join(base, "*.pkl"))):
            if "repaired" not in os.path.basename(f):              # repaired = 下游修复工具的产物，不是本函数写的
                os.remove(f)
    os.makedirs(os.path.join(out_dir, "gvhmr"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "hamer"), exist_ok=True)

    world, incam = _branches(seq, F)
    valid = np.asarray(seq.get("hand_valid", np.ones((F, 2), bool)), dtype=bool)
    conf = np.asarray(seq["hand_conf"], dtype=np.float32) if "hand_conf" in seq else None

    person = {"smpl_params_global": world, "smpl_params_incam": incam}
    raw = hamer_raw_from_seq(seq, incam, valid, conf)
    merged = merged_from(world, incam, seq, raw, valid, conf, seq.get("hand_frames_original"))

    files = {}
    p_person = os.path.join(out_dir, PERSON_REL)
    with open(p_person, "wb") as fh:
        pickle.dump(person, fh, protocol=PICKLE_PROTOCOL)
    files["person"] = p_person
    p_hamer = os.path.join(out_dir, "hamer", f"{hamer_stem or take}_hamer_data.pkl")
    with open(p_hamer, "wb") as fh:
        pickle.dump(raw, fh)                                    # 参考素材：默认协议
    files["hamer"] = p_hamer
    p_merged = os.path.join(out_dir, MERGED_REL)
    with open(p_merged, "wb") as fh:
        pickle.dump(merged, fh, protocol=PICKLE_PROTOCOL)
    files["merged"] = p_merged
    p_top = os.path.join(out_dir, take + ".pkl")
    shutil.copyfile(p_merged, p_top)                            # 参考素材：逐字节相同
    files["top"] = p_top
    if source_video:
        dst = os.path.join(out_dir, "gvhmr", "0_input_video.mp4")
        shutil.copyfile(source_video, dst)
        files["video"] = dst
    meta = {"take": take, "frames": F, "fps": float(seq.get("fps", 30.0)), "format": "PoseCapture R1 contract",
            "producer": "fsb.package", "incam": "camera" if "incam_global_orient" in seq else "copy_of_world"}
    meta.update(extra_meta or {})
    with open(os.path.join(out_dir, "fsb_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=1, default=str)
    return files


def seq_from_take(take_dir: str, branch: str = "smpl_params_global") -> dict:
    """反向：从现有合同目录取回中间格式（用于往返测试 / 对比）。"""
    from .contract import load_pkl, take_paths
    p = take_paths(take_dir)
    m = load_pkl(p["merged"])
    g, c = m["smpl_params_global"], m["smpl_params_incam"]
    F = g["body_pose"].shape[0]
    seq = {"fps": 30.0, "body_pose": g["body_pose"], "left_hand_pose": g["left_hand_pose"],
           "right_hand_pose": g["right_hand_pose"], "betas": g["betas"],
           "world_global_orient": g["global_orient"], "world_transl": g["transl"],
           "hand_valid": np.stack([g["left_hand_is_detected"], g["right_hand_is_detected"]], 1)}
    if not (np.array_equal(g["global_orient"], c["global_orient"]) and np.array_equal(g["transl"], c["transl"])):
        seq["incam_global_orient"], seq["incam_transl"] = c["global_orient"], c["transl"]
    if "left_hand_keypoint_conf" in g:
        seq["hand_conf"] = np.stack([g["left_hand_keypoint_conf"], g["right_hand_keypoint_conf"]], 1)
    info = m.get("hamer_merge_info", {})
    if info.get("hamer_frames_original"):
        seq["hand_frames_original"] = int(info["hamer_frames_original"])
    return seq
