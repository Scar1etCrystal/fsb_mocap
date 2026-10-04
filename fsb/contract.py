"""R1 输出合同校验器：一个 take 目录是否能被 PoseCapture 插件 / MoCap Doctor 原样消费。

合同（需求文档 §2，已用 reference_output 两段素材逐字段核对）：

  <take>/
  ├ gvhmr/hmr4d_results.pt_person-0.pkl   smpl_params_global / smpl_params_incam，各 4 字段
  ├ hamer/<任意名>_hamer_data.pkl          手部原始（MANO，旋转矩阵，左手为镜像右手表示）
  ├ hamer/merged.pkl                      两分支各 身体 4 字段 + 手部展平字段 + hamer_merge_info
  └ <take>.pkl                            merged 的顶层副本（参考素材里逐字节相同）

用法：python -m fsb.contract <take_dir> [--json]
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import pickle
import sys
from typing import Any

import numpy as np

BRANCHES = ("smpl_params_global", "smpl_params_incam")
PERSON_REL = os.path.join("gvhmr", "hmr4d_results.pt_person-0.pkl")
MERGED_REL = os.path.join("hamer", "merged.pkl")
BODY_FIELDS = {"body_pose": 63, "global_orient": 3, "transl": 3, "betas": 10}
HAND_MERGED_FIELDS = {"left_hand_pose": (45,), "right_hand_pose": (45,),
                      "left_hand_global_orient": (3,), "right_hand_global_orient": (3,),
                      "left_hand_is_detected": (), "right_hand_is_detected": ()}
HAND_MERGED_OPTIONAL = {"left_hand_keypoint_conf": (), "right_hand_keypoint_conf": ()}
HAMER_FIELDS = {"global_orient": (2, 3, 3), "hand_pose": (2, 15, 3, 3), "global_orient_valid": (2,),
                "hand_pose_abs_flat": (2, 15, 3, 3), "global_orient_abs_lr": (2, 3, 3)}
HAMER_OPTIONAL = {"hand_keypoint_conf": (2,)}
MERGE_INFO_KEYS = ("source", "method", "wrist_mode", "reference_branch", "mirror_left_hand",
                   "gvhmr_frames", "hamer_frames_original", "merged_frames")
# 消费端语义依赖的取值：method ∈ 这两个之一 → 插件认为左手已镜像，不再二次翻转
MERGE_INFO_FIXED = {"method": ("build_hand_pose_arrays", "mano2smplx"), "wrist_mode": ("stitch",),
                    "reference_branch": ("incam",), "mirror_left_hand": (True,)}


def load_pkl(path: str) -> Any:
    import warnings
    with open(path, "rb") as fh, warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)   # numpy 1.x 写的 pkl 引用 numpy.core
        return pickle.load(fh, encoding="latin1")      # 与插件 / MoCap Doctor 的读法一致


def take_paths(take_dir: str) -> dict:
    take_dir = os.path.abspath(take_dir)
    name = os.path.basename(take_dir.rstrip(os.sep))
    hamer = sorted(glob.glob(os.path.join(take_dir, "hamer", "*_hamer_data.pkl")))
    # 顶层副本：插件重新合并时写 <目录名>.pkl；MoCap Doctor 回退时取根目录第一个非 repaired 的 *.pkl
    # （999v2 素材目录改过名，顶层文件名 ≠ 目录名——所以按 MoCap Doctor 的规则找）
    tops = [f for f in sorted(glob.glob(os.path.join(take_dir, "*.pkl"))) if "repaired" not in os.path.basename(f)]
    exact = os.path.join(take_dir, name + ".pkl")
    top = exact if os.path.isfile(exact) else (tops[0] if tops else exact)
    return {"take": name, "dir": take_dir,
            "person": os.path.join(take_dir, PERSON_REL),
            "merged": os.path.join(take_dir, MERGED_REL),
            "top": top, "top_candidates": tops,
            "hamer": hamer}


def _sha1(path):
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rot_err(R):
    """|RᵀR − I| 与 |det − 1| 的最大值（只对非零矩阵算）。"""
    R = np.asarray(R, dtype=np.float64).reshape(-1, 3, 3)
    nz = np.abs(R).sum(axis=(1, 2)) > 0
    if not nz.any():
        return 0.0, 0
    R = R[nz]
    orth = np.abs(np.einsum("nji,njk->nik", R, R) - np.eye(3)).max()
    det = np.abs(np.linalg.det(R) - 1.0).max()
    return float(max(orth, det)), int(nz.sum())


class _Report:
    def __init__(self):
        self.errors, self.warnings, self.facts = [], [], {}

    def err(self, msg):
        self.errors.append(msg)

    def warn(self, msg):
        self.warnings.append(msg)

    def as_dict(self):
        return {"ok": not self.errors, "errors": self.errors, "warnings": self.warnings, "facts": self.facts}


def _check_array(rep, where, arr, shape_tail, F, dtypes):
    if not isinstance(arr, np.ndarray):
        rep.err(f"{where}: 不是 ndarray（{type(arr).__name__}）")
        return False
    want = (F,) + tuple(shape_tail) if F is not None else None
    if want is not None and arr.shape != want:
        rep.err(f"{where}: 形状 {arr.shape} ≠ {want}")
        return False
    if F is None and arr.shape[1:] != tuple(shape_tail):
        rep.err(f"{where}: 形状 {arr.shape} 尾部 ≠ {tuple(shape_tail)}")
        return False
    if arr.dtype.name not in dtypes:
        rep.err(f"{where}: dtype {arr.dtype} ∉ {dtypes}")
        return False
    if arr.dtype.kind == "f" and not np.isfinite(arr).all():
        rep.err(f"{where}: 含 NaN/Inf")
        return False
    return True


def validate_take(take_dir: str) -> dict:
    rep = _Report()
    p = take_paths(take_dir)
    rep.facts["take"] = p["take"]
    for key in ("person", "merged"):
        if not os.path.isfile(p[key]):
            rep.err(f"缺文件：{os.path.relpath(p[key], p['dir'])}")
    if not os.path.isfile(p["top"]):
        rep.err(f"缺顶层副本：{p['take']}.pkl（根目录没有任何 *.pkl）")
    else:
        rep.facts["top_file"] = os.path.basename(p["top"])
        if os.path.basename(p["top"]) != p["take"] + ".pkl":
            rep.warn(f"顶层副本名 {os.path.basename(p['top'])} ≠ <目录名>.pkl；MoCap Doctor 能找到（取根目录第一个 pkl），"
                     "插件重新合并时会另写 <目录名>.pkl")
        if len(p["top_candidates"]) > 1:
            rep.warn(f"根目录有 {len(p['top_candidates'])} 个 pkl：{[os.path.basename(x) for x in p['top_candidates']]}")
    if not p["hamer"]:
        rep.err("缺文件：hamer/*_hamer_data.pkl")
    elif len(p["hamer"]) > 1:
        rep.warn(f"hamer/ 下有 {len(p['hamer'])} 个 *_hamer_data.pkl；插件取 os.listdir 顺序里的第一个")
    if rep.errors:
        return rep.as_dict()

    # ---- 身体（person pkl）
    person = load_pkl(p["person"])
    F = None
    for br in BRANCHES:
        if br not in person:
            rep.err(f"person: 缺分支 {br}")
            continue
        for k, n in BODY_FIELDS.items():
            if k not in person[br]:
                rep.err(f"person.{br}: 缺字段 {k}")
                continue
            arr = person[br][k]
            if F is None and isinstance(arr, np.ndarray):
                F = arr.shape[0]
            _check_array(rep, f"person.{br}.{k}", arr, (n,), F, ("float32",))
    rep.facts["frames"] = F
    if F is None:
        return rep.as_dict()
    if all(br in person for br in BRANCHES):
        same = {k: bool(np.array_equal(person[BRANCHES[0]][k], person[BRANCHES[1]][k]))
                for k in BODY_FIELDS if k in person[BRANCHES[0]] and k in person[BRANCHES[1]]}
        rep.facts["person_global_eq_incam"] = same
        lp = {k: bool(np.array_equal(person[BRANCHES[0]][k], person[BRANCHES[1]][k])) for k in ("body_pose",)}
        if not lp["body_pose"]:
            rep.warn("person: 两分支 body_pose 不同——INCAM/BOTH 模式与 GLOBAL 模式的局部姿态会不一致")
    extra = sorted(set(person) - set(BRANCHES))
    if extra:
        rep.facts["person_extra_keys"] = extra

    # ---- 手部原始（hamer_data pkl）
    hd = load_pkl(p["hamer"][0])
    Fh = None
    for k, tail in HAMER_FIELDS.items():
        if k not in hd:
            rep.err(f"hamer_data: 缺字段 {k}")
            continue
        arr = hd[k]
        if Fh is None and isinstance(arr, np.ndarray):
            Fh = arr.shape[0]
        _check_array(rep, f"hamer_data.{k}", arr, tail, Fh,
                     ("uint8",) if k == "global_orient_valid" else ("float32",))
    for k, tail in HAMER_OPTIONAL.items():
        if k in hd:
            _check_array(rep, f"hamer_data.{k}", hd[k], tail, Fh, ("float32",))
    rep.facts["hamer_frames"] = Fh
    if Fh is not None and Fh not in (F, F + 1):
        rep.err(f"hamer_data 帧数 {Fh} 与身体 {F} 相差 > 1")
    elif Fh == F + 1:
        rep.warn(f"hamer_data 比身体多 1 帧（{Fh} vs {F}），合并时按身体帧数裁齐（参考素材 1499 段同样如此）")
    if "global_orient_valid" in hd and isinstance(hd["global_orient_valid"], np.ndarray):
        v = hd["global_orient_valid"].astype(bool)
        rep.facts["hand_valid_frames"] = [int(v[:, 0].sum()), int(v[:, 1].sum())]
        if "global_orient" in hd:
            e, n = _rot_err(hd["global_orient"][v])
            rep.facts["hamer_global_orient_rot_err"] = round(e, 6)
            if e > 1e-3:
                rep.err(f"hamer_data.global_orient 有效帧不是旋转矩阵（误差 {e:.2e}）")
        if "hand_pose" in hd:
            e, n = _rot_err(hd["hand_pose"][v])
            rep.facts["hamer_hand_pose_rot_err"] = round(e, 6)
            if e > 1e-3:
                rep.err(f"hamer_data.hand_pose 有效帧不是旋转矩阵（误差 {e:.2e}）")
        if "global_orient_abs_lr" in hd and "global_orient" in hd:
            from .rot import mirror_lr
            l_ok = v[:, 0]
            if l_ok.any():
                d = np.abs(mirror_lr(hd["global_orient"][l_ok, 0]) - hd["global_orient_abs_lr"][l_ok, 0]).max()
                rep.facts["abs_lr_left_is_mirror"] = bool(d < 1e-4)
                if d >= 1e-4:
                    rep.warn(f"global_orient_abs_lr[左] ≠ M·global_orient[左]·M（最大差 {d:.2e}）")
            r_ok = v[:, 1]
            if r_ok.any():
                d = np.abs(hd["global_orient"][r_ok, 1] - hd["global_orient_abs_lr"][r_ok, 1]).max()
                if d >= 1e-4:
                    rep.warn(f"global_orient_abs_lr[右] ≠ global_orient[右]（最大差 {d:.2e}）")

    # ---- 合并（merged pkl）
    merged = load_pkl(p["merged"])
    for br in BRANCHES:
        if br not in merged:
            rep.err(f"merged: 缺分支 {br}")
            continue
        b = merged[br]
        for k, n in BODY_FIELDS.items():
            if k not in b:
                rep.err(f"merged.{br}: 缺字段 {k}")
            else:
                _check_array(rep, f"merged.{br}.{k}", b[k], (n,), F, ("float32",))
        for k, tail in HAND_MERGED_FIELDS.items():
            if k not in b:
                rep.err(f"merged.{br}: 缺字段 {k}")
            else:
                _check_array(rep, f"merged.{br}.{k}", b[k], tail, F,
                             ("bool",) if k.endswith("is_detected") else ("float32",))
        for k, tail in HAND_MERGED_OPTIONAL.items():
            if k in b:
                _check_array(rep, f"merged.{br}.{k}", b[k], tail, F, ("float32",))
    info = merged.get("hamer_merge_info")
    if not isinstance(info, dict):
        rep.err("merged: 缺 hamer_merge_info")
    else:
        for k in MERGE_INFO_KEYS:
            if k not in info:
                rep.err(f"hamer_merge_info: 缺 {k}")
        for k, allowed in MERGE_INFO_FIXED.items():
            if k in info and info[k] not in allowed:
                rep.err(f"hamer_merge_info.{k} = {info[k]!r} ∉ {allowed}（消费端按它判断左手是否已镜像）")
        if info.get("merged_frames") != F or info.get("gvhmr_frames") != F:
            rep.err(f"hamer_merge_info 帧数 {info.get('gvhmr_frames')}/{info.get('merged_frames')} ≠ 身体 {F}")
        if Fh is not None and info.get("hamer_frames_original") != Fh:
            rep.warn(f"hamer_merge_info.hamer_frames_original={info.get('hamer_frames_original')} ≠ hamer_data 帧数 {Fh}")
        rep.facts["merge_info"] = {k: info[k] for k in info}

    # ---- 一致性：merged 身体 = person 身体（手腕两列除外）
    if all(br in merged and br in person for br in BRANCHES):
        for br in BRANCHES:
            for k in ("global_orient", "transl", "betas"):
                if k in merged[br] and k in person[br] and not np.array_equal(merged[br][k], person[br][k]):
                    rep.err(f"merged.{br}.{k} ≠ person.{br}.{k}")
            if "body_pose" in merged[br] and "body_pose" in person[br]:
                diff = np.abs(merged[br]["body_pose"].reshape(F, 21, 3) - person[br]["body_pose"].reshape(F, 21, 3)).max(axis=(0, 2))
                changed = [int(j) for j in np.nonzero(diff > 0)[0]]
                rep.facts.setdefault("merged_body_pose_changed_joints", {})[br] = changed
                if set(changed) - {17, 18, 19, 20}:
                    rep.err(f"merged.{br}.body_pose 改动了手腕以外的关节 {changed}")
                elif set(changed) & {17, 18}:
                    rep.warn(f"merged.{br}.body_pose 改动了肘部 {sorted(set(changed) & {17, 18})}："
                             "合并时开了「前臂扭转」（--forearm_twist，把部分手腕扭转转给肘，肘+腕合成旋转不变）")
        same = {k: bool(np.array_equal(merged[BRANCHES[0]][k], merged[BRANCHES[1]][k]))
                for k in merged[BRANCHES[0]] if k in merged[BRANCHES[1]]}
        rep.facts["merged_global_eq_incam"] = all(same.values())
        for k in ("body_pose", "left_hand_pose", "right_hand_pose"):
            if k in same and not same[k]:
                rep.warn(f"merged: 两分支 {k} 不同（局部旋转应与坐标系无关）")

    # ---- 顶层副本
    if os.path.isfile(p["top"]):
        same_bytes = _sha1(p["top"]) == _sha1(p["merged"])
        rep.facts["top_bytes_eq_merged"] = same_bytes
        if not same_bytes:
            top = load_pkl(p["top"])
            eq = all(np.array_equal(top[br][k], merged[br][k]) for br in BRANCHES if br in top and br in merged
                     for k in merged[br]) and top.get("hamer_merge_info") == merged.get("hamer_merge_info")
            if not eq:
                rep.err(f"{p['take']}.pkl 与 hamer/merged.pkl 内容不同（MoCap Doctor 回退入口会读到别的数据）")
            else:
                rep.warn(f"{p['take']}.pkl 与 merged.pkl 内容相同但字节不同")
    _numpy_refs(rep, p)
    return rep.as_dict()


def _numpy_refs(rep, p):
    """pkl 里引用的 numpy 模块名。numpy≥2 写的（numpy._core.*）在干净的 numpy 1.26（Blender 4.x 自带）里读不了
    numpy._core.numeric，要靠 PoseCapture 插件 import 时装的兼容垫片（_register_numpy_compat）；参考素材的
    merged.pkl 也是这样。numpy 1.x 写的（numpy.core.*）哪边都能读。"""
    import re
    refs = {}
    for key in ("person", "merged", "top"):
        if os.path.isfile(p[key]):
            with open(p[key], "rb") as fh:
                refs[key] = sorted({m.decode() for m in re.findall(rb"numpy\.[_a-z]+\.[a-z_]+", fh.read())})
    rep.facts["numpy_refs"] = refs
    needs_shim = [k for k, v in refs.items() if "numpy._core.numeric" in v]
    if needs_shim:
        rep.warn(f"{needs_shim} 由 numpy≥2 写出（引用 numpy._core.numeric）：干净的 numpy 1.26 读不了，"
                 f"PoseCapture 插件里有兼容垫片所以插件能读（参考素材 merged.pkl 同样如此）；用 numpy 1.x 重写可去掉这条")


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("take_dirs", nargs="+")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    bad = 0
    for d in a.take_dirs:
        r = validate_take(d)
        bad += 0 if r["ok"] else 1
        if a.json:
            print(json.dumps(r, ensure_ascii=False, indent=1, default=str))
            continue
        print(f"{'OK  ' if r['ok'] else 'FAIL'} {d}")
        for e in r["errors"]:
            print("   ERROR", e)
        for w in r["warnings"]:
            print("   warn ", w)
        print("   facts", json.dumps(r["facts"], ensure_ascii=False, default=str))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
