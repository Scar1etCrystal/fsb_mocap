"""重定向 + 整段转换自检（需要 data/ 下的云端导出数据；缺数据则跳过）。

  R1 躯干类关节在 MHR 零位上局部旋转 = 0（零位对零位）
  R2 随机 MHR 姿态：做了摆动校正的肢体骨，指向与 MHR 完全一致（≤0.1°）
  C1 999 段：合同校验 OK、帧数 999、黑场 5 帧判无效
  C2 真实帧：可对应骨指向误差 ≤0.1°
  C3 相机系骨盆投影到画面，与 YOLO 两髋中点的中位距离 < 25 px（独立检测器，验证根位置/相机约定）
  C4 地面拟合残差 < 2 cm，平滑后身体抖动下降
"""
import json
import os
import shutil
import sys
import tempfile

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from fsb import contract, convert, lbs, package, retarget, rot  # noqa: E402

D = os.path.join(ROOT, "data")
fails = 0


def check(name, ok, detail=""):
    global fails
    fails += 0 if ok else 1
    print(f"[{'PASS' if ok else 'FAIL'}] {name} :: {detail}")


if not os.path.isfile(os.path.join(D, "mhr_rest.npz")):
    print("SKIP: data/mhr_rest.npz 不在（云端导出数据不入库）"); sys.exit(0)
rt = retarget.from_rest_npz(os.path.join(D, "mhr_rest.npz"))
L = rt.local_rotations(rt.global_rotations(rt.R0[None], rt.P0[None]))
torso = [0, 3, 6, 9, 12, 13, 14, 15]
a = np.degrees(np.linalg.norm(rot.mat_to_rotvec(L[0, torso]), axis=-1))
check("R1 torso joints identity at MHR rest", a.max() < 0.01, dict(zip([rot.SMPLX_JOINT_NAMES[j] for j in torso], a.round(4))))
z = np.load(os.path.join(D, "mhr_rest.npz"))
Rr = rot.quat_to_mat(z["mhr_rand_quat_xyzw"][..., [3, 0, 1, 2]]); Pr = z["mhr_rand_pos"]
G = rt.global_rotations(Rr, Pr)
e, names = rt.bone_dir_errors(G, Pr)
fixed = [i for i, n in enumerate(names) if not n.startswith(("L_Wrist→", "R_Wrist→"))]
check("R2 swing-fixed limb bones exact on random poses (≤0.1°)", e[:, fixed].max() < 0.1, f"max {e[:, fixed].max():.4f}° over {len(fixed)} bones")

take = os.path.join(D, "t999")
if not os.path.isdir(take) or not os.path.isfile(os.path.join(D, "models", "SMPLX_NEUTRAL.npz")):
    print("SKIP: data/t999 或 SMPL-X 模型不在")
else:
    sm = lbs.SMPLXLBS(os.path.join(D, "models", "SMPLX_NEUTRAL.npz"))
    seq, info, dbg = convert.convert_take(take, rt, J0=sm.J[0])
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "t999_fsb")
        package.package_take(seq, out)
        r = contract.validate_take(out)
        check("C1 999: contract OK, 999 frames, 5 black frames invalid", r["ok"] and r["facts"]["frames"] == 999
              and info["invalid_frames"] == [994, 995, 996, 997, 998], (r["errors"][:2], info["invalid_frames"]))
    check("C2 real frames: mappable bone directions exact (≤0.1°)", info["bone_dir_err_deg"]["exact_bones_max"] <= 0.1, info["bone_dir_err_deg"])
    d = convert.load_take(take)
    K = np.array(d["cam"]["K"])
    root_cam = d["pred_cam_t"] + d["P_model"][:, rt.idx["root"]] @ retarget.F_FLIP.T
    uv = (root_cam[:, :2] / root_cam[:, 2:3]) * K[0, 0] + K[:2, 2]
    hips = d["yolo_kpts"][:, [11, 12]]
    ok = (hips[:, :, 2] > 0.5).all(1)
    mid = hips[ok, :, :2].mean(1)
    dist = np.linalg.norm(uv[ok] - mid, axis=1)
    check("C3 camera-frame pelvis projects onto YOLO hip midpoint (median < 25 px)", np.median(dist) < 25,
          f"median {np.median(dist):.1f}px p90 {np.percentile(dist, 90):.1f}px over {ok.sum()} frames")
    g = info["ground"]; j = info["body_jitter_deg_per_frame"]
    check("C4 floor fit < 2 cm RMS, smoothing reduces jitter", g["plane_rms_m"] < 0.02 and j["after"] < j["before"], (g, j))
print(f"==== {'0 FAIL' if not fails else f'{fails} FAIL'} ====")
sys.exit(1 if fails else 0)
