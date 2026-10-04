"""对比脚本自检：参考 vs 自己 = 0；人为扰动（常值朝向差、单帧尖刺、关节噪声、帧偏移）都能被量出来。"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fsb import compare, package, rot  # noqa: E402

REF = "/home/sb/fastsam_handoff/reference_output/0001-1499_f_full_20260925_165539"
fails = 0


def check(name, ok, detail=""):
    global fails
    fails += 0 if ok else 1
    print(f"[{'PASS' if ok else 'FAIL'}] {name} :: {detail}")


r = compare.compare(REF, REF)
check("self vs self: zero error", r["body_local"]["all"]["max"] == 0 and r["world"]["root_orient"]["max"] < 1e-3
      and r["offset"]["chosen"] == 0, (r["body_local"]["all"], r["offset"]["chosen"]))
check("self: jump counts equal old/new", r["jumps"]["old"] == r["jumps"]["new"], r["jumps"]["old"]["body_spikes"])

seq = package.seq_from_take(REF)
F = seq["body_pose"].shape[0]
rng = np.random.default_rng(0)
s2 = {k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v) for k, v in seq.items()}
# 1) 整个世界绕 Y 转 37°（根朝向 + 位移）
Ry = compare._rot_y(np.radians(37.0))
go = rot.rotvec_to_mat(s2["world_global_orient"])
s2["world_global_orient"] = rot.mat_to_rotvec(Ry @ go).astype(np.float32)
s2["world_transl"] = ((s2["world_transl"] - s2["world_transl"][0]) @ Ry.T + s2["world_transl"][0]).astype(np.float32)
# 2) 左膝（关节 4 → body_pose 下标 3）加 ~3° 随机噪声
bp = s2["body_pose"].reshape(F, 21, 3).astype(np.float64)
noise = rng.normal(size=(F, 3)); noise = noise / np.linalg.norm(noise, axis=1, keepdims=True) * np.radians(3.0)
bp[:, 3] = rot.mat_to_rotvec(rot.rotvec_to_mat(bp[:, 3]) @ rot.rotvec_to_mat(noise))
# 3) 第 500 帧右肘（关节 19 → 下标 18）单帧尖刺 80°
bp[500, 18] = rot.mat_to_rotvec(rot.rotvec_to_mat(bp[500, 18]) @ rot.rotvec_to_mat(np.array([0.0, np.radians(80.0), 0.0])))
s2["body_pose"] = bp.reshape(F, 63).astype(np.float32)
with tempfile.TemporaryDirectory() as tmp:
    out = os.path.join(tmp, "perturbed")
    package.package_take(s2, out)
    r = compare.compare(REF, out)
    check("yaw offset recovered (37°)", abs(r["world"]["yaw_offset_deg"] - 37.0) < 0.5, r["world"]["yaw_offset_deg"])
    check("root orient error ≈ 0 after yaw alignment", r["world"]["root_orient"]["max"] < 0.5, r["world"]["root_orient"])
    check("root translation ≈ 0 after alignment", r["root_transl"]["dist_m"]["max"] < 1e-3, r["root_transl"]["dist_m"])
    lk = r["body_local"]["joints"]["L_Knee"]
    check("L_Knee noise measured ≈ 3°", 2.9 < lk["mean"] < 3.1, lk)
    check("untouched joint exact", r["body_local"]["joints"]["R_Knee"]["max"] == 0, r["body_local"]["joints"]["R_Knee"])
    check("spike frame 500 detected in new only", 500 in r["jumps"]["new"]["body_spike_frames"]
          and 500 not in r["jumps"]["old"]["body_spike_frames"], r["jumps"]["new"]["body_spike_frames"][:10])
    check("verdict: new has spikes", r["verdict"]["new_no_body_spikes"] is False)
    md = compare.to_markdown(r)
    check("markdown renders", "## 跳帧检查" in md and "L_Knee" in md, len(md))
# 4) 帧偏移：新序列前面多 2 帧
s3 = {k: (np.concatenate([v[:1], v[:1], v]) if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == F else v) for k, v in seq.items()}
with tempfile.TemporaryDirectory() as tmp:
    out = os.path.join(tmp, "shifted")
    package.package_take(s3, out)
    r = compare.compare(REF, out)
    check("auto offset finds +2", r["offset"]["chosen"] == 2 and r["body_local"]["all"]["max"] == 0, r["offset"])
print(f"==== {'0 FAIL' if not fails else f'{fails} FAIL'} ====")
sys.exit(1 if fails else 0)
