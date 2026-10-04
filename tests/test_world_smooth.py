"""世界系落地 + 平滑 自检（合成数据，不需要 SMPL-X 模型）。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fsb import rot, smooth, world  # noqa: E402

fails = 0


def check(name, ok, detail=""):
    global fails
    fails += 0 if ok else 1
    print(f"[{'PASS' if ok else 'FAIL'}] {name} :: {detail}")


rng = np.random.default_rng(3)
F = 600
t = np.arange(F)
# ---- 世界系真值：Y 上，地面 y=0。骨盆绕圈走、躯干前倾 ±15°，两脚交替着地
pelvis = np.stack([0.8 * np.sin(t / 90), 0.95 + 0.03 * np.sin(t / 7), 0.6 * np.cos(t / 90) + 3.0], 1)
lean = np.radians(15 * np.sin(t / 40))
neck = pelvis + np.stack([np.zeros(F), 0.5 * np.cos(lean), 0.5 * np.sin(lean)], 1)
phase = (t // 20) % 2
lfoot = pelvis + [-0.12, 0, 0]; rfoot = pelvis + [0.12, 0, 0]
lfoot[:, 1] = np.where(phase == 0, 0.0, 0.12 * np.abs(np.sin((t % 20) / 20 * np.pi)))
rfoot[:, 1] = np.where(phase == 1, 0.0, 0.12 * np.abs(np.sin((t % 20) / 20 * np.pi)))
ltoe = lfoot + [0, 0, 0.15]; ltoe[:, 1] = lfoot[:, 1]
rtoe = rfoot + [0, 0, 0.15]; rtoe[:, 1] = rfoot[:, 1]
Jw = np.stack([pelvis, neck, lfoot, rfoot, ltoe, rtoe], 1)          # (F,6,3)
# 相机：俯视 12°、侧倾 2°，世界→相机（OpenCV）：先世界→"水平相机系"(RZ180)，再加俯仰/侧倾
pitch, roll = np.radians(12.0), np.radians(2.0)
Rx = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])
Rz = np.array([[np.cos(roll), -np.sin(roll), 0], [np.sin(roll), np.cos(roll), 0], [0, 0, 1]])
R_cw = Rz @ Rx @ np.diag([-1.0, -1.0, 1.0])                        # 世界 → 相机
cam_pos = np.array([0.0, 1.6, -1.0])
Jc = (Jw - cam_pos) @ R_cw.T + rng.normal(scale=0.008, size=Jw.shape)
up, d, info = world.estimate_up(Jc, foot_idx=(2, 3, 4, 5), pelvis_idx=0, neck_idx=1)
true_up = R_cw @ np.array([0.0, 1.0, 0.0])
err = np.degrees(np.arccos(np.clip(up @ true_up, -1, 1)))
check("floor-plane up within 1° of truth (camera pitched 12°, rolled 2°)", err < 1.0, f"err={err:.3f}° info={info}")
tilt = world.camera_tilt(up)
check("camera tilt reported ≈ pitch 12 / roll 2", abs(tilt["pitch_deg"] - 12) < 1 and abs(abs(tilt["roll_deg"]) - 2) < 1, tilt)
torso_only = np.degrees(np.arccos(np.clip(world._unit(np.mean([world._unit(v) for v in Jc[:, 1] - Jc[:, 0]], 0)) @ true_up, -1, 1)))
check("torso-average alone is worse than the floor fit (lean bias)", torso_only > err, f"torso {torso_only:.2f}° vs floor {err:.2f}°")

# ---- 根参数：世界真值 → 相机系 → to_world 还原
J0 = np.array([0.003, -0.35, 0.012])
go_w_true = rot.mat_to_rotvec(rot.rotvec_to_mat(rng.normal(size=(F, 3)) * 0.4) @ rot.rotvec_to_mat(np.tile([0, 0.3, 0], (F, 1))))
tr_w_true = pelvis - J0
go_c = rot.mat_to_rotvec(np.einsum("ij,fjk->fik", R_cw, rot.rotvec_to_mat(go_w_true)))
tr_c = (pelvis - cam_pos) @ R_cw.T - J0
go_w, tr_w, R_wc = world.to_world(go_c, tr_c, J0, true_up, float(true_up @ ((np.zeros(3) - cam_pos) @ R_cw.T)))
check("global_orient back to world (< 1e-6°)", rot.geodesic_deg(rot.rotvec_to_mat(go_w), rot.rotvec_to_mat(go_w_true)).max() < 1e-6, "")
exp_t = tr_w_true.copy(); exp_t[:, 0] -= exp_t[0, 0]; exp_t[:, 2] -= exp_t[0, 2]
check("transl back to world: floor y=0, frame-0 XZ=0 (< 1e-9 m)", np.abs(tr_w - exp_t).max() < 1e-9, np.abs(tr_w - exp_t).max())
level = world.camera_rotation(np.array([0.0, -1.0, 0.0]))
check("level camera ⇒ R_wc = RZ180 (same as hand_body_merge compensation)", np.allclose(level, np.diag([-1.0, -1.0, 1.0])), level.round(3).tolist())

# ---- 平滑
truth = rot.rotvec_to_mat(np.stack([np.sin(t / 30) * 0.8, np.cos(t / 45) * 0.5, t * 0.01], 1))   # 平滑真值
noise = rot.rotvec_to_mat(rng.normal(scale=np.radians(2.0), size=(F, 3)))
noisy = truth @ noise
spikes = [100, 300, 450]
for s in spikes:
    noisy[s] = noisy[s] @ rot.rotvec_to_mat(np.array([np.radians(70.0), 0, 0]))
ds, fixed = smooth.despike(noisy, thr_deg=30)
check("despike removes the 3 injected single-frame spikes", set(spikes) <= set(fixed), f"fixed={fixed}")
sm = smooth.gaussian(ds, sigma=1.5)
e_before = rot.geodesic_deg(noisy, truth); e_after = rot.geodesic_deg(sm, truth)
check("gaussian: error vs truth drops > 40% (noise 2°/frame)", e_after.mean() < 0.6 * np.delete(e_before, spikes).mean(),
      f"before {np.delete(e_before, spikes).mean():.2f}° after {e_after.mean():.2f}°")
lin = rot.rotvec_to_mat(np.stack([t * 0.02, np.zeros(F), np.zeros(F)], 1))       # 匀速转动
check("gaussian: zero lag on constant-speed rotation", rot.geodesic_deg(smooth.gaussian(lin, sigma=2.0)[10:-10], lin[10:-10]).max() < 1e-6, "")
print(f"==== {'0 FAIL' if not fails else f'{fails} FAIL'} ====")
sys.exit(1 if fails else 0)
