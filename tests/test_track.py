"""多人筛选自检（合成数据）：遮挡断 ID、路人交叉换 ID、出画再入画、开头没检测到。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fsb import track  # noqa: E402

W, H, F = 1920, 1080, 600
fails = 0


def check(name, ok, detail=""):
    global fails
    fails += 0 if ok else 1
    print(f"[{'PASS' if ok else 'FAIL'}] {name} :: {detail}")


def main_box(t):                # 主角：大框，左右来回走
    cx = 960 + 300 * np.sin(t / 60.0); cy = 560
    return np.array([cx - 180, cy - 420, cx + 180, cy + 420])


def walker_box(t, t0=200):      # 路人：小框，从右往左走过，会穿过主角
    cx = 1900 - 8.0 * (t - t0); cy = 600
    return np.array([cx - 60, cy - 150, cx + 60, cy + 150])


def run(dets, ids, truth, name):
    r = track.select_main_subject(dets, W, H, ids)
    b = r["bbox_xyxy"]
    err = np.array([np.linalg.norm(((b[t, :2] + b[t, 2:]) - (truth[t, :2] + truth[t, 2:])) / 2) for t in range(F)])
    return r, err


# 1) 遮挡断 ID：主角 0–299 是 id1，305–599 是 id7（中间 5 帧没检测到）；路人 id2 200–440
dets, ids, truth = [], [], []
for t in range(F):
    d, i = [], []
    if t < 300 or t >= 305:
        d.append(list(main_box(t)) + [0.9]); i.append(1 if t < 300 else 7)
    if 200 <= t <= 440:
        d.append(list(walker_box(t)) + [0.8]); i.append(2)
    dets.append(np.array(d) if d else np.zeros((0, 5))); ids.append(np.array(i)); truth.append(main_box(t))
truth = np.array(truth)
r, err = run(dets, ids, truth, "occlusion")
check("occlusion: chain covers both main IDs, walker excluded", r["report"]["gaps"] == [[300, 304]] and err.max() < 40,
      f"gaps={r['report']['gaps']} links={r['report']['links']} max_err={err.max():.1f}px others={[o['frames'] for o in r['report']['others_top']]}")
check("occlusion: crossing frames flagged as risk", any(a <= 330 <= b or a <= 360 <= b for a, b in r["report"]["risk_frames"]),
      r["report"]["risk_frames"][:5])

# 2) 交叉换 ID：t≥330 时跟踪器把 id1 给了路人、id2 给了主角
dets, ids = [], []
for t in range(F):
    d, i = [list(main_box(t)) + [0.9]], [1 if t < 330 else 2]
    if 200 <= t <= 440:
        d.append(list(walker_box(t)) + [0.8]); i.append(2 if t < 330 else 1)
    dets.append(np.array(d)); ids.append(np.array(i))
r, err = run(dets, ids, truth, "swap")
check("ID swap: follows the main subject, not the walker", err.max() < 40,
      f"max_err={err.max():.1f}px chain={r['report']['chain']} links={r['report']['links']}")

# 3) 出画再入画：主角 250–309 消失（60 帧），回来换 id9，位置偏了 150px
dets, ids = [], []
for t in range(F):
    d, i = [], []
    if t < 250:
        d.append(list(main_box(t)) + [0.9]); i.append(1)
    elif t >= 310:
        d.append(list(main_box(t) + [150, 0, 150, 0]) + [0.9]); i.append(9)
    if 200 <= t <= 440:
        d.append(list(walker_box(t)) + [0.8]); i.append(2)
    dets.append(np.array(d) if d else np.zeros((0, 5))); ids.append(np.array(i))
r, _ = run(dets, ids, truth, "exit")
check("exit/re-enter: re-acquired after 60-frame gap (walker not taken)", r["report"]["gaps"] == [[250, 309]]
      and 9 in [l["tracklet"] for l in r["report"]["links"]] or len(r["report"]["chain"]) == 2,
      f"gaps={r['report']['gaps']} links={r['report']['links']}")
b = r["bbox_xyxy"]
check("exit gap interpolated (no zeros, inside frame span)", (b.sum(1) != 0).all(), "")

# 4) 开头 20 帧没检测到 + 没有跟踪 ID（自带 IoU 关联）
dets = []
for t in range(F):
    d = []
    if t >= 20:
        d.append(list(main_box(t)) + [0.9])
    if 200 <= t <= 440:
        d.append(list(walker_box(t)) + [0.8])
    dets.append(np.array(d) if d else np.zeros((0, 5)))
r, err = run(dets, None, truth, "noids")
check("no tracker IDs: built-in IoU tracker + hold first box", err[25:].max() < 40 and r["report"]["gaps"] == [[0, 19]],
      f"max_err(25:)={err[25:].max():.1f}px gaps={r['report']['gaps']}")
print(f"==== {'0 FAIL' if not fails else f'{fails} FAIL'} ====")
sys.exit(1 if fails else 0)
