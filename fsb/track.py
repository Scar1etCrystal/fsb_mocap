"""R2 多人筛选：从逐帧人体检测里锁定主角，全程一条无缺口的包围盒轨迹。

旧行为（GVHMR tracker.get_one_track）：YOLO 跟踪 → 按"累计包围盒面积 / 画面面积"选**一条**跟踪 ID →
缺帧线性插值 → 5 帧滑动平均两遍。弱点：主角被路人挡一下就换 ID，后半段直接丢（只能插值糊过去）；
路人与主角交叉时跟踪器把 ID 换到路人身上，就跟着路人走了。

这里：
  1. 跟踪段（tracklet）：用跟踪器给的 ID；没有 ID 就自己做 IoU 关联。段内出现**不合理跳变**
     （中心跳 > 0.6×框高，或面积比 > 2.0）就切开——这正是 ID 换到路人身上的样子。
  2. 锚点 = 累计归一化面积最大的段（与 GVHMR 同一判据）。
  3. 从锚点向前、向后**接段**：候选段必须在时间上接得上（重叠 ≤ 3 帧、间隔 ≤ max_gap），
     位置与匀速外推一致、大小比在 [1/1.6, 1.6] 内；挑代价最小的接上，直到接不上为止。
  4. 缺帧线性插值、两头保持，再 5 帧滑动平均两遍（与旧链路相同，裁剪框不抖）。
  5. 报告：覆盖率、缺口、接了哪些段、路人段、**风险帧**（别人的框有 > 30% 落在主角框里——
     交集 / 较小框面积，小路人走过时 IoU 很低反映不出来——即交叉/遮挡，需要抽查的关键帧）。

输入：逐帧检测 dets[t] = (N,5) [x1,y1,x2,y2,conf]，可选 ids[t] = (N,) 跟踪 ID（−1 = 无）。
"""
from __future__ import annotations

import numpy as np

CUT_CENTER_JUMP = 0.6        # × 框高
CUT_AREA_RATIO = 2.0
LINK_SIZE_RATIO = 1.6
LINK_MAX_OVERLAP = 3
RISK_IOU = 0.3


def iou_xyxy(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])
    return inter / np.maximum(area(a)[:, None] + area(b)[None, :] - inter, 1e-9)


def iomin_xyxy(a, b):
    """交集 / 较小框面积：小路人走过大主角时 IoU 很低（≤ 面积比），这个量才反映"挡住了"。"""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])
    return inter / np.maximum(np.minimum(area(a)[:, None], area(b)[None, :]), 1e-9)


def _area(b):
    return max(float(b[2] - b[0]), 1e-6) * max(float(b[3] - b[1]), 1e-6)


def _center(b):
    return np.array([(b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5])


def _iou_tracker(dets, iou_thr=0.3, max_age=10):
    """没有跟踪 ID 时的简易关联（贪心 IoU，带短时记忆）。返回 ids[t]。"""
    nxt, live, ids = 0, {}, []           # live: id -> (last_frame, box)
    for t, d in enumerate(dets):
        d = np.asarray(d).reshape(-1, 5)
        cur = -np.ones(len(d), dtype=int)
        cand = [(i, tid) for tid, (lt, _) in live.items() if t - lt <= max_age for i in [tid]]
        if len(d) and cand:
            tids = [c[1] for c in cand]
            M = iou_xyxy(d[:, :4], np.stack([live[k][1] for k in tids]))
            while M.size and M.max() >= iou_thr:
                i, j = np.unravel_index(np.argmax(M), M.shape)
                cur[i] = tids[j]
                M[i, :] = -1; M[:, j] = -1
        for i in range(len(d)):
            if cur[i] < 0:
                cur[i] = nxt; nxt += 1
            live[int(cur[i])] = (t, d[i, :4].copy())
        ids.append(cur)
    return ids


def build_tracklets(dets, ids=None):
    """→ list of {'id','frames':(n,),'boxes':(n,4),'conf':(n,)}，已按不合理跳变切开。"""
    if ids is None or all(i is None for i in ids):
        ids = _iou_tracker(dets)
    raw = {}
    for t, (d, idv) in enumerate(zip(dets, ids)):
        d = np.asarray(d).reshape(-1, 5)
        idv = np.asarray(idv if idv is not None else -np.ones(len(d)), dtype=int).reshape(-1)
        for k in range(len(d)):
            key = int(idv[k]) if idv[k] >= 0 else f"solo{t}_{k}"
            raw.setdefault(key, []).append((t, d[k, :4], float(d[k, 4])))
    out = []
    for key, items in raw.items():
        items.sort(key=lambda x: x[0])
        seg = [items[0]]
        for it in items[1:]:
            pb, cb = seg[-1][1], it[1]
            h = max(pb[3] - pb[1], 1.0)
            jump = np.linalg.norm(_center(cb) - _center(pb)) / h
            ratio = max(_area(cb), _area(pb)) / min(_area(cb), _area(pb))
            gap = it[0] - seg[-1][0]
            if jump > CUT_CENTER_JUMP * max(1, gap) ** 0.5 or ratio > CUT_AREA_RATIO:
                out.append(seg); seg = [it]
            else:
                seg.append(it)
        out.append(seg)
    tracklets = []
    for n, seg in enumerate(out):
        tracklets.append({"id": n, "frames": np.array([s[0] for s in seg]),
                          "boxes": np.array([s[1] for s in seg], dtype=np.float64),
                          "conf": np.array([s[2] for s in seg])})
    return tracklets


def _score(tr, W, H):
    b = tr["boxes"]
    return float(((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) / (W * H)).sum())


def _end_state(tr, forward, k=5):
    """外推用：端点框 + 每帧速度（取端点附近 k 帧）。"""
    f, b = tr["frames"], tr["boxes"]
    if forward:
        sel = slice(max(0, len(f) - k), len(f)); t0, b0 = f[-1], b[-1]
    else:
        sel = slice(0, min(k, len(f))); t0, b0 = f[0], b[0]
    ff, bb = f[sel], b[sel]
    v = (bb[-1] - bb[0]) / max(ff[-1] - ff[0], 1) if len(ff) > 1 else np.zeros(4)
    return int(t0), b0, v


def _link_cost(chain_end, cand, forward, max_gap):
    t0, b0, v = chain_end
    f, b = cand["frames"], cand["boxes"]
    t1, b1 = (int(f[0]), b[0]) if forward else (int(f[-1]), b[-1])
    gap = (t1 - t0) if forward else (t0 - t1)
    if gap < -LINK_MAX_OVERLAP or gap > max_gap or gap == 0:
        return None
    pred = b0 + v * (gap if forward else -gap) * (1.0 if gap <= 30 else 30.0 / gap)
    h = max(b0[3] - b0[1], 1.0)
    dist = np.linalg.norm(_center(b1) - _center(pred)) / h
    ratio = max(_area(b1), _area(b0)) / min(_area(b1), _area(b0))
    if ratio > LINK_SIZE_RATIO:
        return None
    allow = 0.5 + 0.03 * max(gap, 0)                 # 间隔越长，位置越宽容（出画再入画）
    if dist > allow:
        return None
    return dist + 0.5 * (ratio - 1.0) + 0.01 * max(gap, 0)


def select_main_subject(dets, W, H, ids=None, *, max_gap=90, smooth_window=5, smooth_passes=2):
    F = len(dets)
    trs = build_tracklets(dets, ids)
    if not trs:
        m = 0.05 * min(W, H)                         # 与旧 detect_person_bbx 兜底一致：全图留 5% 边距
        box = np.tile([m, m, W - m, H - m], (F, 1)).astype(np.float32)
        return {"bbox_xyxy": box, "detected": np.zeros(F, bool), "report": {"note": "整段没有检测到人，用全图兜底"}}
    scores = {t["id"]: _score(t, W, H) for t in trs}
    anchor = max(trs, key=lambda t: scores[t["id"]])
    chain, used = [anchor], {anchor["id"]}
    links = []
    for forward in (True, False):
        while True:
            end_tr = chain[-1] if forward else chain[0]
            st = _end_state(end_tr, forward)
            best = None
            for c in trs:
                if c["id"] in used:
                    continue
                cost = _link_cost(st, c, forward, max_gap)
                if cost is not None and (best is None or cost < best[0]):
                    best = (cost, c)
            if best is None:
                break
            c = best[1]
            used.add(c["id"])
            gap = int(c["frames"][0] - st[0]) if forward else int(st[0] - c["frames"][-1])
            links.append({"tracklet": int(c["id"]), "dir": "fwd" if forward else "bwd", "gap": gap,
                          "cost": round(float(best[0]), 3), "frames": [int(c["frames"][0]), int(c["frames"][-1])]})
            if forward:
                chain.append(c)
            else:
                chain.insert(0, c)
    box = np.zeros((F, 4)); have = np.zeros(F, bool); src = -np.ones(F, int)
    for c in chain:                                  # 重叠帧：后接上的段覆盖前段（接缝处只差 ≤3 帧）
        for t, b in zip(c["frames"], c["boxes"]):
            if 0 <= t < F:
                box[t] = b; have[t] = True; src[t] = c["id"]
    idx = np.nonzero(have)[0]
    for k in range(4):                               # 线性插值，两头保持
        box[:, k] = np.interp(np.arange(F), idx, box[idx, k])
    sm = box.copy()
    for _ in range(smooth_passes):
        pad = smooth_window // 2
        p = np.pad(sm, ((pad, pad), (0, 0)), mode="edge")
        sm = np.stack([np.convolve(p[:, k], np.ones(smooth_window) / smooth_window, mode="valid") for k in range(4)], 1)
    # 风险帧：别人的框与主角框重叠（交叉 / 遮挡）
    risk = []
    for t, d in enumerate(dets):
        d = np.asarray(d).reshape(-1, 5)
        if not len(d):
            continue
        ious = iou_xyxy(sm[t], d[:, :4])[0]
        own = int(np.argmax(ious)) if have[t] else -1
        cover = iomin_xyxy(sm[t], d[:, :4])[0]
        others = [float(v) for i, v in enumerate(cover) if i != own]
        if others and max(others) > RISK_IOU:
            risk.append(t)
    gaps, start = [], None
    for t in range(F):
        if not have[t] and start is None:
            start = t
        if have[t] and start is not None:
            gaps.append([start, t - 1]); start = None
    if start is not None:
        gaps.append([start, F - 1])
    others = sorted((t for t in trs if t["id"] not in used), key=lambda t: -scores[t["id"]])
    report = {
        "frames": F, "coverage": round(float(have.mean()), 4), "gaps": gaps,
        "anchor_tracklet": int(anchor["id"]), "chain": [int(c["id"]) for c in chain], "links": links,
        "tracklets_total": len(trs),
        "others_top": [{"tracklet": int(t["id"]), "frames": [int(t["frames"][0]), int(t["frames"][-1])],
                        "n": int(len(t["frames"])), "area_score": round(scores[t["id"]], 2)} for t in others[:8]],
        "main_area_score": round(sum(scores[c["id"]] for c in chain), 2),
        "risk_frames": _runs(risk),
    }
    return {"bbox_xyxy": sm.astype(np.float32), "detected": have, "source_tracklet": src, "report": report}


def _runs(frames):
    out = []
    for f in frames:
        if out and f == out[-1][1] + 1:
            out[-1][1] = f
        else:
            out.append([f, f])
    return out
