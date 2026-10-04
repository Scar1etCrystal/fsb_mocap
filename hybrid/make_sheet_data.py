"""关键帧对照图的数据：各版本 SMPL-X 网格（含手）+ 视频人物/左右手裁剪。Blender 那边只负责渲染拼图。

列：视频 | 旧（GVHMR+HaMeR）| FSB 自带手 | 混合 H4（FSB 身体 + HaMeR 手 + 俯仰补偿 + 门控 + 跨帧连续的小臂扭转）
"""
from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from fsb import lbs, rot  # noqa: E402
from merge_tools import load  # noqa: E402
from signals import TAKES  # noqa: E402

FRAMES = {   # pkl 帧号：标注坏段里挑代表帧 + 几个正常帧
    "t999": {100: "左手 87–126（你确认的坏段）", 115: "左手 87–126", 540: "A 右手 532–556", 691: "B1 右手 688–696",
             737: "左小臂甩动处（H2f）", 746: "左小臂甩动处（H2f）", 750: "左小臂甩动处（H2f）", 771: "E 左手 ~771（不修）",
             788: "D1 双手 786–791", 806: "D2 左手 800–817", 300: "正常", 900: "正常"},
    "t1499": {520: "遮挡 右手 512–554", 540: "遮挡 右手 512–554", 890: "遮挡 左手 884–898", 200: "正常", 700: "正常", 1200: "正常"},
}
VERSIONS = {"old": "{ref}/hamer/merged.pkl", "fsb": "{fsb}/hamer/merged.pkl", "H2f": ROOT + "/out_hybrid/{take}_H4/hamer/merged.pkl"}  # 第 4 列 = H4
SMPLX = os.path.join(ROOT, "data", "models", "SMPLX_NEUTRAL.npz")


def crop(img, box, out_wh, pad=0.15):
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    ow, oh = out_wh
    s = max(w * (1 + 2 * pad) / ow, h * (1 + 2 * pad) / oh)
    M = np.array([[1 / s, 0, ow / 2 - cx / s], [0, 1 / s, oh / 2 - cy / s]])
    return cv2.warpAffine(img, M, (ow, oh), flags=cv2.INTER_AREA, borderValue=(40, 40, 40))


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    model = lbs.SMPLXLBS(SMPLX)
    for take, cfg in TAKES.items():
        frames = sorted(FRAMES[take])
        mt = np.load(f"{cfg['raw']}/main_track.npz")
        kp2 = np.concatenate([np.load(f)["pred_keypoints_2d"] for f in sorted(
            __import__("glob").glob(f"{cfg['raw']}/sam3d_part_*.npz"))])
        data = {"faces": model.f.astype(np.int32), "frames": np.array(frames)}
        for name, tpl in VERSIONS.items():
            g = load(tpl.format(ref=cfg["ref"], fsb=cfg["fsb"], take=take))["smpl_params_global"]
            G = rot.chain_global_rotations(np.asarray(g["global_orient"], float), np.asarray(g["body_pose"], float),
                                           np.asarray(g["left_hand_pose"], float), np.asarray(g["right_hand_pose"], float))
            for f in frames:
                v, p = model.pose(G[f])
                data[f"v_{name}_{f}"] = v.astype(np.float32)
                data[f"j_{name}_{f}"] = p.astype(np.float32)
        cap = cv2.VideoCapture(f"{cfg['ref']}/gvhmr/0_input_video.mp4")
        i = 0
        want = set(frames)
        while want:
            ok, img = cap.read()
            if not ok:
                break
            if i in want:
                want.discard(i)
                cv2.imwrite(f"{out_dir}/{take}_{i:05d}_person.png", crop(img, mt["bbox_xyxy"][i], (540, 360), pad=0.05))
                for side, sl in (("L", slice(42, 63)), ("R", slice(21, 42))):
                    h = kp2[i, sl]
                    box = [h[:, 0].min(), h[:, 1].min(), h[:, 0].max(), h[:, 1].max()]
                    sz = max(box[2] - box[0], box[3] - box[1], 30) * 1.4      # 正方形、留足边
                    c = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                    cv2.imwrite(f"{out_dir}/{take}_{i:05d}_hand{side}.png",
                                crop(img, (c[0] - sz / 2, c[1] - sz / 2, c[0] + sz / 2, c[1] + sz / 2), (270, 180), pad=0.0))
            i += 1
        np.savez_compressed(f"{out_dir}/{take}_meshes.npz", **data)
        with open(f"{out_dir}/{take}_labels.json", "w", encoding="utf-8") as fh:
            json.dump({str(f): FRAMES[take][f] for f in frames}, fh, ensure_ascii=False)
        print(take, "frames", frames)


if __name__ == "__main__":
    main(sys.argv[1])
