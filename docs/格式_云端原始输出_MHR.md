# 格式：云端原始输出（FastSAM3DBody / MHR，一段 take 一个目录）

由 `cloud/run_take.py`（跟踪/选人/焦距/推理）+ `cloud/recompute_skel.py`（重算骨架）产生。

| 文件 | 内容 |
|---|---|
| `track_raw.npz` | 逐帧全部人体检测：`frame (M,)`、`xyxy (M,4)` 像素、`conf (M,)`、`tid (M,)` ByteTrack ID、`kpts (M,17,3)` COCO 17 点（x, y, conf），`W H fps F` |
| `main_track.npz` / `.json` | 主角框 `bbox_xyxy (F,4)`（无缺口，5 帧滑动平均两遍）、`detected (F,)` 真检测到、`source_tracklet (F,)`、`kpts (F,17,3)` 主角的 COCO 点（无则 NaN）；json = 筛选报告（覆盖率、缺口、接段、路人段、风险帧） |
| `cam.json` | 固定内参 `K`（MoGe-2 在 15 帧上估计取中位数）、各帧样本、离散度 |
| `sam3d_part_XXXX.npz` | 每 200 帧一块，见下表 |
| `skel_part_XXXX.npz` | 用最终参数重算的 MHR 骨架：`joint_rots_model (n,127,3,3)`、`joint_coords_model (n,127,3)` |
| `mhr2smpl.npz` | 交叉验证用：作者前馈 MHR→SMPL（`go_aa` `body_pose (63)` `betas`） |

## sam3d_part 字段（n = 本块帧数）

| 键 | 形状 | 坐标系 / 单位 | 说明 |
|---|---|---|---|
| pred_cam_t | (n,3) | OpenCV 相机系，米 | 模型原点（两脚之间、约地面高）在相机系的位置 |
| pred_keypoints_3d | (n,70,3) | **相机轴向**（y 下），相对模型原点，米 | 70 点（MHR70：脚跟/大脚趾/小脚趾、手指各节、肘尖、肩峰…）；+cam_t 后用 K 投影 = pred_keypoints_2d（实测误差 0 px） |
| pred_keypoints_2d | (n,70,2) | 像素 | |
| pred_joint_coords | (n,127,3) | 相机轴向，相对模型原点 | MHR 127 关节位置 |
| pred_global_rots | (n,127,3,3) | **模型系**（x 角色左、y 上、z 前） | ⚠ 手部关节是手部融合前的旧值（与位置差到 115°）——不要用，用 skel_part |
| pred_vertices | (n,18439,3) f16 | 相机轴向，相对模型原点 | MHR 网格 |
| global_rot / body_pose_params (133) / hand_pose_params (108) / scale_params (28) / shape_params (45) | | | MHR 参数（重算骨架用） |
| focal_length / bbox / lhand_bbox / rhand_bbox / times | | 像素 / 秒 | |

模型系 ↔ 相机系：`P_cam = F · P_model`，`F = diag(1, −1, −1)`（绕 x 转 180°）。旋转：相机系朝向 = `F · G_model`。

## 帧约定

视频解码帧 0..F−1（OpenCV 解码；1499 段文件头写 1500 帧、实际可解 1499 帧，与旧链路 GVHMR 一致）。
没有检测到人的片头/片尾帧（999 段最后 5 帧黑场）推理结果无效，转换时沿用最近有效帧。
