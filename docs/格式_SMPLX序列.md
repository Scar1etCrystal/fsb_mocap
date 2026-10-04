# 格式：SMPL-X 序列（云端 → 打包器的中间格式，`fsb.package` 的输入）

一个 take 一个 `.npz`（或 dict），键：

| 键 | 形状 | 说明 |
|---|---|---|
| fps | 标量 | 30 |
| body_pose | (F,63) | SMPL-X 21 关节局部轴角（**含手腕**，FastSAM3DBody 原生有手腕） |
| left_hand_pose / right_hand_pose | (F,45) | SMPL-X 手指局部轴角，真左手表示，**绝对旋转**（相对平直手；插件直接套到骨上）。已核实旧链路 HaMeR 值也是绝对旋转：按绝对比差 27°，若按"相对均值手"比差 50° |
| betas | (10,) 或 (F,10) | 新链路写 0（插件不读 betas；SMPL 的 betas 不能搬进 SMPL-X 字段） |
| world_global_orient / world_transl | (F,3) | 世界系根（fsb.world 从相机系落地得到） |
| incam_global_orient / incam_transl | (F,3) | 可选：相机系根；缺省 = 世界系副本 |
| hand_valid | (F,2) bool | 可选，左/右手可信 |
| hand_conf | (F,2) | 可选 |
| hand_frames_original | 标量 | 可选，写进 hamer_merge_info |

F = 视频帧数（以身体为准）。所有旋转都已做时间处理（fsb.smooth：去尖刺 + 高斯）。

新链路实际产出（fsb.convert）：world_* 由脚跟/脚尖关键点拟合地面落地（Y 上、首帧 XZ=0、地面 y=0）；incam 缺省 = 世界系副本；
hand_valid = YOLO 手腕置信度 > 0.3 且非无效帧；hand_conf = 该置信度。
