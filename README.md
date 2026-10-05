# fsb_frontend — FastSAM3DBody 替换 GVHMR+HaMeR 的前端（与 GVHMR 并存）

需求合同：`/home/sb/fastsam_handoff/PoseCapture_Pack_code/前端替换需求_FastSAM3DBody.md`

| 模块 | 作用 | 自检 |
|---|---|---|
| fsb/contract.py | R1 合同校验器（`python -m fsb.contract <take>`；含 pkl 的 numpy 版本兼容检查） | 两段参考素材 OK |
| fsb/package.py | SMPL-X 序列 → 合同目录（pickle 协议 4，numpy 1.26 / 2.x 都能读） | tests/test_package_roundtrip.py |
| fsb/compare.py | 新旧对比报告（`python -m fsb.compare 旧 新 --out 目录`） | tests/test_compare.py |
| fsb/track.py | R2 主角锁定（路人排除） | tests/test_track.py |
| fsb/retarget.py · convert.py | MHR → SMPL-X 关节集骨骼重定向 + 整段转换 | tests/test_retarget_convert.py |
| fsb/world.py · smooth.py | 相机系 → GVHMR 世界系（脚步拟合地面）；去尖刺 + 切空间高斯 | tests/test_world_smooth.py |
| fsb/pipeline.py | 视频 → 合同目录的完整流水线（跟踪/锁主角/内参/推理/重算骨架/打包，可续跑） | tests/test_winpack.py |
| winpack/ | Windows 一键整合包：入口、自检、权重下载、交叉组装构建器、锁文件 | tests/test_winpack.py |
| addon/ | PoseCapture 插件补丁：加「身体前端 = FastSAM3DBody」 | tests/test_winpack.py + tests_blender/addon_fsb_e2e.py |
| cloud/ | 2026-10-04 云端实跑用的脚本（FSB_ROOT 可配） | — |

文档：docs/格式_交付合同.md、格式_SMPLX序列.md、格式_云端原始输出_MHR.md、方案_骨骼重定向.md、方案_Windows整合包.md、云端运行手册.md。

测试：`for f in tests/test_*.py; do python3 $f; done`（不需要 GPU）；Blender 端：`bash /home/sb/remote_kit_1.7.1/tools/mcd.sh run tests_blender/addon_fsb_e2e.py <blend> -- <补丁后插件> <FSB 包> <PKL 目录> <视频> <本仓库>`。

## 公开仓库里没有的文件（本地生成）

公开仓库 `Scar1etCrystal/fsb_mocap` 是本地仓库的净化镜像（`tools/publish_public.sh` 生成），少两样东西：

| 文件 | 为什么不公开 | 怎么得到 |
|---|---|---|
| `fsb/assets/mhr_rest.npz` | 从 SAM 3D Body 的 `mhr_model.pt` 导出（MHR 静止姿态关节），受 SAM License 约束；那份权重是 gated，公开派生数据等于绕过申请 | 自己申请并下载权重后：`python cloud/export_mhr_rest.py fsb/assets/mhr_rest.npz`（脚本里 `mhr_model.pt` 的路径按你的机器改） |
| `addon/gvhmr_pose_capture_fsb.py` | 是 PoseCapture 插件打过补丁的完整源码（插件本身不在本仓库发布） | `python addon/apply_fsb_patch.py <你的 gvhmr_pose_capture.py> -o addon/gvhmr_pose_capture_fsb.py` |

云端登录地址不进仓库：`cloud/ssh.sh` 从 `cloud/ssh_target.env`（gitignore）或环境变量 `AUTODL_HOST` / `AUTODL_PORT` 读。

