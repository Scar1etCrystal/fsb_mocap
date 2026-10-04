# winpack — FastSAM3DBody Windows 一键整合包

方案、体积、验证情况：`../docs/方案_Windows整合包.md`。

| 文件 | 用途 |
|---|---|
| build_pack.py | 组装整合包（Linux / Windows 都能跑；`--dry-run`、`--estimate`、`--steps`、`--profile cu124/cu128`、`--cn`） |
| requirements_win.in / .lock.txt | 依赖（cu124 = 云端验证版）；`*_cu128.*` = RTX 50 系版（未验证）。锁文件由 `uv pip compile --python-platform x86_64-pc-windows-msvc --python-version 3.10` 生成 |
| weights_manifest.json · download_weights.py | 模型文件清单（大小 / 哈希）与下载校验（gated token、hf-mirror、GitHub 代理、续传） |
| runtime_env.py | 目录约定、与云端相同的推理开关、离线补丁（DINOv3 本地 hub、utils3d 别名）、中文路径处理 |
| run_video.py | 入口：视频 → 合同目录；`--from-raw` 只转换打包；最后一行 `[FSB] RESULT {json}` |
| check_env.py | 自检；`--smoke` 加载模型真跑一帧，报速度和峰值显存 |
| compare_takes.py | 两个 take 对比（验收：云端交付 vs 本机跑出） |

快速组装（不含权重）：`python build_pack.py --out D:/PoseCapture_Pack/FastSAM3DBody --steps python,wheels,install,app,launchers,check`
