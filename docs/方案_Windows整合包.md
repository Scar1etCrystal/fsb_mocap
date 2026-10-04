# FastSAM3DBody 并入 PoseCapture 整合包：方案 v2

> **2026-10-04 晚更新。** PoseCapture 本身就是整合包：GVHMR、HAMER、WiLoR 各带 Windows Python 环境，插件按目录调用，WiLoR 已经在借用 GVHMR 的环境。
> 用户选定的组合又是"FastSAM3DBody 身体 + HaMeR 手"。所以 v2 不再给 FastSAM3DBody 单独带一套 4 GB 环境，改为：
> - **借 HAMER 的 Python 3.11 环境**，加一个约 0.2 GB 的补包目录；
> - **HAMER 目录里只多一个脚本**（锁主角的 HaMeR 运行器）；
> - 默认的 GVHMR + HaMeR 链路逐位不变。
>
> v1（独立环境整合包，上午的方案）原样保留在附录 A，给不装 HAMER 的人用。

## 1. 结论

| | v1 独立环境 | **v2 并入 PoseCapture（推荐）** |
|---|---|---|
| Python 环境 | 自带嵌入式 Python 3.10 + 74 个包，4.08 GB | **借 `HAMER\envs`**（Python 3.11 / torch 2.5.1+cu121 / numpy 2.4.4），只补 HAMER 环境里没有的 9 个包，下载 55.5 MB、装好约 0.2 GB |
| 整合包增加 | ≈ 6.9 GB | **≈ 3.0 GB**（模型权重 2.78 GB 占大头，两版一样） |
| 手 | FSB 自带手（用户验收：远不如 HaMeR） | **HaMeR**：锁主角、每帧检测 + 合并前预处理（阶段 B 的 B7） |
| 能不能在同一环境跑 | — | **能**：云端用和 HAMER 同版本的环境（py3.11 / torch 2.5.1 / numpy 2.4.4）从视频跑完整流程，结果见 §4 |
| 要不要在 Windows 上编译 | 不要 | 不要：9 个补包都有 `cp311-win_amd64` 或纯 Python wheel（逐个查过 PyPI） |
| GVHMR 链路 | 不变 | 不变（身体选 GVHMR 时插件逐位走原流程） |

## 2. 目录结构

```
PoseCapture_Pack\
  GVHMR\  WiLoR\  ffmpeg\                    ← 不动
  HAMER\                                      ← 环境不动，只新增一个脚本
    envs\python.exe                           Python 3.11 / torch 2.5.1+cu121 / numpy 2.4.4（FastSAM3DBody 也用它）
    hamer_video.py                            原样（身体选 GVHMR 时照旧用它）
    hamer_locked.py                           新增：锁主角、每帧检测、导出遮挡信号（身体选 FastSAM3DBody 时用）
    hand_body_merge.py  mano2smplx_merge.py   原样（合并脚本一行不改）
    _DATA\                                    原样（hamer.ckpt、vitpose wholebody、MANO）
  FastSAM3DBody\                              ← 新增，约 3.0 GB
    site-packages\                            补包（只装 HAMER 环境里没有的）：ultralytics 8.4.172、ultralytics-thop、
                                              ultralytics-platform、lap 0.5.13、roma 1.6.1、polars 1.44.2 + polars-runtime-32、
                                              psutil、nvidia-ml-py；另放 MoGe / utils3d / pipeline 源码（纯 Python）
    app\Fast-SAM-3D-Body\  app\fsb\  app\winpack\
    app\fsb\hands.py                          新增：合并前预处理（阶段 B 的 B7：俯仰补偿 + 门控补帧 + 小臂扭转分配）
    checkpoints\                              2.78 GB（与 v1 相同：model.ckpt、mhr_model.pt、MoGe-2 ViT-S、YOLO11m-pose、DINOv3 结构代码）
    run_video.bat  check_env.bat  download_weights.bat  使用说明.txt
  blender_addon\                              补丁版：身体 GVHMR | FastSAM3DBody × 手 HaMeR | WiLoR | 融合 | FSB 自带
```

`run_video.bat` 和插件调 FastSAM3DBody 都用 `..\HAMER\envs\python.exe`，子进程环境：
- `PYTHONPATH = FastSAM3DBody\site-packages;FastSAM3DBody\app;FastSAM3DBody\app\Fast-SAM-3D-Body`（补包放最前）；
- 沿用插件现成的 `_prepend_python_env_paths`，给 HAMER 环境的 DLL 目录加 PATH。

HAMER 环境里一个包都不装、不升级，补包和 HAMER 环境里已有的包没有重名（构建时检查）。

## 3. 一次捕捉（身体 = FastSAM3DBody，手 = HaMeR，默认组合）

```
1 身体   HAMER\envs\python.exe FastSAM3DBody\app\winpack\run_video.py --video … --out …
         跟踪 → 锁主角 → 内参（MoGe-2）→ 逐帧推理 → 打包      → gvhmr\…person-0.pkl + main_track.npz
2 手     HAMER\envs\python.exe HAMER\hamer_locked.py --video … --main_track main_track.npz --tta 1
         人框 = 主角跟踪框，ViTPose 只跑主角，每帧检测         → hamer\<stem>_hamer_data.pkl（与原版同格式）+ _hamer_raw.npz
3 预处理 fsb.hands（B7）：俯仰补偿；门控补帧（HaMeR 与 FSB 手朝向分歧 / YOLO 腕置信度 / 角速度 / 扭转翻转
         + 本帧没检到手 / 主角手部置信度 < 0.5 / 重投影误差 > 半个手框）；小臂扭转跨帧连续地分给小臂
                                                                → 改过的 hamer pkl + 改过肘部的身体 pkl
4 合并   HAMER\hand_body_merge.py 原件、插件默认参数（不开 --forearm_twist） → merged.pkl
5 之后   插件原有的保存 / 导入 / 重定向 / MoCap Doctor，不用改
```

- 身体选 GVHMR：1–3 不跑，`hamer_video.py` + 合并原样，逐位不变。
- 身体选 FastSAM3DBody、手选 WiLoR / 融合：照原流程合并，不锁主角（阶段 D 没做），没验证过，界面上提示"推荐 HaMeR"。
- 手选"FSB 自带"：跳过 2–4，直接用 FSB 的手（v1 的行为）。
- `hamer_locked.py` 的多裁剪（`--tta 7`）和路人手（`--others`）只是分析用的。B7 只要原版那 1 个裁剪（`--tta 1`），速度见 §6。

## 4. 版本兼容：同一个环境跑 FastSAM3DBody 行不行

HAMER 环境和 FastSAM3DBody 的验证环境差在：Python 3.11 / 3.10，numpy 2.4.4 / 1.26.4，scipy 1.17.1 / 1.15.3，opencv 4.13 / 4.11，timm 1.0.26 / 1.0.30，pytorch-lightning 2.6.1 / 2.6.6，huggingface_hub 1.13 / 2.1.1，trimesh 4.12 / 5.1。torch 两边都是 2.5.1（cu121 / cu124）。

**云端实测**（阶段 B ②，2026-10-04）：
- 环境：照 HAMER 版本搭（py3.11.17 / torch 2.5.1 / numpy 2.4.4 / scipy 1.17.1 / opencv 4.13.0 / timm 1.0.26 / lightning 2.6.1）。
- 补包：dry-run 解析只缺上面 9 个包（另加 MoGe 源码）。
- 跑法：从视频跑完整流程（跟踪 → 锁主角 → 内参 → 推理 → 打包），和 py3.10 / numpy 1.26 环境的中间结果逐项比。

| | t999 | t1499 |
|---|---|---|
| 跟踪：检测数、轨迹号 | 完全相同（3099 个检测，12 条轨迹） | 完全相同 |
| 跟踪框 / 主角框 | 最大差 1.0 / 0.15 像素 | 1.2 / 0.31 像素 |
| 内参 fx（MoGe-2） | 1615.52 / 1615.52（差 1e-5） | 2320.49 / 2320.49（差 1e-4） |
| 打包后身体局部旋转，与已交付 take 比 | 均值 0.05°、P95 0.09° | 均值 0.2°、P95 0.34° |
| 其中腿 / 脚 / 脊柱 / 手臂 P95 | ≤ 0.04° | ≤ 0.07° |
| 手腕（FSB 自己的手腕） | 均值 0.38°、P95 0.82°、最大 25.6° | 均值 1.9°、P95 11°、最大 40° |
| 根位移 | 逐帧差 0.0 m；路径 7.24 / 7.24 m | 0.02 m；15.78 / 15.82 m |
| 推理速度 | 274 ms/帧（py3.10 环境当时 407） | 347 ms/帧（当时 466） |

结论：身体在两种环境里只差浮点级，共用 HAMER 环境可行。
- 手腕个别帧差得大，是 FSB 自己的手部估计在模棱两可的帧上不稳。混合方案里手腕取自 HaMeR，不受影响。
- 新环境没有变慢。

## 5. 体积

| 项 | v1 独立环境 | v2 并入 PoseCapture |
|---|---|---|
| Python + 依赖 | 4.08 GB（瘦身后） | 补包约 0.2 GB（下载 55.5 MB：polars 运行库 51 MB 占大头） |
| 代码 | 0.004 GB | 0.005 GB（多 fsb/hands.py 和 hamer_locked.py） |
| 模型权重 | 2.78 GB | 2.78 GB |
| **合计（解压后）** | **≈ 6.9 GB** | **≈ 3.0 GB** |

## 6. 硬件

- 显卡要求跟 HAMER 环境走：torch 2.5.1+cu121 支持 RTX 20/30/40、GTX 16；**RTX 50 系（sm_120）不支持**，现有 HaMeR 本来就有这个限制。v2 不改变它。要支持 50 系得给 HAMER 换 cu128 环境，不在本方案范围。
- 显存：FastSAM3DBody 和 HaMeR 是先后两个进程，峰值取两者较大值。云端没记峰值；`check_env.bat --smoke` 会实测。建议 ≥ 8 GB，Blender 同时开着会抢显存。
- 速度（云端 RTX 4080 SUPER）：
  - FastSAM3DBody 在 HAMER 同版本环境里 274 / 347 ms/帧（两段素材；py3.10 环境当时是 407 / 466 ms/帧）。
  - HaMeR 锁主角、1 个裁剪：实测每帧 0.15 s（ViTPose 50 ms + 双手 HaMeR 100 ms），1000 帧约 3 分钟（含加载模型半分钟）。
    原版 HaMeR 本来也是逐帧推理（隔帧的只是检人），锁主角版少了 detectron2、ViTPose 只跑 1 个人，不会更慢。
  - 合计（身体 + 手 + 合并）：每 1000 帧约 8–9 分钟。

## 7. 构建与插件要改的（阶段 C，等确认后做）

1. `build_pack.py` 新增 `--profile posecap`（设为默认）：
   - 不下嵌入式 Python；
   - wheels 只下 `requirements_posecap_overlay.lock.txt` 里的 9 个 `cp311-win_amd64` / 纯 Python wheel（`--python-version 3.11 --abi cp311`）；
   - install 到 `FastSAM3DBody\site-packages`；
   - app 步骤顺带把 `hamer_locked.py` 拷进 `HAMER\`；
   - check 步骤核对 `HAMER\envs` 的 torch / numpy / opencv / timm 版本，并确认补包与它无重名。
2. `check_env.py`：用 `HAMER\envs\python.exe` 跑；`--smoke` 依次真跑一帧 FastSAM3DBody 和一帧 HaMeR。
3. 插件补丁：
   - `_fsb_python()` 先找 `FastSAM3DBody\envs`（v1 布局），没有就用 `HAMER\envs`（v2）；
   - 子进程 PYTHONPATH 前置补包；
   - 手部下拉框按身体前端显示；
   - 身体选 FSB + 手选 HaMeR 时走 §3 的 2–4 步。
4. 回归：
   - 不改设置时 GVHMR + HaMeR 结果逐位不变（v1 已有同样的测试）；
   - FSB + HaMeR 在 Blender 4.5 里端到端跑通；
   - 用两段测试视频和阶段 B 交付的 B7 对比，预期浮点级差别。

## 8. 风险

| 风险 | 对策 |
|---|---|
| 用户自己升级了 HAMER 环境里的包（比如 numpy） | check_env 按版本报出；补包不依赖具体小版本，大版本变了才报错 |
| 补包和 HAMER 环境出现重名 | 构建时只装 HAMER 环境里没有的包（云端 dry-run 已验证这套过滤），check 再核一次 |
| ultralytics 首次运行要写 `%APPDATA%\Ultralytics`、缺 lap 时会现场 pip install | 沿用 v1 的 `YOLO_CONFIG_DIR` / `YOLO_OFFLINE`；lap 在补包里 |
| DINOv3 结构代码 torch.hub 先联网查 GitHub | 沿用 v1 的 `patch_torch_hub`（本次云端也是靠它离线跑通的：不打补丁时被 GitHub 断开连接） |
| HaMeR 锁主角要 FSB 的主角框 | 身体选 GVHMR 时仍用原版 `hamer_video.py`；以后可以改用 GVHMR 自己的人框锁主角 |
| 快舞视频 HaMeR 本身拿不准（1499 段门控 27–33%） | B7 的门控按绝对阈值，干净视频几乎不多砍；嫌太"敷衍"可退回 B4 |

## 附录 A：v1 独立环境整合包（2026-10-04 上午的原方案，保留作备选）

> 2026-10-04。结论：**可行**，代码和构建脚本都已写好并在本机能测的部分全部测过；剩下的是"真下载 + 在一台 Windows N 卡机器上跑一遍验收"（大环境和权重按要求没下）。
> 形态照 PoseCapture 整合包：`PoseCapture_Pack\FastSAM3DBody\`，与 GVHMR、HAMER、ffmpeg 并排；插件里多一个「身体前端」选项，GVHMR 链路原样保留。

### A.1. 结论与依据

| 问题 | 结论 | 依据（本机实测） |
|---|---|---|
| 依赖在 Windows 上有没有现成二进制 | **全有**：74 个包里 71 个有 `cp310-win_amd64` 或纯 Python wheel，另 3 个（utils3d、pipeline 的 git 版、antlr4 源码包）都是纯 Python，构建时现场打成 wheel | `uv pip compile --python-platform x86_64-pc-windows-msvc --python-version 3.10` 解析出完整锁文件；逐个查 PyPI / PyTorch 索引元数据确认每个版本都有 Windows wheel |
| 要不要在 Windows 上编译东西 | **不要** | flash-attn：只有 ViT 骨干用，且 `try/except` 包着，我们用 DINOv3 骨干；torch.compile / TensorRT：默认关（`USE_COMPILE=0`、`FOV_TRT=0`）；detectron2：只有可视化模块 import，推理链不碰；MHR：没装 pymomentum 时走 `torch.jit.load(mhr_model.pt)`（TorchScript，跨平台）——云端就是这么跑的 |
| 能不能完全离线 | **能** | DINOv3：FastSAM3DBody 只认 `~/.cache/torch/hub/...`，否则去 GitHub 查默认分支（云端就卡在这）→ `runtime_env.patch_torch_hub` 把它改成读包内目录；MoGe-2：权重放包内 HF 缓存 + `HF_HUB_OFFLINE=1`；YOLO：`YOLO_OFFLINE`，ByteTrack 要的 `lap` 预装（否则 ultralytics 会运行时 pip install） |
| 新前端产物插件能不能直接用 | **能** | 产物与 GVHMR+HAMER+合并完全同格式（R1 合同）；补丁版插件在 Blender 4.5 里走完整捕捉流程 19/19 通过（§6）；pkl 在 Blender 自带 numpy 1.26.4 下无需任何垫片即可读 |
| 结果会不会和云端不一样 | 应该只有浮点级差别 | 同权重、同推理开关（`runtime_env.VALIDATED_ENV` = 云端 `env.sh`）、同 fp32；整合包入口用云端原始结果重新打包，与交付 take **逐位一致**（max\|Δ\| = 0） |

### A.2. 体积（实测，单位 GiB）

只读 wheel 的 zip 目录（HTTP 分段请求，每个包取 ~1 MB）算出了解压后的准确大小，没下整包。

| 项 | cu124（默认，= 云端验证环境） | cu128（RTX 50 系，未验证） |
|---|---|---|
| 依赖 wheel 下载 | 2.52（其中 torch 2.34） | 3.23（torch 3.05） |
| 装好的 site-packages | 4.86 | 5.96 |
| 默认瘦身（删 torch 的 `.lib` + 头文件，推理用不到） | −0.78 → **4.08** | −0.84 → **5.12** |
| 嵌入式 Python 3.10.11 | 0.016 | 同 |
| 代码（FastSAM3DBody 快照 + fsb + winpack + MoGe 源码） | 0.004 | 同 |
| 模型权重 | **2.78**（model.ckpt 1.96 + mhr_model.pt 0.65 + MoGe-2 ViT-S 0.13 + YOLO11m-pose 0.04） + DINOv3 结构代码 0.02 | 同 |
| **整合包合计（解压后）** | **≈ 6.9** | ≈ 7.9 |
| 7z 压缩后（估计：权重几乎不可压，CUDA DLL 约压一半） | ≈ 5 | ≈ 5.7 |
| 每 1000 帧中间结果（outputs\\_work，成功后自动删） | ≈ 11 MB | 同 |

对照：PoseCapture 现有包的 GVHMR+HAMER 两个环境就 11.3 GB（剥离清单），这个包一个环境 4 GB 左右。

### A.3. 硬件要求

- NVIDIA 显卡。cu124 版 = torch 2.5.1+cu124：RTX 20/30/40、GTX 16；驱动 ≥ 528（CUDA 12 次版本兼容，建议 ≥ 551）。
- RTX 50 系（sm_120）：torch 2.5.1 没有这个架构（插件里给 GVHMR 写过同样的报错说明）→ 用 `--profile cu128`（torch 2.7.1+cu128，驱动 ≥ 570）。锁文件已解析好（与 cu124 只差 torch / torchvision / sympy），**没在云端验证过**，先跑 `check_env.bat`。
- 显存：**未实测**（云端没记峰值）。fp32、单人、512 输入；权重 2.1 GB。`check_env.bat` 的 smoke 测试会打印实测峰值。建议 ≥ 8 GB；Blender 同时开着会抢显存。
- 内存 ≥ 16 GB（加载 2.1 GB 的 checkpoint 时 CPU 内存峰值约为其 2–3 倍）。
- 速度：云端 RTX 4080 实测 407 / 466 ms/帧（两段素材）+ 跟踪约 1 分钟 + 模型加载约半分钟；1000 帧约 8–9 分钟。Windows 上同卡预计相近；低端卡按算力折算（未测）。

### A.4. 目录结构

```
PoseCapture_Pack\
  GVHMR\  HAMER\  WiLoR\  ffmpeg\  blender_addon\      ← 原有，不动
  FastSAM3DBody\                                        ← 新增（插件偏好可留空，自动识别）
    envs\python.exe                嵌入式 Python 3.10.11；python310._pth 打开 site
    envs\Lib\site-packages\        torch 2.5.1+cu124 等 74 个包（锁文件 winpack\requirements_win.lock.txt）
    app\Fast-SAM-3D-Body\          FastSAM3DBody 源码快照 d72aa369（只取 sam_3d_body\ + tools\）
    app\fsb\                       重定向 / 转换 / 打包 / 校验（含 assets\mhr_rest.npz）
    app\winpack\                   run_video / check_env / download_weights / build_pack
    app\third_party\MoGe\moge\     MoGe 源码 74fbce0（不经 pip，见 §8）
    checkpoints\sam-3d-body-dinov3\  model.ckpt  model_config.yaml  assets\mhr_model.pt  LICENSE
    checkpoints\yolo\yolo11m-pose.pt
    checkpoints\torch_hub\facebookresearch_dinov3_main\   DINOv3 结构代码（权重在 model.ckpt 里）
    checkpoints\hf\hub\models--Ruicheng--moge-2-vits-normal\
    outputs\                       单独跑时的结果；outputs\_work\ 中间结果（可续跑）
    run_video.bat  check_env.bat  download_weights.bat  vc_redist.x64.exe  使用说明.txt  BUILD_INFO.json
```

### A.5. 运行流程（一次捕捉）

插件选「FastSAM3DBody」后，`PoseCapturePipeline` 在原来的"安全复制 / GIF→MP4 / 裁剪 / VFR→CFR"之后分叉：

```
envs\python.exe app\winpack\run_video.py --video <处理后的视频> --out FastSAM3DBody\outputs\<stem>
        --take-name <stem> --hamer-stem <stem> --clean-work [--f-mm <焦距>]
  1 跟踪   YOLO11m-pose + ByteTrack（所有人）                         → track_raw.npz
  2 锁主角 fsb.track：轨迹切段/接段/插值，路人排除                     → main_track.npz/json
  3 内参   MoGe-2 ViT-S 抽 15 帧取中位数（给了 f_mm 就不跑；失败退回 60°）→ cam.json
  4 推理   逐帧 FastSAM3DBody + 用最终参数重算 MHR 骨架（每 200 帧一块）  → sam3d_part / skel_part
  5 打包   骨骼重定向 → SMPL-X 关节集、去尖刺+平滑、脚步拟合地面 → 合同目录 + 合同校验
  最后一行  [FSB] RESULT {"ok": true, "merged": …, "person": …, "fps": …}
```

插件把 `merged.pkl`（身体+手）或 `gvhmr\…person-0.pkl`（仅身体）照原流程 `_save_pkl`：捕捉文件夹里同样是 `<名字>.pkl + 视频 + gvhmr\ + hamer\`（多一个 `fsb_meta.json`），之后的导入、重定向到 MMR、MoCap Doctor 都不用改。中途取消或失败：`outputs\_work\<stem>` 保留，再捕捉同一个视频从断点续跑（文件签名不同会自动清掉重算）。

### A.6. 已做完并验证的（本机，无 GPU）

| 项 | 文件 | 验证 |
|---|---|---|
| 平台无关流水线（云端脚本的整合版，可续跑） | `fsb/pipeline.py` | 云端原始结果重新打包 = 交付 take（max\|Δ\| = 0）；Python 3.10 + numpy 1.26.4（= 整合包版本）下续跑通过 |
| 整合包入口 / 运行时环境 / 离线补丁 | `winpack/run_video.py`、`runtime_env.py` | 同上；中文路径自动复制到 ASCII 路径；只为还要干活的阶段检查模型文件 |
| 环境自检（含 `--smoke` 真跑一帧、报速度和峰值显存） | `winpack/check_env.py` | 本机跑通（缺包时逐项报 FAIL） |
| 权重下载（gated token、hf-mirror、GitHub 代理、断点续传、大小 + md5/sha256 校验） | `winpack/download_weights.py`、`weights_manifest.json` | `--dry-run --probe`：7 个来源全部可达，gated 文件无 token 时 401（符合预期）；DINOv3 结构代码真下载解压通过 |
| 构建器（Linux / Windows 都能跑；交叉组装 Windows 包） | `winpack/build_pack.py` | `--estimate` 得出 §2 数字；实际跑了 python / app / launchers / check 步骤；交叉安装机制用 6 个包实测（含 Windows `.pyd`、git 包、源码包） |
| 锁文件（cu124 / cu128） | `winpack/requirements_win*.lock.txt` | uv 解析；静态扫描 import 可达性时发现 DINOv3 的 hubconf 顶层要 `termcolor`（原清单漏了），已补 |
| 插件补丁（与 GVHMR 并存） | `addon/apply_fsb_patch.py`、`gvhmr_pose_capture_fsb.py`、`fsb_frontend.diff` | 17 处锚点全部唯一命中；只改原有 9 行、新增 172 行；BOM 和 CRLF/LF 混排原样；重复打会被拒绝；**Blender 4.5 里完整捕捉 19/19 通过**（注册、偏好、面板显隐、捕捉前检查、进度实时显示、捕捉文件夹结构、合同校验、帧率回填） |
| 单测 | `tests/test_winpack.py` | 26 项全过；原有 5 个测试文件全过 |

**顺带查出并修掉的问题：** 本机（numpy 2.4）写的交付 pkl 用的是协议 5，引用 `numpy._core.numeric`，干净的 numpy 1.26（Blender 4.x 自带）读不了——插件 import 时装的兼容垫片能救（参考素材的 merged.pkl 也是这样），但不该依赖它。打包器改为固定协议 4，两段交付 take 用 numpy 1.26.4 重新生成（数值不变，对比报告逐字节相同），现在不装垫片也能读。合同校验加了这一项检查。

### A.7. 还没验证的（需要真下载 + Windows N 卡机器）

1. 嵌入式 Python 真启动、`python310._pth` 生效、torch 的 CUDA DLL 加载（大概率缺的是 VC++ 运行库 → 包里带 `vc_redist.x64.exe`，check_env 会提示）。
2. GPU 推理全链路在 Windows 上的速度 / 显存 / 与云端结果的差（预期浮点级）。
3. cu128 版在 RTX 50 系上（torch 2.7 + 同样代码，理论上没问题，没跑过）。
4. Windows Defender 第一次扫描 4 GB DLL 会让首次启动很慢（几十秒到几分钟），属正常。

**第一次在 Windows 上的验收步骤（约 30 分钟，大头是下载）：**
1. `python build_pack.py --out D:\PoseCapture_Pack\FastSAM3DBody --hf-token hf_xxx`（国内加 `--cn`；或在本机 Linux 组装好 7z 拷过去）。
2. `check_env.bat` → 最后一行"全部通过"，记下单帧 ms 和峰值显存。
3. 把云端那两段测试视频拖到 `run_video.bat`。
4. `envs\python.exe app\winpack\compare_takes.py <交付的 out\…_fsb> <outputs\…> --out <报告目录>`：
   两边都是 FastSAM3DBody，预期身体局部旋转平均差 < 1°、无系统偏差；超过就说明 Windows 环境和云端不一致，查 `BUILD_INFO.json` 的版本。
5. 插件打补丁 → Blender 里选「FastSAM3DBody」捕捉同一段视频，导入看效果。

### A.8. 几个设计决定

- **嵌入式 Python + pip `--target` 交叉安装，而不是 conda 环境**：和 ComfyUI 便携包同一路子；整个环境可在 Linux 上组装（不需要 Windows 也能出包），解压即用、可搬家；不需要编译。
- **MoGe 用源码、不走 pip**：MoGe 最新元数据要 numpy≥2、flex-gemm（CUDA 扩展，Linux）、gradio，推理都用不到；云端就是源码 + numpy 1.26.4 跑的。`moge.model.v2` 对 utils3d 的 `try: import utils3d_moge except: import utils3d` 有兜底，再加 `utils3d.pt → utils3d.torch` 别名（云端核对过用到的两个函数）。
- **只锁推理链用到的包**：云端环境 300 个包（含 TensorRT、RealSense、可视化），整合包 74 个。
- **默认瘦身**：torch 的 `.lib` 和头文件只在编译 C++ 扩展 / torch.compile 时用，删掉省 0.8 GB；`--no-slim` 可保留。
- **插件导入动画仍用 GVHMR 目录里的 SMPL / SMPL-X 身体模型**（许可文件）：所以选 FastSAM3DBody 时检查的是"GVHMR 目录 + body_models"，不检查 GVHMR 的 Python 环境。
- **仅视频**：FastSAM3DBody 前端不接单张图片（时间处理/落地要序列），图片仍走 GVHMR，插件会直接提示。

### A.9. 许可与分发

- FastSAM3DBody 代码 MIT；MoGe 代码和 MoGe-2 权重 MIT；DINOv3 结构代码为 DINOv3 License；YOLO11（ultralytics）AGPL-3.0。
- **SAM 3D Body 权重（facebook/sam-3d-body-dinov3）是 gated（SAM License），整合包默认不带**：每个使用者自己在 HF 申请、用自己的 token 跑 `download_weights.bat`。自己用可以把权重一起打进 7z。
- `app\fsb\assets\mhr_rest.npz`（896 KB，MHR 静止姿态的关节位置/朝向，重定向标定用）是从 `mhr_model.pt` 导出的，同样受 SAM License 约束；给别人分发时一并说明（或删掉后用 `cloud/export_mhr_rest.py` 在对方机器上重新导出）。

### A.10. 风险与对策

| 风险 | 对策 |
|---|---|
| RTX 50 系 | cu128 版（已有锁文件），check_env 会准确报"架构不在 torch 支持列表" |
| 驱动太旧 / 没装 VC++ 运行库 | check_env 逐项报出；包里带 vc_redist.x64.exe |
| 中文 / 过长路径 | run_video 自动把中文路径的视频复制到 ASCII 路径；check_env 提示整合包放纯英文浅路径 |
| 显存不够（Blender 同时占用） | 关掉 Blender 视口的 Cycles/EEVEE 预览再捕捉；或先用 run_video.bat 单独跑，Blender 里再"导入已有 pkl" |
| MoGe 失败 | 自动退回 60° 视角（只影响深度尺度与落地），日志会写明；也可在插件里填焦距（f_mm）跳过 MoGe |
| 杀软误报 / 拦截 python.exe | 把整合包目录加白名单 |
| 中途断电 / 取消 | outputs\_work 可续跑；视频变了会自动清空重算 |
