# fsb_frontend — FastSAM3DBody 替换 GVHMR+HaMeR 的转换层

需求合同：`/home/sb/fastsam_handoff/PoseCapture_Pack_code/前端替换需求_FastSAM3DBody.md`

| 模块 | 作用 | 自检 |
|---|---|---|
| fsb/contract.py | R1 合同校验器（`python -m fsb.contract <take>`） | 两段参考素材 OK |
| fsb/package.py | SMPL-X 序列 → 合同目录 | tests/test_package_roundtrip.py |
| fsb/compare.py | 新旧对比报告（`python -m fsb.compare 旧 新 --out 目录`） | tests/test_compare.py |
| fsb/track.py | R2 主角锁定 | tests/test_track.py |
| fsb/world.py | 相机系 → GVHMR 世界系 | tests/test_world_smooth.py |
| fsb/smooth.py | 去尖刺 + 切空间高斯 | tests/test_world_smooth.py |

文档：docs/格式_交付合同.md、docs/格式_SMPLX序列.md、docs/云端运行手册.md。
