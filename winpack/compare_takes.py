"""两个 take 目录逐帧对比（fsb.compare 的整合包入口）。验收用：云端交付的 take vs 本机整合包跑出来的 take。

    envs\\python.exe app\\winpack\\compare_takes.py <take A> <take B> --out <报告目录>
同为 FastSAM3DBody 的两次运行，预期身体局部旋转平均差 < 1°、没有常值偏差；报告写到 --out（compare.md / compare.json）。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import runtime_env  # noqa: E402

runtime_env.setup()
from fsb import compare  # noqa: E402

if __name__ == "__main__":
    sys.exit(compare.main())
