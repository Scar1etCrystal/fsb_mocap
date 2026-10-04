"""Blender 自带 numpy（4.5 LTS = 1.26.4）能否读我们的 pkl：先不装垫片读一遍，再按插件 _register_numpy_compat 的做法装垫片读一遍。"""
import glob
import pickle
import sys

import numpy

files = sorted(glob.glob("/home/sb/fastsam_handoff/reference_output/*/*.pkl") + glob.glob("/home/sb/fastsam_handoff/reference_output/*/*/*.pkl")
               + glob.glob("/home/sb/fsb_frontend/out/*/*.pkl") + glob.glob("/home/sb/fsb_frontend/out/*/*/*.pkl"))
extra = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
files += extra
print("BLENDER numpy", numpy.__version__, "python", sys.version.split()[0], "numpy._core preloaded:", "numpy._core" in sys.modules)


def try_all(tag):
    bad = 0
    for p in files:
        try:
            with open(p, "rb") as f:
                pickle.load(f, encoding="latin1")
            r = "OK"
        except Exception as e:
            r, bad = f"FAIL {e!r}"[:90], bad + 1
        print(f"  [{tag}] {r:6s} {'/'.join(p.split('/')[-3:])[:90]}")
    return bad


b0 = try_all("no-shim")
# 插件 gvhmr_pose_capture.py:5125 原样
try:
    import numpy._core.numeric as _nc
except ImportError:
    import numpy.core.numeric as _nc
if 'numpy._core' not in sys.modules:
    sys.modules['numpy._core'] = type(sys)('numpy._core')
if 'numpy._core.numeric' not in sys.modules:
    sys.modules['numpy._core.numeric'] = _nc
b1 = try_all("addon-shim")
print(f"RESULT no-shim fails={b0} addon-shim fails={b1}")
