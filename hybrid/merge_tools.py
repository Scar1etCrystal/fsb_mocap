"""调用用户自己的 hand_body_merge.py（移交包原件，不改代码），参数 = PoseCapture 插件默认（STABLE 预设）。"""
from __future__ import annotations

import os
import pickle
import subprocess
import sys
import warnings

import numpy as np

HANDOFF_HAMER = "/home/sb/fastsam_handoff/PoseCapture_Pack_code/HAMER"
STAGE1_HAMER = "/home/sb/freemocap_doctor/pack_patch_stage1"       # 只多带 hand_keypoint_conf 透传，其余与原件逐行相同
ADDON_DEFAULT = ["--smooth", "one_euro", "--wrist_cutoff", "1.0", "--wrist_beta", "0.007", "--finger_cutoff", "2.0",
                 "--finger_beta", "0.01", "--max_jump_deg", "90", "--hold_missing_frames", "6", "--interp_missing_frames", "15"]


def run_merge(gvhmr_pkl, hamer_pkl, output, *, fps=None, script_dir=HANDOFF_HAMER, extra=(), python=sys.executable):
    """与插件 _run_hand_body_merge 相同的命令行；返回 stdout。"""
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    script = os.path.join(script_dir, "hand_body_merge.py")
    cmd = [python, script, "--gvhmr_pkl", gvhmr_pkl, "--hamer_pkl", hamer_pkl, "--output", output, *ADDON_DEFAULT]
    if fps:
        cmd += ["--gvhmr_fps", str(fps)]
    cmd += list(extra)
    env = dict(os.environ, PYTHONPATH=HANDOFF_HAMER + os.pathsep + os.environ.get("PYTHONPATH", ""))
    r = subprocess.run(cmd, cwd=HANDOFF_HAMER, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"merge failed rc={r.returncode}\n{r.stdout[-2000:]}\n{r.stderr[-3000:]}")
    return r.stdout


def _numpy_compat():
    """与插件 _register_numpy_compat 相同：numpy 1.x 里读 numpy≥2 写的协议 5 pkl（numpy._core.numeric）。"""
    try:
        import numpy._core.numeric  # noqa: F401
    except ImportError:
        import numpy.core.numeric as _nc
        sys.modules.setdefault("numpy._core.numeric", _nc)


def load(p):
    _numpy_compat()
    with open(p, "rb") as fh, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pickle.load(fh, encoding="latin1")


def max_abs_diff(a, b, keys=None):
    out = {}
    for br in ("smpl_params_global", "smpl_params_incam"):
        for k in (keys or sorted(set(a[br]) & set(b[br]))):
            x, y = np.asarray(a[br][k]), np.asarray(b[br][k])
            if x.shape != y.shape:
                out[f"{br}.{k}"] = f"shape {x.shape} vs {y.shape}"
            elif x.dtype.kind in "fc":
                out[f"{br}.{k}"] = float(np.abs(x.astype(float) - y.astype(float)).max()) if x.size else 0.0
            else:
                out[f"{br}.{k}"] = 0.0 if np.array_equal(x, y) else "differs"
    return out
