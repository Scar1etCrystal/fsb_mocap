"""往返测试：参考素材 merged → 中间格式 → 打包 → 合同校验通过，且 merged 与参考逐数组相同。"""
import glob
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fsb import contract, package, rot  # noqa: E402

REF = "/home/sb/fastsam_handoff/reference_output"
fails = 0


def check(name, ok, detail=""):
    global fails
    fails += 0 if ok else 1
    print(f"[{'PASS' if ok else 'FAIL'}] {name} :: {detail}")


for ref in sorted(glob.glob(os.path.join(REF, "*/"))):
    take = os.path.basename(ref.rstrip("/"))
    seq = package.seq_from_take(ref)
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, take)
        files = package.package_take(seq, out, hamer_stem="roundtrip")
        rep = contract.validate_take(out)
        check(f"{take[:9]} contract OK", rep["ok"], rep["errors"][:3])
        ref_m = contract.load_pkl(os.path.join(ref, "hamer", "merged.pkl"))
        new_m = contract.load_pkl(files["merged"])
        diffs = [f"{br}.{k}" for br in contract.BRANCHES for k in ref_m[br]
                 if not np.array_equal(ref_m[br][k], new_m[br][k]) and not k.endswith("hand_global_orient")]
        check(f"{take[:9]} merged arrays == reference (except *_hand_global_orient)", not diffs, diffs[:5])
        dt = {f"{br}.{k}": (str(ref_m[br][k].dtype), str(new_m[br][k].dtype)) for br in contract.BRANCHES
              for k in ref_m[br] if ref_m[br][k].dtype != new_m[br][k].dtype}
        check(f"{take[:9]} merged dtypes == reference", not dt, dt)
        info_ref = dict(ref_m["hamer_merge_info"]); info_new = dict(new_m["hamer_merge_info"])
        info_ref.pop("source"); info_new.pop("source")
        check(f"{take[:9]} merge_info == reference (source aside)", info_ref == info_new, (info_ref, info_new))
        # 手部原始 ↔ 重新合并互逆：用 hand_body_merge 的公式从 raw 反推手腕局部旋转，应与 body_pose 手腕一致
        raw = contract.load_pkl(files["hamer"])
        br = new_m["smpl_params_incam"]
        G = rot.chain_global_rotations(br["global_orient"], br["body_pose"])
        hg = np.einsum("ij,fhjk->fhik", package.RZ180, raw["global_orient"].astype(np.float64))
        hg[:, 0] = rot.mirror_lr(hg[:, 0])
        wl = np.linalg.inv(G[:, 18]) @ hg[:, 0]
        wr = np.linalg.inv(G[:, 19]) @ hg[:, 1]
        bp = br["body_pose"].reshape(-1, 21, 3)
        v = raw["global_orient_valid"].astype(bool)
        el = rot.geodesic_deg(wl[v[:, 0]], rot.rotvec_to_mat(bp[v[:, 0], 19])).max() if v[:, 0].any() else 0
        er = rot.geodesic_deg(wr[v[:, 1]], rot.rotvec_to_mat(bp[v[:, 1], 20])).max() if v[:, 1].any() else 0
        check(f"{take[:9]} re-merge of raw hands reproduces wrists (< 0.01°, float32 storage)", el < 0.01 and er < 0.01,
              f"L {el:.2e}° R {er:.2e}°")
        fl = rot.mirror_lr(raw["hand_pose"][:, 0].astype(np.float64))
        dl = rot.geodesic_deg(fl[v[:, 0]], rot.rotvec_to_mat(br["left_hand_pose"].reshape(-1, 15, 3))[v[:, 0]]).max()
        check(f"{take[:9]} raw left fingers = mirror of merged left (< 0.01°)", dl < 0.01, f"{dl:.2e}°")
        top_eq = open(files["top"], "rb").read() == open(files["merged"], "rb").read()
        check(f"{take[:9]} top-level pkl byte-identical to merged", top_eq)
print(f"==== {'0 FAIL' if not fails else f'{fails} FAIL'} ====")
sys.exit(1 if fails else 0)
