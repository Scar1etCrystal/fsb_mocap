"""整合包 / 插件补丁的本地可测部分（不需要 GPU、torch、cv2）。"""
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "winpack"))
sys.path.insert(0, os.path.join(ROOT, "addon"))

from fsb import contract, pipeline  # noqa: E402
import runtime_env  # noqa: E402
import download_weights  # noqa: E402
import build_pack  # noqa: E402
import apply_fsb_patch  # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAILS.append(name)


def test_f_mm():
    K = pipeline.K_from_f_mm(1920, 1080, 24.0)               # GVHMR create_camera_sensor：对角线换算
    f = (1920 ** 2 + 1080 ** 2) ** 0.5 / (24 ** 2 + 36 ** 2) ** 0.5 * 24
    check("f_mm 换算同 GVHMR", abs(K[0, 0] - f) < 1e-9 and K[0, 2] == 960 and K[1, 2] == 540, f"fx={K[0, 0]:.2f}")


def test_pending_and_guard():
    with tempfile.TemporaryDirectory() as d:
        st = pipeline.pending_stages(d, pipeline.STAGES, 450)
        check("空 work：track/fov/infer 都要跑", st == ["track", "fov", "infer"], str(st))
        open(os.path.join(d, "track_raw.npz"), "wb").close()
        with open(os.path.join(d, "cam.json"), "w", encoding="utf-8") as fh:
            json.dump({"K": [[1000, 0, 32], [0, 1000, 18], [0, 0, 1]], "source": "moge2", "f_mm": 0.0}, fh)
        for p in range(3):
            open(os.path.join(d, f"sam3d_part_{p:04d}.npz"), "wb").close()
        open(os.path.join(d, "skel_part_0000.npz"), "wb").close()
        open(os.path.join(d, "skel_part_0001.npz"), "wb").close()
        todo, done = pipeline.pending_parts(d, 450)
        check("续跑：缺骨架的块算未完成", todo == [2] and done == {0, 1}, f"todo={todo} done={sorted(done)}")
        check("f_mm>0 时不需要 fov", "fov" not in pipeline.pending_stages(d, pipeline.STAGES, 450, f_mm=0)
              and pipeline.needs_gpu(d, pipeline.STAGES, 450))
        open(os.path.join(d, "skel_part_0002.npz"), "wb").close()
        check("全完成：不需要 GPU", not pipeline.needs_gpu(d, pipeline.STAGES, 450))
        # work 守卫：同一视频保留，换视频清空
        v1, v2 = os.path.join(d, "a.bin"), os.path.join(d, "b.bin")
        open(v1, "wb").write(os.urandom(1000)); open(v2, "wb").write(os.urandom(1000))
        meta = (64, 36, 30.0, 450)
        pipeline._guard_work(d, v1, meta)
        n_before = len(os.listdir(d))
        pipeline._guard_work(d, v1, meta)
        check("同一视频：中间结果保留", len(os.listdir(d)) == n_before)
        pipeline._guard_work(d, v2, meta)
        left = sorted(f for f in os.listdir(d) if f.endswith((".npz", ".json")) and f != "video_sig.json")
        check("换了视频：中间结果清空", left == [], str(left))


def test_resume_robustness():
    # 目录名带 [..]（yt-dlp 默认文件名"标题 [ID]"）：glob 不能把它当字符类
    with tempfile.TemporaryDirectory() as root:
        d = os.path.join(root, "My Dance [abc123]_work")
        os.makedirs(d)
        for p in range(2):
            open(os.path.join(d, f"sam3d_part_{p:04d}.npz"), "wb").close()
            open(os.path.join(d, f"skel_part_{p:04d}.npz"), "wb").close()
        todo, done = pipeline.pending_parts(d, 400)
        check("目录名带方括号：已完成的分块能找到", todo == [] and done == {0, 1}, f"todo={todo} done={sorted(done)}")
        # 换了焦距：旧内参和分块作废
        with open(os.path.join(d, "cam.json"), "w", encoding="utf-8") as fh:
            json.dump({"K": [[1000, 0, 32], [0, 1000, 18], [0, 0, 1]], "source": "moge2"}, fh)   # 旧格式：没有 f_mm 键
        check("旧 cam.json（MoGe）+ 不给焦距：内参有效", "fov" not in pipeline.pending_stages(d, ("fov",), 400))
        check("旧 cam.json（MoGe）+ 改给焦距 35mm：仍不需要跑 MoGe（用焦距现算）",
              "fov" not in pipeline.pending_stages(d, ("fov",), 400, f_mm=35.0))
        res = pipeline.stage_fov(os.path.join(d, "none.mp4"), d, None, (64, 36, 30.0, 400), f_mm=35.0)
        left = sorted(f for f in os.listdir(d) if f.startswith(("sam3d_part", "skel_part")))
        check("改了焦距：重算内参、清掉旧推理分块", res["source"] == "f_mm=35" and res["f_mm"] == 35.0 and left == [],
              f"{res['source']} {left}")
        res2 = pipeline.stage_fov(os.path.join(d, "none.mp4"), d, None, (64, 36, 30.0, 400), f_mm=35.0)
        check("焦距不变：直接复用", res2 == res)
        # 截断的 cam.json 当作要重算，不崩
        open(os.path.join(d, "cam.json"), "w").write('{"K": [[1')
        check("截断的 cam.json：判为要重算", "fov" in pipeline.pending_stages(d, ("fov",), 400))
        # 原子写：不留临时文件
        pipeline._atomic_savez(os.path.join(d, "x.npz"), a=np.arange(3))
        pipeline._atomic_json(os.path.join(d, "y.json"), {"a": 1})
        check("原子写入不留临时文件", not [f for f in os.listdir(d) if ".tmp" in f]
              and np.load(os.path.join(d, "x.npz"))["a"].tolist() == [0, 1, 2])


def test_ascii_copy():
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "舞蹈 视频.mp4")
        open(src, "wb").write(b"x" * 100)
        dst, copied = runtime_env.ascii_safe_copy(src, os.path.join(d, "work"))
        ok = copied and os.path.isfile(dst) and dst.isascii() and os.path.getsize(dst) == 100
        check("非 ASCII 路径复制到 ASCII", ok, dst)
        same, c2 = runtime_env.ascii_safe_copy(dst, os.path.join(d, "work"))
        check("ASCII 路径原样返回", same == dst and not c2)


def test_weights_status():
    items = json.load(open(download_weights.MANIFEST, encoding="utf-8"))["files"]
    names = [i["name"] for i in items]
    check("清单 7 项", names == ["sam3d_model", "sam3d_config", "mhr_model", "sam3d_license", "yolo_pose", "moge2_vits",
                                  "dinov3_hub"], str(names))
    with tempfile.TemporaryDirectory() as d:
        it = {"name": "x", "dest": "ck/x.bin", "size": 10, "md5": "781e5e245d69b566979b86e28d23f2c7",
              "source": {"type": "url", "url": "http://example.invalid/x"}}
        check("缺文件 → missing", download_weights.local_status(d, it)[0] == "missing")
        os.makedirs(os.path.join(d, "ck"))
        open(os.path.join(d, "ck/x.bin"), "wb").write(b"0123456789")
        st, why = download_weights.local_status(d, it, deep=True)
        check("大小+md5 一致 → ok", st == "ok", why)
        open(os.path.join(d, "ck/x.bin"), "wb").write(b"01234")
        check("大小不符 → bad", download_weights.local_status(d, it)[0] == "bad")
        # HF 缓存布局：refs/main → snapshots/<sha>/file
        it2 = {"name": "m", "dest": "hf/hub/models--a--b", "size": 3, "source": {"type": "hf_cache", "repo": "a/b", "file": "model.pt"}}
        root = os.path.join(d, "hf/hub/models--a--b")
        os.makedirs(os.path.join(root, "refs")); os.makedirs(os.path.join(root, "snapshots", "abc"))
        open(os.path.join(root, "refs", "main"), "w").write("abc")
        open(os.path.join(root, "snapshots", "abc", "model.pt"), "wb").write(b"123")
        check("HF 缓存布局识别", download_weights.local_status(d, it2)[0] == "ok")


def test_lock():
    for prof, torch_v in (("cu124", "torch==2.5.1+cu124"), ("cu128", "torch==2.7.1+cu128")):
        binary, build = build_pack.read_lock(prof)
        check(f"锁文件 {prof}", torch_v in binary and len(build) == 3 and "termcolor==3.3.0" in binary
              and "numpy==1.26.4" in binary and not any(s.startswith("moge") for s in binary),
              f"{len(binary)} 个 wheel + {len(build)} 个现场打包")


def test_pickle_protocol():
    take = os.path.join(ROOT, "out", "0001-0999_f_full_20260925_162525v2_fsb")
    if not os.path.isdir(take):
        print("[SKIP] 没有 out/ 交付 take")
        return
    for rel in ("gvhmr/hmr4d_results.pt_person-0.pkl", "hamer/merged.pkl"):
        b = open(os.path.join(take, rel), "rb").read()
        refs = set(re.findall(rb"numpy\.[_a-z]+\.[a-z_]+", b))
        check(f"{rel} 协议 4、不引用 numpy._core.numeric", b[1] == 4 and b"numpy._core.numeric" not in refs, str(sorted(refs)))


def test_from_raw_regression():
    raw = os.path.join(ROOT, "data", "t999")
    ref = os.path.join(ROOT, "out", "0001-0999_f_full_20260925_162525v2_fsb")
    if not (os.path.isdir(raw) and os.path.isdir(ref)):
        print("[SKIP] 没有 data/t999 或交付 take")
        return
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "t999")
        r = subprocess.run([sys.executable, os.path.join(ROOT, "winpack", "run_video.py"), "--from-raw", raw, "--out", out,
                            "--hamer-stem", "0001-0999_f_full_20260925_162525", "--fps", "30"], capture_output=True, text=True)
        res = [l for l in r.stdout.splitlines() if l.startswith("[FSB] RESULT ")]
        ok = r.returncode == 0 and res and json.loads(res[-1][13:])["ok"]
        check("run_video --from-raw 成功", bool(ok), r.stderr[-300:] if not ok else "")
        if not ok:
            return
        check("合同校验", contract.validate_take(out)["ok"])
        a = pickle.load(open(os.path.join(out, "hamer", "merged.pkl"), "rb"))
        b = pickle.load(open(os.path.join(ref, "hamer", "merged.pkl"), "rb"))
        mx = max(float(np.abs(np.asarray(a[br][k], float) - np.asarray(b[br][k], float)).max())
                 for br in ("smpl_params_global", "smpl_params_incam") for k in a[br])
        check("与交付 take 一致", mx < 1e-5, f"max|Δ|={mx:.1e}")


def test_addon_patch():
    src = "/home/sb/fastsam_handoff/PoseCapture_Pack_code/blender_addon/gvhmr_pose_capture.py"
    if not os.path.isfile(src):
        print("[SKIP] 没有原插件")
        return
    raw = open(src, "rb").read()
    text = raw[3:].decode("utf-8")
    new, report = apply_fsb_patch.patch_text(text)
    check("补丁 17 处全部命中", len(report) == len(apply_fsb_patch.EDITS) == 17, str(len(report)))
    compile(new, "patched", "exec")
    check("补丁后可编译", True)
    try:
        apply_fsb_patch.patch_text(new)
        check("重复打补丁会被拒绝", False)
    except RuntimeError:
        check("重复打补丁会被拒绝", True)
    old_lines = text.replace("\r\n", "\n").split("\n")
    new_lines = new.replace("\r\n", "\n").split("\n")
    import difflib
    sm = difflib.SequenceMatcher(None, old_lines, new_lines, autojunk=False)
    removed = sum(i2 - i1 for op, i1, i2, _, _ in sm.get_opcodes() if op in ("replace", "delete"))
    check("只改了少量原有行（其余原样）", removed <= 12, f"改动原有行 {removed}")
    lf_only = lambda t: t.count("\n") - t.count("\r\n")
    check("原有 LF 行数不变（插入行沿用 CRLF）", lf_only(new) == lf_only(text), f"{lf_only(text)} → {lf_only(new)}")


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        try:
            fn()
        except Exception as exc:
            import traceback
            traceback.print_exc()
            check(fn.__name__, False, repr(exc))
    print(f"==== {len(FAILS)} FAIL ====")
    sys.exit(1 if FAILS else 0)
