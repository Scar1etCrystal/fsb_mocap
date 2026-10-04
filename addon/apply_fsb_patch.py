"""给 PoseCapture 插件（gvhmr_pose_capture.py）加 FastSAM3DBody 身体前端，与 GVHMR 并存（默认仍是 GVHMR）。

    python apply_fsb_patch.py <原插件.py> [-o <输出.py>]      不给 -o 就原地改（先备份成 .bak_fsb）
    python apply_fsb_patch.py <插件.py> --check               只检查能不能打（锚点是否都唯一命中）

改动（全部带 [FSB] 标记，可搜索）：
  - 偏好设置：新增「FastSAM3DBody 路径」fsb_dir（在 PoseCapture_Pack 下名为 FastSAM3DBody 时可留空）
  - 设置面板：新增「身体前端」GVHMR / FastSAM3DBody；选 FastSAM3DBody 时隐藏手部后端/相机模式
  - 捕捉：选 FastSAM3DBody 时检查整合包 + GVHMR 目录里的身体模型（导入动画要用；GVHMR 的 Python 环境不需要），
    跑 <fsb>/app/winpack/run_video.py，产物（gvhmr/ + hamer/ + merged.pkl，
    与 GVHMR+HAMER+合并完全同格式）按原流程存进捕捉文件夹；BODY 模式存 gvhmr pkl，FULL 模式存 merged.pkl
  - _run_cmd 新增 progress_prefix：以 "[FSB]" 开头的进度行实时显示在面板上
  - GVHMR / HAMER / WiLoR 的代码路径一行没删；批量捕捉、重新生成身体等其它操作仍走 GVHMR

原文件是 UTF-8 BOM + CRLF/LF 混合换行：本脚本按字节保留原样，插入行沿用锚点所在行的换行符。
"""
from __future__ import annotations

import argparse
import difflib
import os
import shutil
import sys

MARK = "[FSB]"

# ---------------------------------------------------------------- 插入的代码块

GET_FSB_DIR = '''

def get_fsb_dir():
    """Return FastSAM3DBody pack directory: preference, else <PoseCapture_Pack>/FastSAM3DBody if present. [FSB]"""
    p = _prefs()
    if p and getattr(p, "fsb_dir", ""):
        return p.fsb_dir
    try:
        candidate = os.path.join(_pack_root(), "FastSAM3DBody")
    except Exception:
        return ""
    return candidate if os.path.isdir(candidate) else ""
'''

FSB_HELPERS = '''

def _fsb_path(*parts):
    """Join FastSAM3DBody pack root + parts. [FSB]"""
    return os.path.join(get_fsb_dir(), *parts)


def _fsb_python():
    """Interpreter bundled with the FastSAM3DBody pack (Windows: envs/python.exe). [FSB]"""
    for parts in (("envs", "python.exe"), ("envs", "bin", "python"), ("envs", "python")):
        path = _fsb_path(*parts)
        if os.path.isfile(path):
            return path
    return _fsb_path("envs", "python.exe")


def _fsb_pack_error():
    """User-facing error when the FastSAM3DBody pack is not usable, else "". [FSB]"""
    root = get_fsb_dir()
    if not root or not os.path.isdir(root):
        return "请在路径设置中设置 FastSAM3DBody 路径（整合包目录，内含 envs/、app/、checkpoints/）"
    if not os.path.isfile(_fsb_python()):
        return f"未找到 FastSAM3DBody Python：{_fsb_python()}"
    if not os.path.isfile(_fsb_path("app", "winpack", "run_video.py")):
        return "FastSAM3DBody 整合包不完整：缺 app/winpack/run_video.py"
    return ""
'''

RUN_FSB_METHOD = '''    def _run_fsb_pipeline(self, work_path, stem):
        """[FSB] FastSAM3DBody frontend: body + hands in one pass.

        Runs <fsb>/app/winpack/run_video.py with the pack's own Python. It writes <fsb>/outputs/<stem>/ with
        gvhmr/hmr4d_results.pt_person-0.pkl, hamer/<stem>_hamer_data.pkl and hamer/merged.pkl (same formats as
        GVHMR + HAMER + hand_body_merge.py); that is then saved exactly like a GVHMR capture.
        An interrupted run resumes from <fsb>/outputs/_work/<stem>/ the next time the same video is captured.
        """
        if self.is_image:
            self.error = "FastSAM3DBody 前端目前只支持视频/GIF；图片请用 GVHMR 前端"
            self._log(self.error)
            return
        pack_error = _fsb_pack_error()
        if pack_error:
            self.error = pack_error
            self._log(pack_error)
            return
        if self.camera_motion != 'static':
            self._log(f"注意：FastSAM3DBody 按固定机位处理，相机模式 {self.camera_motion} 不生效")
        if self.person_idx:
            self._log("注意：FastSAM3DBody 自动锁定主角，人物编号不生效")
        out_dir = _fsb_path("outputs", stem)
        cmd = [_fsb_python(), _fsb_path("app", "winpack", "run_video.py"), "--video", work_path,
               "--out", out_dir, "--take-name", stem, "--hamer-stem", stem, "--clean-work"]
        if self.f_mm and self.f_mm > 0:
            cmd += ["--f-mm", str(self.f_mm)]
        self._log("步骤 1/1：正在运行 FastSAM3DBody（身体 + 手部一次完成，可能需要一些时间）...")
        rc, out = self._run_cmd(cmd, cwd=get_fsb_dir(), progress_prefix="[FSB]")
        if self._cancelled:
            return
        result = None
        for line in reversed(out.splitlines()):
            if line.startswith("[FSB] RESULT "):
                try:
                    result = json.loads(line[len("[FSB] RESULT "):])
                except Exception:
                    result = None
                break
        if rc != 0 or not result or not result.get("ok"):
            reason = (result or {}).get("error") or f"exit code {rc}"
            self.error = (f"FastSAM3DBody failed: {reason}\\n"
                          f"--- FastSAM3DBody output ---\\n{self._error_tail(out, limit=1600)}")
            self._log(self.error)
            return
        body_pkl = result.get("person") or os.path.join(out_dir, "gvhmr", "hmr4d_results.pt_person-0.pkl")
        merged_pkl = result.get("merged") or os.path.join(out_dir, "hamer", "merged.pkl")
        src_pkl = merged_pkl if self.mode == 'FULL' else body_pkl
        if not os.path.isfile(src_pkl):
            self.error = f"FastSAM3DBody output not found: {src_pkl}"
            self._log(self.error)
            return
        self._log(f"FastSAM3DBody 完成：{result.get('frames')} 帧，{result.get('seconds')} 秒")
        if not self.src_fps and result.get("fps"):           # 没有 ffprobe 时用前端读到的帧率（同一个视频文件）
            self.src_fps = float(result["fps"])
            self._log(f"Source video fps (from FastSAM3DBody): {self.src_fps:.3f}")
        saved = self._save_pkl(src_pkl)
        self.output_pkl = saved
        self._log(f"FastSAM3DBody output saved: {saved}")

'''

COLLECT_FSB = '''
        # [FSB] FastSAM3DBody take folder: <fsb>/outputs/{stem}/ (gvhmr/, hamer/, fsb_meta.json)
        fsb_root = get_fsb_dir()
        fsb_out = os.path.join(fsb_root, "outputs", stem) if fsb_root else ""
        if fsb_out and os.path.isdir(fsb_out):
            try:
                for sub in ("gvhmr", "hamer"):
                    if os.path.isdir(os.path.join(fsb_out, sub)):
                        shutil.copytree(os.path.join(fsb_out, sub), os.path.join(subfolder, sub), dirs_exist_ok=True)
                meta_json = os.path.join(fsb_out, "fsb_meta.json")
                if os.path.isfile(meta_json):
                    shutil.copy2(meta_json, os.path.join(subfolder, "fsb_meta.json"))
                shutil.rmtree(fsb_out)
                self._log(f"FastSAM3DBody intermediates moved to: {subfolder}")
            except Exception as e:
                self._log(f"Warning: could not move FastSAM3DBody intermediates: {e}")
'''

# ---------------------------------------------------------------- 编辑表
# (说明, 范围起点锚点 or None, 原文行（连续、整行、忽略行尾）, 新文本, 方式)  方式：after / before / replace

EDITS = [
    ("get_fsb_dir", None, [
        'def get_wilor_dir():',
        '    """Return WiLoR root directory from addon preferences."""',
        '    p = _prefs()',
        '    return p.wilor_dir if p and p.wilor_dir else ""',
    ], GET_FSB_DIR, "after"),
    ("_fsb_path / _fsb_python / _fsb_pack_error", None, [
        'def _wilor_path(*parts):',
        '    """Join WiLoR root + parts."""',
        '    return os.path.join(get_wilor_dir(), *parts)',
    ], FSB_HELPERS, "after"),
    ("safe_copy_file：没有 GVHMR 目录时的落脚处", "def safe_copy_file(src):", [
        '    safe_dir = _gvhmr_path("inputs", "demo")',
    ], '''    if get_gvhmr_dir() or not get_fsb_dir():
        safe_dir = _gvhmr_path("inputs", "demo") if get_gvhmr_dir() else os.path.join(tempfile.gettempdir(), "pose_capture_inputs")
    else:
        safe_dir = _fsb_path("outputs", "_inputs")  # [FSB] FastSAM3DBody-only setup: no GVHMR dir
''', "replace"),
    ("PoseCapturePipeline.__init__ 参数 body_frontend", "class PoseCapturePipeline:", [
        '                 trim_start_frame=0, trim_end_frame=0,',
        '                 on_complete=None, on_progress=None):',
        '        self.filepath = filepath',
    ], '''                 trim_start_frame=0, trim_end_frame=0,
                 on_complete=None, on_progress=None, body_frontend='GVHMR'):
        self.filepath = filepath
        self.body_frontend = (body_frontend or 'GVHMR').upper()  # [FSB] 'GVHMR' | 'FASTSAM3DB'
''', "replace"),
    ("_run_cmd 参数 progress_prefix", "class PoseCapturePipeline:", [
        '    def _run_cmd(self, cmd, cwd=None, timeout=3600):',
    ], '''    def _run_cmd(self, cmd, cwd=None, timeout=3600, progress_prefix=None):  # [FSB] progress_prefix
''', "replace"),
    ("_run_cmd 实时转发进度行", "    def _run_cmd(self, cmd, cwd=None, timeout=3600", [
        '                stdout_lines.append(clean_line)',
        '                if len(stdout_lines) % 20 == 0:',
    ], '''                stdout_lines.append(clean_line)
                if progress_prefix and clean_line.startswith(progress_prefix):  # [FSB] forward progress at once
                    self._log(clean_line[:200])
                elif len(stdout_lines) % 20 == 0:
''', "replace"),
    ("_run_pipeline 分支", "    def _run_pipeline(self):", [
        '        # Save final stem for intermediate file collection',
        '        self._work_stem = stem',
    ], '''
        # [FSB] FastSAM3DBody frontend: body + hands in one pass, same capture-folder layout as GVHMR+HAMER
        if self.body_frontend == 'FASTSAM3DB':
            self._run_fsb_pipeline(work_path, stem)
            return
''', "after"),
    ("_run_fsb_pipeline 方法", "class PoseCapturePipeline:", [
        '    def _save_pkl(self, src_pkl):',
    ], RUN_FSB_METHOD, "before"),
    ("_collect_intermediate_files 收 FastSAM3DBody 产物", "    def _collect_intermediate_files(self, subfolder):", [
        '            except Exception as e:',
        '                self._log(f"Warning: could not move fusion intermediates: {e}")',
    ], COLLECT_FSB, "after"),
    ("_cleanup_pipeline_residue 清未收走的 take 目录", "    def _cleanup_pipeline_residue(self):", [
        '            if wilor_dir:',
        '                candidates.append(os.path.join(wilor_dir, "outputs", f"{stem}_wilor"))',
    ], '''            fsb_root = get_fsb_dir()  # [FSB] uncollected take dir; outputs/_work/<stem> is kept for resume
            if fsb_root:
                candidates.append(os.path.join(fsb_root, "outputs", stem))
''', "after"),
    ("偏好设置 fsb_dir", "class PoseCapturePreferences(bpy.types.AddonPreferences):", [
        '    wilor_dir: bpy.props.StringProperty(',
        '        name="WiLoR 路径",',
        '        description="WiLoR 根目录（包含 wilor_video.py、pretrained_models/、mano_data/）",',
        "        subtype='DIR_PATH',",
        '        default="",',
        '    )',
    ], '''
    fsb_dir: bpy.props.StringProperty(  # [FSB]
        name="FastSAM3DBody 路径",
        description="FastSAM3DBody 整合包目录（包含 envs/、app/、checkpoints/）；放在 PoseCapture_Pack 下名为 FastSAM3DBody 时可留空",
        subtype='DIR_PATH',
        default="",
    )
''', "after"),
    ("偏好设置界面 fsb_dir", "class PoseCapturePreferences(bpy.types.AddonPreferences):", [
        '        layout.prop(self, "wilor_dir")',
    ], '''        layout.prop(self, "fsb_dir")  # [FSB]
''', "after"),
    ("设置属性 body_frontend", "class PoseCaptureSettings(bpy.types.PropertyGroup):", [
        "    mode: bpy.props.EnumProperty(",
    ], '''    body_frontend: bpy.props.EnumProperty(  # [FSB]
        name="身体前端",
        description="姿态估计前端。GVHMR：原链路（GVHMR 身体 + HAMER/WiLoR 手部）；FastSAM3DBody：身体和手一次出，脚步动作更完整，需要 FastSAM3DBody 整合包",
        items=[('GVHMR', 'GVHMR', 'GVHMR 身体 + HAMER/WiLoR 手部（原链路）'),
               ('FASTSAM3DB', 'FastSAM3DBody', '身体 + 手部一次完成；固定机位；画面有路人时自动锁定主角')],
        default='GVHMR',
    )

''', "before"),
    ("设置面板", "class POSE_PT_settings(bpy.types.Panel):", [
        '        layout.prop(props, "gender")',
        '        layout.prop(props, "mode")',
        "        if props.mode == 'FULL':",
        '            layout.prop(props, "hand_backend")',
        '            layout.prop(props, "smooth_preset")',
        '        layout.prop(props, "camera_motion")',
    ], '''        layout.prop(props, "body_frontend")  # [FSB]
        use_fsb = props.body_frontend == 'FASTSAM3DB'
        layout.prop(props, "gender")
        layout.prop(props, "mode")
        if props.mode == 'FULL' and not use_fsb:
            layout.prop(props, "hand_backend")
            layout.prop(props, "smooth_preset")
        if use_fsb:
            layout.label(text="FastSAM3DBody：身体+手一次出，固定机位，自动锁定主角", icon='INFO')
        else:
            layout.prop(props, "camera_motion")
''', "replace"),
    ("捕捉：环境检查", "class POSE_OT_capture(bpy.types.Operator):", [
        "        if not get_gvhmr_dir() or not os.path.isdir(get_gvhmr_dir()):",
        "            self.report({'ERROR'}, \"请在路径设置中设置 GVHMR 路径\")",
        "            return {'CANCELLED'}",
        "",
        '        gvhmr_python = _gvhmr_path("envs", "python.exe")',
        "        if not os.path.isfile(gvhmr_python):",
        "            self.report({'ERROR'}, f\"未找到 GVHMR Python：{gvhmr_python}\")",
        "            return {'CANCELLED'}",
        "",
        "        if props.mode == 'FULL':",
    ], '''        use_fsb = getattr(props, "body_frontend", 'GVHMR') == 'FASTSAM3DB'  # [FSB]
        if use_fsb:
            fsb_error = _fsb_pack_error()
            if fsb_error:
                self.report({'ERROR'}, fsb_error)
                return {'CANCELLED'}
            # 导入动画（import_pose_to_blender）仍读 GVHMR 目录里的 SMPL/SMPL-X 身体模型；GVHMR 的 Python 环境不需要
            if not get_gvhmr_dir() or not os.path.isdir(_gvhmr_path("inputs", "checkpoints", "body_models")):
                self.report({'ERROR'}, "导入动画需要 GVHMR 目录里的身体模型（inputs/checkpoints/body_models），请在路径设置中设置 GVHMR 路径")
                return {'CANCELLED'}

        if not use_fsb and (not get_gvhmr_dir() or not os.path.isdir(get_gvhmr_dir())):
            self.report({'ERROR'}, "请在路径设置中设置 GVHMR 路径")
            return {'CANCELLED'}

        gvhmr_python = _gvhmr_path("envs", "python.exe")
        if not use_fsb and not os.path.isfile(gvhmr_python):
            self.report({'ERROR'}, f"未找到 GVHMR Python：{gvhmr_python}")
            return {'CANCELLED'}

        if props.mode == 'FULL' and not use_fsb:
''', "replace"),
    ("捕捉：图片不走 FastSAM3DBody", "class POSE_OT_capture(bpy.types.Operator):", [
        "        if ext not in IMAGE_EXTS and ext not in VIDEO_EXTS and ext not in GIF_EXTS:",
        "            self.report({'ERROR'}, f\"不支持的文件类型：{ext}\")",
        "            return {'CANCELLED'}",
    ], '''        if use_fsb and ext in IMAGE_EXTS:  # [FSB]
            self.report({'ERROR'}, "FastSAM3DBody 前端目前只支持视频/GIF；图片请用 GVHMR 前端")
            return {'CANCELLED'}
''', "after"),
    ("捕捉：把前端传给流水线", "class POSE_OT_capture(bpy.types.Operator):", [
        "            on_complete=_on_pipeline_complete,",
        "            on_progress=None,",
        "        )",
    ], '''            on_complete=_on_pipeline_complete,
            on_progress=None,
            body_frontend=getattr(props, "body_frontend", 'GVHMR'),  # [FSB]
        )
''', "replace"),
]


def _split(text):
    """[(内容, 行尾)]，行尾 ∈ {'\\r\\n', '\\n', ''}。"""
    out = []
    for raw in text.splitlines(keepends=True):
        if raw.endswith("\r\n"):
            out.append((raw[:-2], "\r\n"))
        elif raw.endswith("\n"):
            out.append((raw[:-1], "\n"))
        else:
            out.append((raw, ""))
    return out


def _find(lines, block, start=0, scope=None):
    """block 在 lines 里（从 scope 锚点之后）第一次出现的下标；scope=None 时必须全文唯一。"""
    end = len(lines)
    if scope is not None:
        hits = [i for i, (c, _) in enumerate(lines) if c.rstrip().startswith(scope.rstrip())]
        if len(hits) != 1:
            raise LookupError(f"范围锚点命中 {len(hits)} 次：{scope!r}")
        start, end = hits[0], min(len(lines), hits[0] + 1500)     # 只在锚点后 1500 行内找，免得改到别的类
    n = len(block)
    hits = [i for i in range(start, end - n + 1)
            if all(lines[i + k][0].rstrip() == block[k].rstrip() for k in range(n))]
    if not hits:
        raise LookupError(f"没找到：{block[0]!r} …")
    if scope is None and len(hits) != 1:
        raise LookupError(f"命中 {len(hits)} 次（应唯一）：{block[0]!r} …")
    return hits[0]


def patch_text(text):
    if MARK in text:
        raise RuntimeError("这个文件已经打过 FastSAM3DBody 补丁（含 [FSB] 标记）")
    lines = _split(text)
    report = []
    for name, scope, block, new, how in EDITS:
        i = _find(lines, block, scope=scope)
        eol = lines[i][1] or "\r\n"
        new_lines = [(c, eol) for c in new.split("\n")[:-1]] if new.endswith("\n") else [(c, eol) for c in new.split("\n")]
        if new.startswith("\n") and how == "after":
            pass                                            # 开头空行已在 new_lines 里
        if how == "after":
            j = i + len(block)
            lines[j:j] = new_lines
        elif how == "before":
            lines[i:i] = new_lines
        elif how == "replace":
            lines[i:i + len(block)] = new_lines
        report.append(f"  ✓ {name}（第 {i + 1} 行，行尾 {'CRLF' if eol == chr(13) + chr(10) else 'LF'}）")
    return "".join(c + e for c, e in lines), report


def main(argv=None):
    ap = argparse.ArgumentParser(description="PoseCapture 插件加 FastSAM3DBody 前端")
    ap.add_argument("addon", help="gvhmr_pose_capture.py")
    ap.add_argument("-o", "--out", help="输出文件（默认原地修改，先备份 .bak_fsb）")
    ap.add_argument("--check", action="store_true", help="只检查")
    ap.add_argument("--diff", help="另存 unified diff 到这个文件")
    a = ap.parse_args(argv)
    raw = open(a.addon, "rb").read()
    bom = raw.startswith(b"\xef\xbb\xbf")
    text = raw[3:].decode("utf-8") if bom else raw.decode("utf-8")
    try:
        new, report = patch_text(text)
    except (LookupError, RuntimeError) as exc:
        print(f"✗ {exc}")
        return 1
    print("\n".join(report))
    if a.check:
        print(f"可以打补丁（{len(EDITS)} 处）。")
        return 0
    out = a.out or a.addon
    if not a.out:
        shutil.copy2(a.addon, a.addon + ".bak_fsb")
        print(f"已备份：{a.addon}.bak_fsb")
    with open(out, "wb") as fh:
        fh.write((b"\xef\xbb\xbf" if bom else b"") + new.encode("utf-8"))
    if a.diff:
        d = difflib.unified_diff(text.replace("\r\n", "\n").splitlines(keepends=True),
                                 new.replace("\r\n", "\n").splitlines(keepends=True),
                                 fromfile="a/" + os.path.basename(a.addon), tofile="b/" + os.path.basename(a.addon))
        with open(a.diff, "w", encoding="utf-8", newline="\n") as fh:
            fh.writelines(d)
        print(f"diff：{a.diff}")
    print(f"已写：{out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
