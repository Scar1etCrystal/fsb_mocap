"""下载 / 校验整合包的模型文件（清单：weights_manifest.json，合计约 3.0 GB）。

    envs\\python.exe app\\winpack\\download_weights.py --dry-run            只列清单和本地状态，不联网
    envs\\python.exe app\\winpack\\download_weights.py --probe              联网探测每个来源可达/有无权限（不下载）
    envs\\python.exe app\\winpack\\download_weights.py --hf-token hf_xxx    下载缺的文件并校验
    envs\\python.exe app\\winpack\\download_weights.py --verify-only        只校验已有文件（大小 + md5/sha256）

SAM 3D Body 权重（facebook/sam-3d-body-dinov3）是 gated 仓库：先在 huggingface.co 该仓库页面申请访问，通过后用
自己的 token（或环境变量 HF_TOKEN）。国内网络：--hf-endpoint https://hf-mirror.com，--github-proxy https://ghfast.top/
（镜像对 gated 仓库同样要 token）。已下载的文件按大小/哈希跳过，可反复运行；中断的 URL 下载会续传。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
import urllib.request
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import runtime_env  # noqa: E402

MANIFEST = os.path.join(HERE, "weights_manifest.json")
UA = "pip/24.0 fsb-winpack"                               # 默认 Python-urllib UA 会被部分 CDN 拒（403）


def human(n):
    return "?" if n is None else (f"{n / 2 ** 30:.2f} GB" if n >= 2 ** 30 else f"{n / 2 ** 20:.1f} MB" if n >= 2 ** 20 else f"{n} B")


def file_hash(path, algo):
    h = hashlib.new(algo)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hf_cache_file(pack, item):
    """HF 缓存布局里的实际文件（refs/main → snapshots/<sha>/file）。"""
    src = item["source"]
    root = os.path.join(pack, item["dest"])
    ref = os.path.join(root, "refs", "main")
    if not os.path.isfile(ref):
        return None
    sha = open(ref, encoding="utf-8").read().strip()
    p = os.path.join(root, "snapshots", sha, src["file"])
    return p if os.path.exists(p) else None


def local_status(pack, item, deep=False):
    """返回 (状态, 说明)。状态：ok / missing / bad。deep=True 时算哈希。"""
    src = item["source"]
    if src["type"] == "github_zip":
        p = os.path.join(pack, item["dest"], item.get("check", ""))
        return ("ok", "") if os.path.isfile(p) else ("missing", "")
    p = hf_cache_file(pack, item) if src["type"] == "hf_cache" else os.path.join(pack, item["dest"])
    if not p or not os.path.isfile(p):
        return "missing", ""
    size = os.path.getsize(p)
    if item.get("size") and size != item["size"]:
        return "bad", f"大小 {size} ≠ {item['size']}"
    if deep:
        for algo in ("sha256", "md5"):
            if item.get(algo):
                got = file_hash(p, algo)
                if got != item[algo]:
                    return "bad", f"{algo} {got} ≠ {item[algo]}"
                return "ok", f"{algo} 一致"
    return "ok", "大小一致"


def source_url(item, hf_endpoint, gh_proxy):
    src = item["source"]
    if src["type"] in ("hf", "hf_cache"):
        rev = src.get("revision", "main")
        return f"{hf_endpoint.rstrip('/')}/{src['repo']}/resolve/{rev}/{src['file']}"
    if src["type"] == "url":
        return (gh_proxy or "") + src["url"] if "github.com" in src["url"] else src["url"]
    if src["type"] == "github_zip":
        return (gh_proxy or "") + f"https://github.com/{src['repo']}/archive/{src['commit']}.zip"
    raise ValueError(src["type"])


def probe(item, hf_endpoint, gh_proxy, token):
    url = source_url(item, hf_endpoint, gh_proxy)
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": UA})
    if token and item["source"]["type"].startswith("hf"):
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return f"HTTP {r.status}，{human(int(r.headers.get('Content-Length') or 0) or None)}"
    except urllib.error.HTTPError as e:
        hint = {401: "（gated：需要 token）", 403: "（gated：token 没有该仓库访问权，先去网页申请）"}.get(e.code, "")
        return f"HTTP {e.code}{hint}"
    except Exception as e:
        return f"不可达：{e!r}"[:160]


def download_url(url, dest, size=None, token=None):
    """urllib 下载 + 断点续传（.part）。"""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = dest + ".part"
    have = os.path.getsize(part) if os.path.isfile(part) else 0
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    if have:
        req.add_header("Range", f"bytes={have}-")
    with urllib.request.urlopen(req, timeout=60) as r:
        if have and r.status != 206:                        # 服务器不支持续传：重下
            have = 0
        total = (int(r.headers.get("Content-Length") or 0) + have) or size
        t0, done = time.time(), have
        with open(part, "ab" if have else "wb") as fh:
            while True:
                chunk = r.read(4 << 20)
                if not chunk:
                    break
                fh.write(chunk)
                done += len(chunk)
                el = max(time.time() - t0, 1e-3)
                print(f"\r  {human(done)} / {human(total)}  {(done - have) / el / 2 ** 20:.1f} MB/s", end="", flush=True)
    print()
    os.replace(part, dest)


def fetch(pack, item, hf_endpoint, gh_proxy, token):
    src = item["source"]
    dest = os.path.join(pack, item["dest"])
    if src["type"] == "hf":
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            return download_url(source_url(item, hf_endpoint, gh_proxy), dest, item.get("size"), token)
        local_dir = dest[: -len(src["file"])].rstrip("/\\")
        hf_hub_download(src["repo"], src["file"], revision=src.get("revision"), local_dir=local_dir, token=token,
                        endpoint=hf_endpoint)
    elif src["type"] == "hf_cache":
        try:
            from huggingface_hub import hf_hub_download
            hf_hub_download(src["repo"], src["file"], revision="main", cache_dir=os.path.dirname(dest), token=token,
                            endpoint=hf_endpoint)
        except ImportError:                                     # 宿主 Python 没装 huggingface_hub：手工写最小缓存布局
            rev = src["revision"]                               # 离线加载只看 refs/main → snapshots/<sha>/<file>
            download_url(source_url(item, hf_endpoint, gh_proxy), os.path.join(dest, "snapshots", rev, *src["file"].split("/")),
                         item.get("size"), token)
            os.makedirs(os.path.join(dest, "refs"), exist_ok=True)
            with open(os.path.join(dest, "refs", "main"), "w", encoding="utf-8") as fh:
                fh.write(rev)
    elif src["type"] == "url":
        download_url(source_url(item, hf_endpoint, gh_proxy), dest, item.get("size"))
    elif src["type"] == "github_zip":
        tmp = os.path.join(tempfile.gettempdir(), f"fsb_{src['commit'][:12]}.zip")
        download_url(source_url(item, hf_endpoint, gh_proxy), tmp)
        with zipfile.ZipFile(tmp) as zf:
            top = zf.namelist()[0].split("/")[0]
            ex = dest + ".extract"
            shutil.rmtree(ex, ignore_errors=True)
            zf.extractall(ex)
        shutil.rmtree(dest, ignore_errors=True)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.move(os.path.join(ex, top), dest)
        shutil.rmtree(ex, ignore_errors=True)
        with open(os.path.join(dest, "FSB_SOURCE.txt"), "w", encoding="utf-8") as fh:
            fh.write(f"{src['repo']}@{src['commit']}\n")
        os.remove(tmp)


def main(argv=None):
    ap = argparse.ArgumentParser(description="下载 / 校验整合包模型文件")
    ap.add_argument("--pack", default=None, help="整合包根目录（默认本脚本所在整合包）")
    ap.add_argument("--dry-run", action="store_true", help="只列清单和本地状态")
    ap.add_argument("--probe", action="store_true", help="联网探测来源（HEAD 请求，不下载）")
    ap.add_argument("--verify-only", action="store_true", help="只校验已有文件（含哈希）")
    ap.add_argument("--only", default="", help="逗号分隔的条目名，只处理这些")
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""))
    ap.add_argument("--hf-endpoint", default=os.environ.get("HF_ENDPOINT", "https://huggingface.co"))
    ap.add_argument("--github-proxy", default="", help="GitHub 下载前缀代理，如 https://ghfast.top/")
    ap.add_argument("--allow-token-to-mirror", action="store_true",
                    help="允许把 HF token 发给非 huggingface.co 的 --hf-endpoint（第三方镜像）；不加则对镜像不带 token")
    a = ap.parse_args(argv)
    runtime_env.safe_stdio()
    official = a.hf_endpoint.rstrip("/").lower() in ("https://huggingface.co", "http://huggingface.co")
    if a.hf_token and not official:
        if a.allow_token_to_mirror:
            print(f"注意：HF token 会发给第三方镜像 {a.hf_endpoint}")
        else:
            print(f"注意：{a.hf_endpoint} 不是 huggingface.co，不把 HF token 发给它（gated 权重会下载失败）；"
                  "确认信任该镜像再加 --allow-token-to-mirror")
            a.hf_token = ""
    pack = runtime_env.layout(a.pack)["pack"]
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ["HF_ENDPOINT"] = a.hf_endpoint
    items = json.load(open(MANIFEST, encoding="utf-8"))["files"]
    if a.only:
        keep = {s.strip() for s in a.only.split(",")}
        items = [i for i in items if i["name"] in keep]
    total = sum(i.get("size") or 0 for i in items)
    print(f"整合包：{pack}\n清单 {len(items)} 项，合计约 {human(total)}（不含 DINOv3 结构代码 zip）")
    failed = []
    for it in items:
        st, why = local_status(pack, it, deep=a.verify_only)
        line = f"- {it['name']:14s} {human(it.get('size')):>10s}  {st:7s} {why}  → {it['dest']}"
        if a.dry_run or a.verify_only or a.probe:
            print(line)
            if a.dry_run:
                print(f"      来源：{source_url(it, a.hf_endpoint, a.github_proxy)}"
                      + ("（gated，需要 token）" if it["source"].get("gated") else ""))
            if a.probe:
                print(f"      探测：{probe(it, a.hf_endpoint, a.github_proxy, a.hf_token)}")
            if a.verify_only and st != "ok":
                failed.append(it["name"])
            continue
        if st == "ok":
            print(line + "（已有，跳过）")
            continue
        if it["source"].get("gated") and not a.hf_token:
            print(line + "\n      ✗ gated 仓库需要 --hf-token（或环境变量 HF_TOKEN）")
            failed.append(it["name"])
            continue
        print(line + "\n      下载中 …")
        try:
            fetch(pack, it, a.hf_endpoint, a.github_proxy, a.hf_token or None)
            st, why = local_status(pack, it, deep=True)
            print(f"      {'✓' if st == 'ok' else '✗'} {st} {why}")
            if st != "ok":
                failed.append(it["name"])
        except Exception as exc:
            print(f"      ✗ 失败：{exc!r}"[:400])
            failed.append(it["name"])
    if failed:
        print(f"\n未完成：{failed}")
        return 1
    if not (a.dry_run or a.probe):
        print("\n全部就绪。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
