#!/usr/bin/env bash
# 本地仓库 → 公开镜像 github.com/Scar1etCrystal/fsb_mocap（main 分支）。本地仓库本身不改。
#
# 镜像相对本地少两样（README「公开仓库里没有的文件」说明了怎么在自己机器上生成）：
#   fsb/assets/mhr_rest.npz          SAM 3D Body 的 MHR 模型导出的数据，受 SAM License 约束（权重是 gated）
#   addon/gvhmr_pose_capture_fsb.py  PoseCapture 插件打过补丁的完整源码
# 另外旧提交里的 cloud/ssh.sh 含云端登录地址，换成占位符（这里只用通配，脚本里不出现真实地址）。
#
# 过滤规则固定、作者/时间原样保留 → 同一个原提交每次都得到同一个镜像提交，之后同步是快进，不需要强推。
#
# 用法：bash tools/publish_public.sh [--dry-run]
#   推送用的 ssh 选项放在 GIT_SSH_COMMAND 里（例如 -i <密钥>）；--dry-run 只生成镜像并打印要推的提交。
set -euo pipefail
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE="${PUBLIC_REMOTE:-git@github.com:Scar1etCrystal/fsb_mocap.git}"
BRANCH="$(git -C "$SRC" rev-parse --abbrev-ref HEAD)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
git clone -q --no-local --branch "$BRANCH" "$SRC" "$TMP/repo"
cd "$TMP/repo"
git remote remove origin
FILTER_BRANCH_SQUELCH_WARNING=1 git filter-branch --prune-empty --tree-filter '
    rm -f fsb/assets/mhr_rest.npz addon/gvhmr_pose_capture_fsb.py
    if [ -f cloud/ssh.sh ]; then
        sed -i -E "s/connect\.[A-Za-z0-9-]+\.seetacloud\.com/<AUTODL_HOST>/g; s/-p [0-9]{4,5} root@/-p <AUTODL_PORT> root@/g" cloud/ssh.sh
    fi' -- "$BRANCH" > /dev/null
# 自检：镜像的任何提交里都不能再有这两样文件和真实登录地址
if git log --all --name-only --format= | grep -qE '^(fsb/assets/mhr_rest\.npz|addon/gvhmr_pose_capture_fsb\.py)$'; then
    echo "镜像里还有不该公开的文件" >&2; exit 1
fi
if git grep -nE 'connect\.[A-Za-z0-9-]+\.seetacloud\.com|-p [0-9]{4,5} root@' $(git rev-list "$BRANCH") -- cloud/ssh.sh | grep -v '<AUTODL_' | grep -q .; then
    echo "镜像里还有云端登录地址" >&2; exit 1
fi
echo "镜像 $(git rev-list --count "$BRANCH") 个提交，HEAD $(git rev-parse --short "$BRANCH")（本地 $(git -C "$SRC" rev-parse --short HEAD)）"
if [ "${1:-}" = "--dry-run" ]; then
    git log --oneline "$BRANCH" | head -20
    exit 0
fi
git push "$REMOTE" "$BRANCH:main"
