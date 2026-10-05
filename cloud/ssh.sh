#!/usr/bin/env bash
# 云端（AutoDL）连接：专用密钥 + 专用 known_hosts（首次连接记录主机指纹）。
# 主机 / 端口 / 用户不进仓库：写在本目录的 ssh_target.env（已 gitignore），或用环境变量覆盖：
#   AUTODL_HOST=<实例地址>  AUTODL_PORT=<端口>  AUTODL_USER=root  AUTODL_KEY=~/.ssh/autodl_ed25519
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$HERE/ssh_target.env" ] && . "$HERE/ssh_target.env"
: "${AUTODL_HOST:?设置 AUTODL_HOST（或写进 cloud/ssh_target.env）}"
: "${AUTODL_PORT:?设置 AUTODL_PORT（或写进 cloud/ssh_target.env）}"
exec ssh -i "${AUTODL_KEY:-$HOME/.ssh/autodl_ed25519}" -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=20 \
    -o UserKnownHostsFile="$HOME/.ssh/known_hosts_autodl" -o StrictHostKeyChecking=accept-new \
    -o ServerAliveInterval=30 -p "$AUTODL_PORT" "${AUTODL_USER:-root}@$AUTODL_HOST" "$@"
