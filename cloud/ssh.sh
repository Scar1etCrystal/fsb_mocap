#!/usr/bin/env bash
# 云端（AutoDL）连接：专用密钥 + 专用 known_hosts（首次连接记录主机指纹）
exec ssh -i "$HOME/.ssh/autodl_ed25519" -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=20 \
    -o UserKnownHostsFile="$HOME/.ssh/known_hosts_autodl" -o StrictHostKeyChecking=accept-new \
    -o ServerAliveInterval=30 -p <AUTODL_PORT> root@<AUTODL_HOST> "$@"
