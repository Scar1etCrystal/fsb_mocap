#!/usr/bin/env bash
# 阶段 B ③：两段素材锁主角重跑 HaMeR。后台跑：setsid nohup bash run_hamer_b.sh > /root/autodl-tmp/logs/hamer_b.log 2>&1 < /dev/null &
set -uo pipefail
PY=/root/autodl-tmp/envs/hamer311/bin/python
RUN=/root/autodl-tmp/fsb_frontend/cloud/hamer/hamer_locked.py
W=/root/autodl-tmp/fsb_work
OUT=/root/autodl-tmp/hamer_b
cd /root/autodl-tmp/HAMER_code                    # _DATA、third-party/ViTPose 按当前目录找
echo "start $(date +%T)"
"$PY" "$RUN" --video /root/autodl-tmp/videos/0001-0999_f_full_20260925_162525.mkv --main_track $W/t999/main_track.npz \
    --track_raw $W/t999/track_raw.npz --others --out_dir $OUT/t999 ${EXTRA:-}
echo "t999 rc=$? $(date +%T)"
"$PY" "$RUN" --video /root/autodl-tmp/videos/0001-1499_f_full_20260925_165539.mkv --main_track $W/t1499/main_track.npz \
    --track_raw $W/t1499/track_raw.npz --others --out_dir $OUT/t1499 ${EXTRA:-}
echo "t1499 rc=$? $(date +%T)"
echo ALL_DONE
