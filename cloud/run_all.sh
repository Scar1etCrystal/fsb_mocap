#!/usr/bin/env bash
# 两段素材整片：999 段（跟踪/选人/焦距已做）→ 推理；1499 段 全部阶段。后台跑：nohup setsid bash run_all.sh &
cd /root/autodl-tmp
source /etc/network_turbo >/dev/null 2>&1
source fsb_frontend/cloud/env.sh
export HF_ENDPOINT=https://hf-mirror.com
LOG=logs/fsb_full.log
: > $LOG
echo "start $(date +%T)" >> $LOG
python fsb_frontend/cloud/run_take.py --video videos/0001-0999_f_full_20260925_162525.mkv --out fsb_work/t999 --stages infer >> $LOG 2>&1
echo "t999 rc=$? $(date +%T)" >> $LOG
mkdir -p fsb_work/t1499
python fsb_frontend/cloud/run_take.py --video videos/0001-1499_f_full_20260925_165539.mkv --out fsb_work/t1499 --stages track,select,sheet,fov,infer >> $LOG 2>&1
echo "t1499 rc=$? $(date +%T)" >> $LOG
echo DONE >> $LOG
