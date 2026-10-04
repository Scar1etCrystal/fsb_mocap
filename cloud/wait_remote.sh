#!/usr/bin/env bash
# 本机等云端后台任务：每 30 s 看一次，日志里出现成功标记、或进程没了（失败/结束）就退出并打印日志尾。
#   bash cloud/wait_remote.sh <远端日志> <成功标记> <进程匹配（pgrep -f 正则，首字母加方括号防自匹配，如 "[h]amer_locked"）> [最长分钟=60]
LOG=$1; OK=$2; PROC=$3; MAXMIN=${4:-60}
cd "$(dirname "$0")/.."
for i in $(seq 1 $((MAXMIN * 2))); do
    s=$(timeout 40 bash cloud/ssh.sh "grep -c '$OK' $LOG 2>/dev/null; pgrep -f '$PROC' >/dev/null && echo RUN || echo GONE" < /dev/null 2>/dev/null | tr '\n' ' ')
    case "$s" in
        [1-9]*) echo "OK $(date +%T)"; break ;;
        *GONE*) echo "GONE $(date +%T)"; break ;;
    esac
    sleep 30
done
timeout 40 bash cloud/ssh.sh "tail -c 3000 $LOG" < /dev/null
