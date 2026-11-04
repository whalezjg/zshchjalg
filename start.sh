#!/bin/sh
# 容器启动入口：进入项目目录，常驻运行 app.py；进程退出后自动拉起
cd "$(dirname -- "$0")" || exit 1

while true; do
    python3 app.py
    echo "[start.sh] app.py 退出（code=$?），3 秒后重启" >&2
    sleep 3
done
