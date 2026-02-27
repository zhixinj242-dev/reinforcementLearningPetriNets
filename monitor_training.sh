#!/bin/bash

# 监控所有训练日志的进度

LOG_DIR="train_logs"

if [ ! -d "$LOG_DIR" ]; then
    echo "日志目录 $LOG_DIR 不存在"
    exit 1
fi

while true; do
    clear
    echo "======================================================================"
    echo "训练进度监控 - $(date '+%Y-%m-%d %H:%M:%S')"
    echo "======================================================================"
    echo ""
    
    for logfile in "$LOG_DIR"/*.log; do
        if [ -f "$logfile" ]; then
            filename=$(basename "$logfile")
            # 提取最后一行包含进度条的内容
            progress=$(tail -n 20 "$logfile" | grep -oP '\d+%\|[^|]+\|\s*\d+/\d+' | tail -1)
            
            if [ -z "$progress" ]; then
                # 如果没有进度条，检查是否完成
                if grep -q "100%" "$logfile" 2>/dev/null; then
                    echo "[$filename] ✓ 完成"
                elif grep -q "Error\|error\|Exception" "$logfile" 2>/dev/null; then
                    echo "[$filename] ✗ 错误"
                else
                    echo "[$filename] ... 启动中"
                fi
            else
                echo "[$filename] $progress"
            fi
        fi
    done
    
    echo ""
    echo "按 Ctrl+C 退出监控"
    sleep 2
done
