#!/bin/bash
set -e

# 创建日志目录
LOG_DIR="/workspace/tmp/fastdatasets_container"
mkdir -p "$LOG_DIR"

# 生成带时间戳的日志文件
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_FILE="$LOG_DIR/fastdatasets_$TIMESTAMP.log"

# 日志函数：输出到控制台和日志文件，带时间戳
log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "$msg" | tee -a "$LOG_FILE"
}

log "=========================================="
log "FastDatasets Job Started"
log "Log file: $LOG_FILE"
log "=========================================="

# 将日志文件路径导出为环境变量，供 cmd_builder.py 使用
export FASTDATASETS_LOG_FILE="$LOG_FILE"

log "Building FastDatasets command..."
python3 /app/cmd_builder.py

log "Starting FastDatasets..."
bash /tmp/fastdatasets_cmd.sh 2>&1 | while IFS= read -r line; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $line" | tee -a "$LOG_FILE"
done

log "FastDatasets completed successfully!"
log "=========================================="
