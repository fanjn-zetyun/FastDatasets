#!/bin/bash
set -euo pipefail

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

if [ -z "$FASTDATASETS_PARAMS" ]; then
    log "ERROR: FASTDATASETS_PARAMS environment variable is not set"
    exit 1
fi

log "FASTDATASETS_PARAMS environment variable is set"
log "Building FastDatasets command..."
python3 /app/cmd_builder.py

log "Starting FastDatasets..."
set +e
bash /tmp/fastdatasets_cmd.sh 2>&1 | while IFS= read -r line; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $line" | tee -a "$LOG_FILE"
done
cmd_status=${PIPESTATUS[0]}
set -e

if [ "$cmd_status" -ne 0 ]; then
    log "ERROR: FastDatasets failed with exit code $cmd_status"
    exit "$cmd_status"
fi

if grep -qi "error" "$LOG_FILE"; then
    log "ERROR: Detected error logs during FastDatasets execution"
    exit 1
fi

log "FastDatasets completed successfully!"
log "=========================================="
