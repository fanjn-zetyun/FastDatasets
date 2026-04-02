#!/usr/bin/env python3
"""
从标准输入或文件读取JSON参数，构建FastDatasets CLI命令
与GraphGen的yaml_builder.py保持一致的架构
"""
import json
import os
import sys
from datetime import datetime
from typing import Any, List

# 固定的输出目录
OUTPUT_DIR = "/workspace/user-data/datasets"

def log(message):
    """输出日志到控制台和日志文件，带时间戳"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] {message}"
    print(log_line)
    
    # 如果设置了日志文件，也写入文件
    log_file = os.environ.get("FASTDATASETS_LOG_FILE")
    if log_file:
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(log_line + "\n")
        except Exception:
            pass  # 忽略日志文件写入错误

def load_params():
    """
    从多种来源读取参数：
    1. 环境变量 FASTDATASETS_PARAMS（JSON字符串）
    2. 文件 /config/params.json
    3. 标准输入
    """
    # 方式1：环境变量
    if "FASTDATASETS_PARAMS" in os.environ:
        return json.loads(os.environ["FASTDATASETS_PARAMS"])

    # 方式2：配置文件
    if os.path.exists("/config/params.json"):
        with open("/config/params.json", "r") as f:
            return json.load(f)

    # 方式3：标准输入
    if not sys.stdin.isatty():
        return json.load(sys.stdin)

    raise Exception("No parameters provided. Set FASTDATASETS_PARAMS env var, mount /config/params.json, or pipe JSON to stdin.")

def normalize_input_files(params: dict) -> List[str]:
    """将多种输入参数统一归一化为文件路径列表。"""
    candidates: List[Any] = []

    upload_method = params.get("upload_method")
    if upload_method and params.get(upload_method):
        candidates.append(params.get(upload_method))

    for key in ("file_path_input", "input_files", "local_file_upload"):
        if params.get(key) is not None:
            candidates.append(params.get(key))

    normalized: List[str] = []
    for candidate in candidates:
        if isinstance(candidate, str):
            value = candidate.strip()
            if value:
                normalized.append(value)
        elif isinstance(candidate, (list, tuple)):
            for item in candidate:
                if isinstance(item, str):
                    value = item.strip()
                    if value:
                        normalized.append(value)

    # 去重并保持顺序
    unique_inputs: List[str] = []
    seen = set()
    for path in normalized:
        if path not in seen:
            unique_inputs.append(path)
            seen.add(path)

    existing_inputs = [path for path in unique_inputs if os.path.exists(path)]
    if existing_inputs:
        return existing_inputs

    return unique_inputs

def build_command(params):
    """构建FastDatasets CLI命令"""

    # 设置环境变量（LLM配置）
    os.environ["LLM_API_KEY"] = params.get("api_key", "")
    os.environ["LLM_API_BASE"] = params.get("base_url", "")
    os.environ["LLM_MODEL"] = params.get("model_name", "")

    # 提取输入文件 - 兼容字符串/数组以及 upload_method 指定的字段
    input_files = normalize_input_files(params)
    output_path = params.get("export_path", '')
    output_formats = params.get("output_formats", ["alpaca", "sharegpt"])
    chunk_min_len = params.get("chunk_min_len", 200)
    chunk_max_len = params.get("chunk_max_len", 1000)
    questions_per_chunk = params.get("questions_per_chunk", 2)
    enable_cot = params.get("enable_cot", False)
    llm_concurrency = params.get("llm_concurrency", 3)
    file_concurrency = params.get("file_concurrency", 2)
    name = params.get("name", None)
    if not name:
        name = params.get("exportFileName", None)

    if not input_files:
        raise ValueError("No valid input files provided in FASTDATASETS_PARAMS")

    # 构建命令参数列表
    # 注意：FastDatasets工具会自动处理同名数据集文件，会在文件名后添加_1, _2等后缀
    # 例如：export_path=/workspace/user-data/datasets/fd-2026-03-31
    # 生成的文件：/workspace/user-data/datasets/fd-2026-03-31-alpaca.json
    # 如果文件已存在，会自动重命名为：/workspace/user-data/datasets/fd-2026-03-31-alpaca_1.json
    cmd_parts = [
        "fastdatasets", "generate",
        *input_files,
        "-o", output_path,
        "-f", ",".join(output_formats),
        "--chunk-min-len", str(chunk_min_len),
        "--chunk-max-len", str(chunk_max_len),
        "--questions-per-chunk", str(questions_per_chunk),
        "--llm-concurrency", str(llm_concurrency),
        "--file-concurrency", str(file_concurrency),
    ]

    if enable_cot:
        cmd_parts.append("--enable-cot")
    
    if name:
        cmd_parts.extend(["-n", name])

    return cmd_parts

def main():
    # 1. 读取参数
    params = load_params()
    log("Loaded parameters:")
    log(json.dumps(params, indent=2, ensure_ascii=False))

    # 2. 构建命令
    cmd_parts = build_command(params)

    # 3. 写入命令脚本（包含环境变量）
    cmd_script = "/tmp/fastdatasets_cmd.sh"
    with open(cmd_script, "w") as f:
        # 写入环境变量
        f.write(f'export LLM_API_KEY="{params.get("api_key", "")}"\n')
        f.write(f'export LLM_API_BASE="{params.get("base_url", "")}"\n')
        f.write(f'export LLM_MODEL="{params.get("model_name", "")}"\n')
        # 写入命令
        f.write(" ".join(cmd_parts))

    log(f"Generated command script: {cmd_script}")
    log("Command:")
    log(" ".join(cmd_parts))

if __name__ == "__main__":
    main()
