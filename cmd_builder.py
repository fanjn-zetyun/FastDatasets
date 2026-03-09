#!/usr/bin/env python3
"""
从标准输入或文件读取JSON参数，构建FastDatasets CLI命令
与GraphGen的yaml_builder.py保持一致的架构
"""
import json
import os
import sys

# 固定的输出目录
OUTPUT_DIR = "/workspace/user-data/dataset"

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

def build_command(params):
    """构建FastDatasets CLI命令"""

    # 设置环境变量（LLM配置）
    os.environ["OPENAI_API_KEY"] = params.get("api_key", "")
    os.environ["OPENAI_BASE_URL"] = params.get("base_url", "")
    os.environ["OPENAI_MODEL"] = params.get("model_name", "")

    # 提取参数
    input_files = params.get("input_files", [])
    output_dir = params.get("output_dir", OUTPUT_DIR)
    output_formats = params.get("output_formats", ["alpaca", "sharegpt"])
    chunk_min_len = params.get("chunk_min_len", 200)
    chunk_max_len = params.get("chunk_max_len", 1000)
    questions_per_chunk = params.get("questions_per_chunk", 2)
    enable_cot = params.get("enable_cot", True)
    llm_concurrency = params.get("llm_concurrency", 3)
    file_concurrency = params.get("file_concurrency", 2)

    # 构建命令参数列表
    cmd_parts = [
        "fastdatasets", "generate",
        *input_files,
        "-o", output_dir,
        "-f", ",".join(output_formats),
        "--chunk-min-len", str(chunk_min_len),
        "--chunk-max-len", str(chunk_max_len),
        "--questions-per-chunk", str(questions_per_chunk),
        "--llm-concurrency", str(llm_concurrency),
        "--file-concurrency", str(file_concurrency),
    ]

    if enable_cot:
        cmd_parts.append("--enable-cot")

    return cmd_parts

def main():
    # 1. 读取参数
    params = load_params()
    print("Loaded parameters:")
    print(json.dumps(params, indent=2, ensure_ascii=False))

    # 2. 构建命令
    cmd_parts = build_command(params)

    # 3. 写入命令脚本
    cmd_script = "/tmp/fastdatasets_cmd.sh"
    with open(cmd_script, "w") as f:
        f.write(" ".join(cmd_parts))

    print(f"\nGenerated command script: {cmd_script}")
    print("Command:")
    print(" ".join(cmd_parts))

if __name__ == "__main__":
    main()
