import argparse
import asyncio
import os
from pathlib import Path
from typing import List

from app.core.config import Config
from app.core.document import DocumentProcessor
from app.core.dataset import DatasetBuilder


def run_generate(input_paths: List[str], output_path: str, formats: List[str], file_format: str,
                 chunk_min_len: int = None, chunk_max_len: int = None,
                 questions_per_chunk: int = None, llm_concurrency: int = None,
                 file_concurrency: int = None, enable_cot: bool = False, name: str = None):
    cfg = Config()

    # 环境变量覆盖（便于 CLI 直接注入）
    api_key = os.getenv("LLM_API_KEY", cfg.API_KEY)
    base_url = os.getenv("LLM_API_BASE", cfg.BASE_URL)
    model = os.getenv("LLM_MODEL", cfg.MODEL_NAME)
    cfg.API_KEY, cfg.BASE_URL, cfg.MODEL_NAME = api_key, base_url, model

    # CLI 参数覆盖配置
    if chunk_min_len is not None:
        cfg.CHUNK_MIN_LEN = chunk_min_len
    if chunk_max_len is not None:
        cfg.CHUNK_MAX_LEN = chunk_max_len
    if questions_per_chunk is not None:
        cfg.DEFAULT_SAMPLE_SIZE = questions_per_chunk
    if llm_concurrency is not None:
        cfg.MAX_LLM_CONCURRENCY = llm_concurrency
    if enable_cot:
        cfg.ENABLE_COT = True

    processor = DocumentProcessor()
    builder = DatasetBuilder()

    # 收集所有块
    all_chunks = []
    for p in input_paths:
        path = Path(p)
        if path.is_dir():
            for fp in path.rglob("*.*"):
                chunks = processor.process_document(str(fp))
                all_chunks.extend(chunks)
        else:
            chunks = processor.process_document(str(path))
            all_chunks.extend(chunks)

    dataset = asyncio.run(builder.build_dataset(all_chunks))

    # 导出
    builder.export_dataset(dataset, output_path, formats=formats, file_format=file_format, name=name)


def main():
    parser = argparse.ArgumentParser(prog="fastdatasets", description="Generate LLM training datasets from documents.")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="Generate dataset from files or directories")
    gen.add_argument("inputs", nargs="+", help="Input file(s) or directory(ies)")
    gen.add_argument("-o", "--output", default="output", help="Output directory or file path (e.g., /app/ok.json)")
    gen.add_argument("-n", "--name", default=None, help="Dataset name (default: 'dataset')")
    gen.add_argument("-f", "--formats", default="alpaca", help="Export formats, comma-separated (alpaca,sharegpt)")
    gen.add_argument("--file-format", default="json", choices=["json", "jsonl"], help="Output file format")
    gen.add_argument("--chunk-min-len", type=int, default=None, help="Minimum chunk length")
    gen.add_argument("--chunk-max-len", type=int, default=None, help="Maximum chunk length")
    gen.add_argument("--questions-per-chunk", type=int, default=None, help="Number of questions per chunk")
    gen.add_argument("--llm-concurrency", type=int, default=5, help="LLM concurrency limit")
    gen.add_argument("--file-concurrency", type=int, default=3, help="File processing concurrency")
    gen.add_argument("--enable-cot", action="store_true", help="Enable chain-of-thought")

    args = parser.parse_args()

    if args.command == "generate":
        formats = [s.strip() for s in str(args.formats).split(",") if s.strip()]
        run_generate(args.inputs, args.output, formats=formats, file_format=args.file_format,
                     chunk_min_len=args.chunk_min_len, chunk_max_len=args.chunk_max_len,
                     questions_per_chunk=args.questions_per_chunk, llm_concurrency=args.llm_concurrency,
                     file_concurrency=args.file_concurrency, enable_cot=args.enable_cot, name=args.name)


if __name__ == "__main__":
    main()




