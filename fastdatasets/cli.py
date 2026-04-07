import argparse
import asyncio
import os
import signal
from pathlib import Path
from typing import List

from app.core.config import config
from app.core.document import DocumentProcessor
from app.core.dataset import DatasetBuilder
from app.core.logger import logger


class GenerationInterrupted(KeyboardInterrupt):
    """用于将 SIGTERM/SIGINT 转换为可收尾的中断异常。"""


def _raise_interrupt(signum, _frame):
    signal_name = signal.Signals(signum).name
    raise GenerationInterrupted(f"Received {signal_name}")


def run_generate(input_paths: List[str], output_path: str, formats: List[str], file_format: str,
                 chunk_min_len: int = None, chunk_max_len: int = None,
                 questions_per_chunk: int = None, llm_concurrency: int = None,
                 file_concurrency: int = None, enable_cot: bool = False, name: str = None):
    # 环境变量覆盖（便于 CLI 直接注入）
    api_key = os.getenv("LLM_API_KEY", config.API_KEY)
    base_url = os.getenv("LLM_API_BASE", config.BASE_URL)
    model = os.getenv("LLM_MODEL", config.MODEL_NAME)
    config.API_KEY, config.BASE_URL, config.MODEL_NAME = api_key, base_url, model

    # CLI 参数覆盖配置
    if chunk_min_len is not None:
        config.CHUNK_MIN_LEN = chunk_min_len
    if chunk_max_len is not None:
        config.CHUNK_MAX_LEN = chunk_max_len
    if questions_per_chunk is not None:
        config.DEFAULT_SAMPLE_SIZE = questions_per_chunk
    if llm_concurrency is not None:
        config.MAX_LLM_CONCURRENCY = llm_concurrency
    if enable_cot:
        config.ENABLE_COT = True

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

    stream_targets = builder.prepare_stream_exports(output_path, formats=formats, file_format=file_format, name=name)

    previous_sigint = signal.getsignal(signal.SIGINT)
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, _raise_interrupt)
    signal.signal(signal.SIGTERM, _raise_interrupt)

    try:
        asyncio.run(
            builder.build_dataset(
                all_chunks,
                on_batch_complete=lambda batch: builder.append_stream_exports(batch, stream_targets),
                keep_in_memory=False,
            )
        )
        finalized_paths = builder.finalize_stream_exports(stream_targets, file_format=file_format)
        callback_ok = builder.notify_result_paths(finalized_paths)
        if callback_ok and finalized_paths:
            logger.info(f"数据集生成成功，结果文件路径: {', '.join(finalized_paths)}")
    except KeyboardInterrupt:
        logger.warning("数据集生成任务被中断，开始整理已生成的部分结果")
        finalized_paths = builder.finalize_stream_exports(stream_targets, file_format=file_format)
        if finalized_paths:
            callback_ok = builder.notify_result_paths(finalized_paths)
            if callback_ok:
                logger.info(f"数据集部分结果已生成并回调成功，结果文件路径: {', '.join(finalized_paths)}")
        raise
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)


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

