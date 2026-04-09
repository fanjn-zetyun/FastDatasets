import asyncio
import json

import pytest

from app.core.dataset import DatasetBuilder, DatasetBuildFailure
from app.core.llm import LLMRequestError
from fastdatasets.cli import run_generate


def test_export_dataset_callbacks_after_success(monkeypatch, tmp_path):
    builder = DatasetBuilder()
    dataset = [
        {
            "question": "问题1",
            "answer": "答案1",
        }
    ]
    callback_calls = []

    def fake_post(url, json=None, timeout=None):
        callback_calls.append(
            {
                "url": url,
                "json": json,
                "timeout": timeout,
            }
        )

        class FakeResponse:
            status_code = 200
            text = "ok"

            def raise_for_status(self):
                return None

        return FakeResponse()

    monkeypatch.setenv("CALLBACK_URL", "http://callback.test/front/callback/dataResultPath")
    monkeypatch.setenv("TASK_ID", "abc123")
    log_file = tmp_path / "fastdatasets.log"
    monkeypatch.setenv("FASTDATASETS_LOG_FILE", str(log_file))
    monkeypatch.setattr("app.core.dataset.httpx.post", fake_post)

    export_base = tmp_path / "job001"
    builder.export_dataset(
        dataset,
        str(export_base),
        formats=["alpaca", "sharegpt"],
        file_format="jsonl",
    )

    expected_paths = [
        str(tmp_path / "job001-alpaca.jsonl"),
        str(tmp_path / "job001-sharegpt.jsonl"),
    ]

    assert callback_calls == [
        {
            "url": "http://callback.test/front/callback/dataResultPath",
            "json": [
                {"id": "abc123", "resultPath": expected_paths[0]},
                {"id": "abc123", "resultPath": expected_paths[1]},
            ],
            "timeout": 30.0,
        }
    ]

    for path in expected_paths:
        with open(path, "r", encoding="utf-8") as file_obj:
            lines = [json.loads(line) for line in file_obj if line.strip()]
        assert lines

    log_text = log_file.read_text(encoding="utf-8")
    assert "开始数据结果路径回调" in log_text
    assert "数据结果路径回调成功" in log_text
    assert "response=ok" in log_text


def test_stream_exports_finalize_to_jsonl_and_callback(monkeypatch, tmp_path):
    builder = DatasetBuilder()
    callback_calls = []

    def fake_post(url, json=None, timeout=None):
        callback_calls.append(
            {
                "url": url,
                "json": json,
                "timeout": timeout,
            }
        )

        class FakeResponse:
            status_code = 200
            text = "ok"

            def raise_for_status(self):
                return None

        return FakeResponse()

    monkeypatch.setenv("CALLBACK_URL", "http://callback.test/front/callback/dataResultPath")
    monkeypatch.setenv("TASK_ID", "partial-001")
    monkeypatch.setenv("FASTDATASETS_LOG_FILE", str(tmp_path / "fastdatasets.log"))
    monkeypatch.setattr("app.core.dataset.httpx.post", fake_post)

    stream_targets = builder.prepare_stream_exports(
        str(tmp_path / "job-partial"),
        formats=["alpaca", "sharegpt"],
        file_format="jsonl",
    )
    builder.append_stream_exports(
        [
            {
                "question": "问题1",
                "answer": "答案1",
            }
        ],
        stream_targets,
    )

    finalized_paths = builder.finalize_stream_exports(stream_targets, file_format="jsonl")
    builder.notify_result_paths(finalized_paths)

    expected_paths = [
        str(tmp_path / "job-partial-alpaca.jsonl"),
        str(tmp_path / "job-partial-sharegpt.jsonl"),
    ]

    assert finalized_paths == expected_paths
    assert callback_calls == [
        {
            "url": "http://callback.test/front/callback/dataResultPath",
            "json": [
                {"id": "partial-001", "resultPath": expected_paths[0]},
                {"id": "partial-001", "resultPath": expected_paths[1]},
            ],
            "timeout": 30.0,
        }
    ]

    for path in expected_paths:
        with open(path, "r", encoding="utf-8") as file_obj:
            lines = [json.loads(line) for line in file_obj if line.strip()]
        assert lines


def test_cli_interrupt_callbacks_partial_results(monkeypatch, tmp_path):
    callback_calls = []

    def fake_post(url, json=None, timeout=None):
        callback_calls.append(
            {
                "url": url,
                "json": json,
                "timeout": timeout,
            }
        )

        class FakeResponse:
            status_code = 200
            text = "ok"

            def raise_for_status(self):
                return None

        return FakeResponse()

    async def fake_build_dataset(self, chunks, on_batch_complete=None, keep_in_memory=True):
        if on_batch_complete:
            on_batch_complete(
                [
                    {
                        "question": "中断前问题",
                        "answer": "中断前答案",
                    }
                ]
            )
        raise KeyboardInterrupt("stop")

    monkeypatch.setenv("CALLBACK_URL", "http://callback.test/front/callback/dataResultPath")
    monkeypatch.setenv("TASK_ID", "interrupt-001")
    monkeypatch.setenv("FASTDATASETS_LOG_FILE", str(tmp_path / "fastdatasets.log"))
    monkeypatch.setattr("app.core.dataset.httpx.post", fake_post)
    monkeypatch.setattr("fastdatasets.cli.DocumentProcessor.process_document", lambda self, path: [{"content": "x"}])
    monkeypatch.setattr("app.core.dataset.DatasetBuilder.build_dataset", fake_build_dataset)

    input_file = tmp_path / "doc.txt"
    input_file.write_text("hello", encoding="utf-8")

    with pytest.raises(KeyboardInterrupt):
        run_generate(
            [str(input_file)],
            str(tmp_path / "job-interrupt"),
            formats=["alpaca"],
            file_format="jsonl",
            name="job-interrupt",
        )

    expected_path = str(tmp_path / "job-interrupt-alpaca.jsonl")
    assert callback_calls == [
        {
            "url": "http://callback.test/front/callback/dataResultPath",
            "json": [
                {"id": "interrupt-001", "resultPath": expected_path},
            ],
            "timeout": 30.0,
        }
    ]

    with open(expected_path, "r", encoding="utf-8") as file_obj:
        lines = [json.loads(line) for line in file_obj if line.strip()]
    assert lines == [
        {
            "instruction": "中断前问题",
            "input": "",
            "output": "中断前答案",
        }
    ]


def test_build_dataset_skips_partial_chunk_failures_and_records_details(monkeypatch):
    builder = DatasetBuilder()
    builder.enable_optimize = False

    async def fake_generate_questions(self, context, number=5):
        if context == "bad chunk":
            raise RuntimeError("question generation failed")
        return [f"{context}-q1"]

    async def fake_generate_answer(self, question, context):
        return f"{question}-answer"

    monkeypatch.setattr("app.core.dataset.DatasetBuilder._generate_questions", fake_generate_questions)
    monkeypatch.setattr("app.core.dataset.DatasetBuilder._generate_answer", fake_generate_answer)

    chunks = [
        {"file": "doc-a.txt", "chunk_id": "doc-a_part_1", "content": "good chunk", "summary": "good"},
        {"file": "doc-a.txt", "chunk_id": "doc-a_part_2", "content": "bad chunk", "summary": "bad"},
    ]

    dataset = asyncio.run(builder.build_dataset(chunks))

    assert len(dataset) == 1
    assert dataset[0]["chunk_id"] == "doc-a_part_1"
    assert builder.last_document_failures == []
    assert builder.last_failed_parts == [
        {
            "file": "doc-a.txt",
            "chunk_id": "doc-a_part_2",
            "summary": "bad",
            "content_preview": "bad chunk",
            "stage": "question_generation",
            "error_type": "RuntimeError",
            "error_message": "question generation failed",
        }
    ]


def test_build_dataset_raises_when_entire_document_fails(monkeypatch):
    builder = DatasetBuilder()
    builder.enable_optimize = False

    async def fake_generate_questions(self, context, number=5):
        if context.startswith("bad"):
            raise RuntimeError(f"{context} failed")
        return [f"{context}-q1"]

    async def fake_generate_answer(self, question, context):
        return f"{question}-answer"

    monkeypatch.setattr("app.core.dataset.DatasetBuilder._generate_questions", fake_generate_questions)
    monkeypatch.setattr("app.core.dataset.DatasetBuilder._generate_answer", fake_generate_answer)

    chunks = [
        {"file": "good.txt", "chunk_id": "good_part_1", "content": "good", "summary": "good"},
        {"file": "bad.txt", "chunk_id": "bad_part_1", "content": "bad-1", "summary": "bad1"},
        {"file": "bad.txt", "chunk_id": "bad_part_2", "content": "bad-2", "summary": "bad2"},
    ]

    with pytest.raises(DatasetBuildFailure) as exc_info:
        asyncio.run(builder.build_dataset(chunks))

    exc = exc_info.value
    assert exc.partial_dataset == [
        {
            "chunk_id": "good_part_1",
            "file": "good.txt",
            "summary": "good",
            "content": "good",
            "question": "good-q1",
            "answer": "good-q1-answer",
        }
    ]
    assert exc.document_failures == [
        {
            "file": "bad.txt",
            "failed_chunk_ids": ["bad_part_1", "bad_part_2"],
        }
    ]
    assert "bad.txt" in str(exc)
    assert "bad_part_1" in str(exc)


def test_build_dataset_raises_when_no_questions_generated_for_any_chunk(monkeypatch):
    builder = DatasetBuilder()
    builder.enable_optimize = False

    async def fake_generate_questions(self, context, number=5):
        raise LLMRequestError(f"{context} llm exhausted retries")

    monkeypatch.setattr("app.core.dataset.DatasetBuilder._generate_questions", fake_generate_questions)

    chunks = [
        {"file": "all-bad.txt", "chunk_id": "all_bad_part_1", "content": "bad-1", "summary": "bad1"},
        {"file": "all-bad.txt", "chunk_id": "all_bad_part_2", "content": "bad-2", "summary": "bad2"},
    ]

    with pytest.raises(DatasetBuildFailure) as exc_info:
        asyncio.run(builder.build_dataset(chunks))

    exc = exc_info.value
    assert exc.document_failures == [
        {
            "file": "all-bad.txt",
            "failed_chunk_ids": ["all_bad_part_1", "all_bad_part_2"],
        }
    ]
    assert len(exc.failed_parts) == 2
    assert exc.failed_parts[0]["content_preview"]


def test_build_dataset_skips_chunk_when_llm_request_error_exhausted(monkeypatch):
    builder = DatasetBuilder()
    builder.enable_optimize = False

    async def fake_generate_questions(self, context, number=5):
        if context == "fatal":
            raise LLMRequestError("llm exhausted retries")
        return [f"{context}-q1"]

    async def fake_generate_answer(self, question, context):
        return f"{question}-answer"

    monkeypatch.setattr("app.core.dataset.DatasetBuilder._generate_questions", fake_generate_questions)
    monkeypatch.setattr("app.core.dataset.DatasetBuilder._generate_answer", fake_generate_answer)

    chunks = [
        {"file": "mix.txt", "chunk_id": "mix_part_1", "content": "ok", "summary": "ok"},
        {"file": "mix.txt", "chunk_id": "mix_part_2", "content": "fatal", "summary": "fatal"},
    ]

    dataset = asyncio.run(builder.build_dataset(chunks))
    assert len(dataset) == 1
    assert dataset[0]["chunk_id"] == "mix_part_1"
    assert builder.last_document_failures == []
    assert builder.last_failed_parts == [
        {
            "file": "mix.txt",
            "chunk_id": "mix_part_2",
            "summary": "fatal",
            "content_preview": "fatal",
            "stage": "question_generation",
            "error_type": "LLMRequestError",
            "error_message": "llm exhausted retries",
        }
    ]


def test_build_dataset_skips_chunk_when_request_source_online_web_is_unsupported(monkeypatch):
    builder = DatasetBuilder()
    builder.enable_optimize = False

    async def fake_generate_questions(self, context, number=5):
        if context == "unsupported":
            raise LLMRequestError(
                "请求参数错误，当前请求来源不被目标服务支持 (code=1500): 未知的请求来源: 'ONLINE_WEB'"
            )
        return [f"{context}-q1"]

    async def fake_generate_answer(self, question, context):
        return f"{question}-answer"

    monkeypatch.setattr("app.core.dataset.DatasetBuilder._generate_questions", fake_generate_questions)
    monkeypatch.setattr("app.core.dataset.DatasetBuilder._generate_answer", fake_generate_answer)

    chunks = [
        {"file": "mix.txt", "chunk_id": "mix_part_1", "content": "ok", "summary": "ok"},
        {"file": "mix.txt", "chunk_id": "mix_part_2", "content": "unsupported", "summary": "unsupported"},
    ]

    dataset = asyncio.run(builder.build_dataset(chunks))

    assert len(dataset) == 1
    assert dataset[0]["chunk_id"] == "mix_part_1"
    assert builder.last_document_failures == []
    assert builder.last_failed_parts == [
        {
            "file": "mix.txt",
            "chunk_id": "mix_part_2",
            "summary": "unsupported",
            "content_preview": "unsupported",
            "stage": "question_generation",
            "error_type": "LLMRequestError",
            "error_message": "请求参数错误，当前请求来源不被目标服务支持 (code=1500): 未知的请求来源: 'ONLINE_WEB'",
        }
    ]


def test_build_dataset_fails_task_when_all_chunks_hit_unsupported_online_web_request_source(monkeypatch):
    builder = DatasetBuilder()
    builder.enable_optimize = False

    async def fake_generate_questions(self, context, number=5):
        raise LLMRequestError(
            "请求参数错误，当前请求来源不被目标服务支持 (code=1500): 未知的请求来源: 'ONLINE_WEB'"
        )

    monkeypatch.setattr("app.core.dataset.DatasetBuilder._generate_questions", fake_generate_questions)

    chunks = [
        {"file": "all-bad.txt", "chunk_id": "all_bad_part_1", "content": "bad-1", "summary": "bad1"},
        {"file": "all-bad.txt", "chunk_id": "all_bad_part_2", "content": "bad-2", "summary": "bad2"},
    ]

    with pytest.raises(DatasetBuildFailure) as exc_info:
        asyncio.run(builder.build_dataset(chunks))

    exc = exc_info.value
    assert exc.document_failures == [
        {
            "file": "all-bad.txt",
            "failed_chunk_ids": ["all_bad_part_1", "all_bad_part_2"],
        }
    ]
    assert len(exc.failed_parts) == 2
    assert all("当前请求来源不被目标服务支持" in item["error_message"] for item in exc.failed_parts)


def test_failure_message_includes_content_preview(monkeypatch):
    builder = DatasetBuilder()
    builder.enable_optimize = False

    bad_content = "开头敏感内容" + ("中间内容" * 40) + "结尾敏感内容"

    async def fake_generate_questions(self, context, number=5):
        raise RuntimeError("question generation failed")

    monkeypatch.setattr("app.core.dataset.DatasetBuilder._generate_questions", fake_generate_questions)

    chunks = [
        {"file": "preview.txt", "chunk_id": "preview_part_1", "content": bad_content, "summary": "preview"},
    ]

    with pytest.raises(DatasetBuildFailure) as exc_info:
        asyncio.run(builder.build_dataset(chunks))

    message = str(exc_info.value)
    assert "content_preview=" in message
    assert "开头敏感内容" in message
    assert "结尾敏感内容" in message
    assert " ... " in message


def test_cli_dataset_failure_callbacks_partial_results(monkeypatch, tmp_path):
    callback_calls = []

    def fake_post(url, json=None, timeout=None):
        callback_calls.append(
            {
                "url": url,
                "json": json,
                "timeout": timeout,
            }
        )

        class FakeResponse:
            status_code = 200
            text = "ok"

            def raise_for_status(self):
                return None

        return FakeResponse()

    async def fake_build_dataset(self, chunks, on_batch_complete=None, keep_in_memory=True):
        if on_batch_complete:
            on_batch_complete(
                [
                    {
                        "question": "成功问题",
                        "answer": "成功答案",
                    }
                ]
            )
        raise DatasetBuildFailure(
            "bad.txt 全部失败",
            document_failures=[{"file": "bad.txt", "failed_chunk_ids": ["bad_part_1"]}],
            failed_parts=[
                {
                    "file": "bad.txt",
                    "chunk_id": "bad_part_1",
                    "summary": "bad",
                    "stage": "question_generation",
                    "error_type": "RuntimeError",
                    "error_message": "bad chunk failed",
                }
            ],
            partial_dataset=[],
        )

    monkeypatch.setenv("CALLBACK_URL", "http://callback.test/front/callback/dataResultPath")
    monkeypatch.setenv("TASK_ID", "doc-failed-001")
    monkeypatch.setenv("FASTDATASETS_LOG_FILE", str(tmp_path / "fastdatasets.log"))
    monkeypatch.setattr("app.core.dataset.httpx.post", fake_post)
    monkeypatch.setattr("fastdatasets.cli.DocumentProcessor.process_document", lambda self, path: [{"content": "x"}])
    monkeypatch.setattr("app.core.dataset.DatasetBuilder.build_dataset", fake_build_dataset)

    input_file = tmp_path / "doc.txt"
    input_file.write_text("hello", encoding="utf-8")

    with pytest.raises(DatasetBuildFailure):
        run_generate(
            [str(input_file)],
            str(tmp_path / "job-doc-failed"),
            formats=["alpaca"],
            file_format="jsonl",
            name="job-doc-failed",
        )

    expected_path = str(tmp_path / "job-doc-failed-alpaca.jsonl")
    assert callback_calls == [
        {
            "url": "http://callback.test/front/callback/dataResultPath",
            "json": [
                {"id": "doc-failed-001", "resultPath": expected_path},
            ],
            "timeout": 30.0,
        }
    ]

    with open(expected_path, "r", encoding="utf-8") as file_obj:
        lines = [json.loads(line) for line in file_obj if line.strip()]
    assert lines == [
        {
            "instruction": "成功问题",
            "input": "",
            "output": "成功答案",
        }
    ]
