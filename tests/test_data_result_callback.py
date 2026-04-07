import json

import pytest

from app.core.dataset import DatasetBuilder
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
