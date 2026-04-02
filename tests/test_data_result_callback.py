import json

from app.core.dataset import DatasetBuilder


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
            def raise_for_status(self):
                return None

        return FakeResponse()

    monkeypatch.setenv("CALLBACK_URL", "http://callback.test/front/callback/dataResultPath")
    monkeypatch.setenv("TASK_ID", "abc123")
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
