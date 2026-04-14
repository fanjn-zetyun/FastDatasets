import json
import os
import subprocess

import cmd_builder


def test_shell_join_quotes_path_with_spaces():
    cmd = [
        "fastdatasets",
        "generate",
        "/workspace/user-data/upload/10-54-59/APS 6.2 ARM环境（华为云）测试报告.docx",
        "-o",
        "/workspace/user-data/datasets/docx/docx",
        "-f",
        "alpaca",
    ]

    rendered = cmd_builder.shell_join(cmd)

    assert "'/workspace/user-data/upload/10-54-59/APS 6.2 ARM环境（华为云）测试报告.docx'" in rendered
    assert rendered.startswith("fastdatasets generate ")


def test_build_command_keeps_spaced_input_as_single_argument():
    params = {
        "file_path_input": "/workspace/user-data/upload/10-54-59/APS 6.2 ARM环境（华为云）测试报告.docx",
        "export_path": "/workspace/user-data/datasets/docx/docx",
        "output_formats": ["alpaca"],
        "chunk_min_len": 200,
        "chunk_max_len": 1000,
        "questions_per_chunk": 2,
        "llm_concurrency": 3,
        "file_concurrency": 2,
        "name": "docx",
    }

    cmd = cmd_builder.build_command(params)

    assert cmd[0:2] == ["fastdatasets", "generate"]
    assert cmd[2] == "/workspace/user-data/upload/10-54-59/APS 6.2 ARM环境（华为云）测试报告.docx"
    assert cmd[3:5] == ["-o", "/workspace/user-data/datasets/docx/docx"]


def test_generated_shell_script_preserves_spaced_filename(monkeypatch, tmp_path):
    input_path = "/workspace/user-data/upload/10-54-59/APS 6.2 ARM环境（华为云）测试报告.docx"
    args_file = tmp_path / "args.json"
    fake_fastdatasets = tmp_path / "fastdatasets"

    fake_fastdatasets.write_text(
        "#!/bin/bash\n"
        "python3 -c 'import json, sys; json.dump(sys.argv[1:], open(sys.argv[1], \"w\", encoding=\"utf-8\"), ensure_ascii=False)' \"$1\" \"${@:2}\"\n",
        encoding="utf-8",
    )
    fake_fastdatasets.chmod(0o755)

    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ.get('PATH', '')}")
    monkeypatch.setenv(
        "FASTDATASETS_PARAMS",
        json.dumps(
            {
                "file_path_input": input_path,
                "export_path": "/workspace/user-data/datasets/docx/docx",
                "output_formats": ["alpaca"],
                "chunk_min_len": 200,
                "chunk_max_len": 1000,
                "questions_per_chunk": 2,
                "llm_concurrency": 3,
                "file_concurrency": 2,
                "name": "docx",
                "api_key": "test-key",
                "base_url": "https://example.test/v1",
                "model_name": "test-model",
            },
            ensure_ascii=False,
        ),
    )

    original_build_command = cmd_builder.build_command

    def build_command_with_capture(params):
        return [
            "fastdatasets",
            str(args_file),
            *original_build_command(params)[1:],
        ]

    monkeypatch.setattr(cmd_builder, "build_command", build_command_with_capture)

    cmd_builder.main()
    subprocess.run(["bash", "/tmp/fastdatasets_cmd.sh"], check=True)

    recorded_args = json.loads(args_file.read_text(encoding="utf-8"))

    assert recorded_args[0] == str(args_file)
    assert recorded_args[1] == "generate"
    assert recorded_args[2] == input_path
    assert recorded_args[3:5] == ["-o", "/workspace/user-data/datasets/docx/docx"]
