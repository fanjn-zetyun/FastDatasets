import asyncio
import json
import httpx
import pytest

from app.core.llm import AsyncLLM, LLMRequestError


def test_local_base_url_connect_error_falls_back_without_retry(monkeypatch):
    llm = AsyncLLM(
        model_name="test-model",
        base_url=" http://127.0.0.1:8000/v1 ",
        api_key="test-key",
        max_concurrency=1,
    )

    call_count = 0

    class DummyAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise httpx.ConnectError("All connection attempts failed")

    async def fail_if_sleep_called(*args, **kwargs):
        raise AssertionError("local connect errors should not sleep and retry")

    monkeypatch.delenv("FASTDATASETS_PARAMS", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_BASE", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.setattr(httpx, "AsyncClient", DummyAsyncClient)
    monkeypatch.setattr("app.core.llm.asyncio.sleep", fail_if_sleep_called)

    with pytest.raises(LLMRequestError, match="本地 LLM 服务不可达"):
        asyncio.run(llm.call_llm_advanced("生成不少于 2 个问题", retries=8))

    assert call_count == 1


def test_remote_base_url_connect_error_retries(monkeypatch):
    llm = AsyncLLM(
        model_name="test-model",
        base_url="https://api.example.com/v1",
        api_key="test-key",
        max_concurrency=1,
    )

    call_count = 0
    sleep_calls = []

    class DummyAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise httpx.ConnectError("All connection attempts failed")

    async def fake_sleep(delay):
        sleep_calls.append(delay)

    monkeypatch.delenv("FASTDATASETS_PARAMS", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_BASE", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.setattr(httpx, "AsyncClient", DummyAsyncClient)
    monkeypatch.setattr("app.core.llm.asyncio.sleep", fake_sleep)

    with pytest.raises(LLMRequestError, match="连接失败，已达到最大重试次数"):
        asyncio.run(llm.call_llm_advanced("普通提示", retries=3))

    assert call_count == 3
    assert sleep_calls == [60, 60]


def test_remote_base_url_uses_default_total_three_attempts(monkeypatch):
    llm = AsyncLLM(
        model_name="test-model",
        base_url="https://api.example.com/v1",
        api_key="test-key",
        max_concurrency=1,
    )

    call_count = 0
    sleep_calls = []

    class DummyAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise httpx.ConnectError("All connection attempts failed")

    async def fake_sleep(delay):
        sleep_calls.append(delay)

    monkeypatch.delenv("FASTDATASETS_PARAMS", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_BASE", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.setattr(httpx, "AsyncClient", DummyAsyncClient)
    monkeypatch.setattr("app.core.llm.asyncio.sleep", fake_sleep)

    with pytest.raises(LLMRequestError, match="连接失败，已达到最大重试次数"):
        asyncio.run(llm.call_llm_advanced("普通提示"))

    assert call_count == 3
    assert sleep_calls == [60, 60]


def test_fastdatasets_params_take_precedence_over_default_localhost(monkeypatch):
    llm = AsyncLLM(
        model_name="your-model-name",
        base_url="http://localhost:8000/v1",
        api_key="your-api-key",
        max_concurrency=1,
    )

    seen = {}

    class DummyResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "ok"}}]}

    class DummyAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, headers=None, json=None, follow_redirects=True):
            seen["url"] = url
            seen["auth"] = headers.get("Authorization")
            seen["model"] = json.get("model")
            return DummyResponse()

    monkeypatch.setenv(
        "FASTDATASETS_PARAMS",
        json.dumps(
            {
                "api_key": "fastdatasets-key",
                "base_url": "https://fd.example.com/v1",
                "model_name": "fd-model",
            }
        ),
    )
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_BASE", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.setattr(httpx, "AsyncClient", DummyAsyncClient)

    response = asyncio.run(llm.call_llm_advanced("普通提示", retries=1))

    assert seen == {
        "url": "https://fd.example.com/v1/chat/completions",
        "auth": "Bearer fastdatasets-key",
        "model": "fd-model",
    }
    assert response == {"choices": [{"message": {"content": "ok"}}]}


def test_moderation_400_does_not_retry(monkeypatch):
    llm = AsyncLLM(
        model_name="test-model",
        base_url="https://api.example.com/v1",
        api_key="test-key",
        max_concurrency=1,
    )

    call_count = 0

    class DummyResponse:
        status_code = 400
        text = '{"code":1601,"message":"内容包含违规信息，未通过审核"}'

        def json(self):
            return {"code": 1601, "message": "内容包含违规信息，未通过审核"}

    class DummyAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, headers=None, json=None, follow_redirects=True):
            nonlocal call_count
            call_count += 1
            request = httpx.Request("POST", url)
            response = DummyResponse()
            raise httpx.HTTPStatusError("bad request", request=request, response=response)

    async def fail_if_sleep_called(*args, **kwargs):
        raise AssertionError("moderation 400 errors should not sleep and retry")

    monkeypatch.delenv("FASTDATASETS_PARAMS", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_BASE", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.setattr(httpx, "AsyncClient", DummyAsyncClient)
    monkeypatch.setattr("app.core.llm.asyncio.sleep", fail_if_sleep_called)

    with pytest.raises(LLMRequestError, match="请求内容未通过审核 \\(HTTP 400\\)"):
        asyncio.run(llm.call_llm_advanced("普通提示", retries=8))

    assert call_count == 1
