# 预留LLM相关接口，便于后续对接不同大模型

import asyncio
import httpx
import random
import time
import logging
import traceback
import os
import json
import re
from urllib.parse import urlparse
from app.core.config import config
from app.core.logger import logger

class LLMRequestError(RuntimeError):
    """Raised when an LLM request fails and the task should stop immediately."""

class AsyncLLM:
    _PLACEHOLDER_VALUES = {
        "your-api-key",
        "your-model-name",
    }
    _ONLINE_WEB_BASE_URLS = {
        "https://cloud.baicaiinfer.com/v1",
        "https://cloud.test.baicaiinfer.com/v1",
    }
    DEFAULT_RETRIES = 3
    DEFAULT_RETRY_INTERVAL_SECONDS = 60

    def __init__(self, model_name=None, base_url=None, api_key=None, language=None, max_concurrency=None, system_prompt=None):
        self.model_name = model_name or config.MODEL_NAME
        self.base_url = base_url or config.BASE_URL
        self.api_key = api_key or config.API_KEY
        self.language = language or config.LANGUAGE
        self.max_concurrency = max_concurrency or config.MAX_LLM_CONCURRENCY
        self.system_prompt = system_prompt or getattr(config, 'SYSTEM_PROMPT', None)
        self.semaphore = asyncio.Semaphore(self.max_concurrency)
        self.headers = {"Authorization": f"Bearer {self.api_key}"}

    def _clean_setting(self, value):
        if value is None:
            return None
        value = str(value).strip()
        if not value or value in self._PLACEHOLDER_VALUES:
            return None
        return value

    def _is_default_base_url(self, value) -> bool:
        if value is None:
            return False
        return str(value).strip() == str(config.BASE_URL).strip()

    def _extract_error_code(self, response) -> int | None:
        try:
            payload = response.json()
        except Exception:
            return None
        code = payload.get("code") if isinstance(payload, dict) else None
        try:
            return int(code) if code is not None else None
        except (TypeError, ValueError):
            return None

    def _load_fastdatasets_params(self):
        raw = os.getenv("FASTDATASETS_PARAMS")
        if not raw:
            return {}
        try:
            params = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("FASTDATASETS_PARAMS 不是合法 JSON，忽略其中的 LLM 配置")
            return {}
        return params if isinstance(params, dict) else {}

    def _resolve_runtime_llm_settings(self):
        params = self._load_fastdatasets_params()
        fastdatasets_api_key = self._clean_setting(params.get("api_key"))
        fastdatasets_base_url = self._clean_setting(params.get("base_url"))
        fastdatasets_model_name = self._clean_setting(params.get("model_name"))

        env_api_key = self._clean_setting(os.getenv("LLM_API_KEY"))
        env_base_url = self._clean_setting(os.getenv("LLM_API_BASE"))
        env_model_name = self._clean_setting(os.getenv("LLM_MODEL"))

        instance_api_key = self._clean_setting(self.api_key)
        instance_model_name = self._clean_setting(self.model_name)
        instance_base_url = self._clean_setting(self.base_url)
        if not fastdatasets_base_url and not env_base_url and self._is_default_base_url(self.base_url):
            instance_base_url = None

        api_key = fastdatasets_api_key or env_api_key or instance_api_key
        base_url = fastdatasets_base_url or env_base_url or instance_base_url
        model_name = fastdatasets_model_name or env_model_name or instance_model_name
        return api_key, base_url, model_name

    def _is_local_base_url(self, base_url=None) -> bool:
        base_url = self._clean_setting(base_url if base_url is not None else self.base_url)
        if not base_url:
            return False
        parsed = urlparse(base_url)
        hostname = (parsed.hostname or "").lower()
        return hostname in {"localhost", "127.0.0.1", "0.0.0.0", "::1"}

    def _normalize_url(self, value):
        value = self._clean_setting(value)
        if not value:
            return None
        if not value.startswith(("http://", "https://")):
            value = f"https://{value}"
        return value.rstrip("/")

    def _should_add_online_web_request_source(self, base_url=None) -> bool:
        params = self._load_fastdatasets_params()
        model_source = self._clean_setting(params.get("model_source"))
        if model_source == "infer":
            return True
        if model_source != "other":
            return False

        candidate_urls = {
            self._normalize_url(base_url),
            self._normalize_url(params.get("base_url")),
            self._normalize_url(params.get("synthesizer_url")),
        }
        return bool(candidate_urls & self._ONLINE_WEB_BASE_URLS)
    
    # 保持原有的简单接口，但内部使用高级实现
    async def call_llm(self, prompt, max_tokens=2048*2):
        """原始的简单LLM调用接口，保持向后兼容"""
        response = await self.call_llm_advanced(prompt=prompt, max_tokens=max_tokens)
        
        # 从响应中提取内容，保持向后兼容性
        if isinstance(response, dict) and 'choices' in response:
            content = response['choices'][0]['message']['content'].strip()
            return content
        return str(response)
    
    async def call_llm_advanced(self, prompt, max_tokens=2048*2, retries=None, backoff_factor=1.8, 
                                 dynamic_timeout=True, return_exceptions=False):
        """高级LLM调用接口，支持错误处理、重试机制、动态超时等功能"""
        # 异步信号量控制
        async with self.semaphore:
            retries = self.DEFAULT_RETRIES if retries is None else int(retries)
            retry_interval = self.DEFAULT_RETRY_INTERVAL_SECONDS
            total_attempts = max(1, retries)

            # 重新从环境变量获取配置，确保使用最新设置
            api_key, base_url, model_name = self._resolve_runtime_llm_settings()
            
            # 更新当前实例的设置
            self.api_key = api_key
            self.base_url = base_url
            self.model_name = model_name
            self.headers = {"Authorization": f"Bearer {self.api_key}"}
            
            # 确保 API URL 格式正确
            if self.base_url and not self.base_url.startswith(('http://', 'https://')):
                self.base_url = f"https://{self.base_url}"
                base_url = self.base_url
                
            # 检查必要参数
            if not self.api_key or not self.base_url or not self.model_name:
                error = LLMRequestError("缺少必要的LLM配置参数")
                logger.error(str(error))
                if return_exceptions:
                    return error
                raise error
            
            # 动态超时设置 - 根据prompt长度调整
            if dynamic_timeout:
                base_timeout = 60
                timeout_per_token = 0.06  # 每token增加的超时时间(秒)
                estimated_tokens = len(prompt) / 3  # 估算token数量
                timeout = base_timeout + min(480, estimated_tokens * timeout_per_token)  # 最多增加480秒(8分钟)
            else:
                timeout = 120*10  # 默认超时
            
            # 生成一个请求ID用于日志追踪
            request_id = f"req-{random.randint(1000, 9999)}"
            last_error = None
            
            for attempt in range(total_attempts):
                try:
                    # 随机化超时时间，避免所有请求同时超时
                    jitter = 1.0 + random.uniform(-0.15, 0.15)  # 随机因子±15%
                    current_timeout = timeout * jitter
                    if attempt > 0:
                        current_timeout *= (1 + attempt * 0.6)  # 每次重试增加60%超时时间
                    
                    logger.debug(f"[{request_id}] API调用超时设置: {current_timeout:.1f}秒 (尝试 {attempt+1}/{total_attempts})")
                    
                    # 准备请求数据
                    data = {
                        "model": self.model_name,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens,
                    }
                    if self._should_add_online_web_request_source(base_url=self.base_url):
                        data["request_source"] = "ONLINE_WEB"
                    
                    # 可选: 添加系统提示
                    if self.system_prompt:
                        data["messages"].insert(0, {"role": "system", "content": self.system_prompt})
                    
                    # 可选: 调整温度等参数
                    if hasattr(config, 'TEMPERATURE') and config.TEMPERATURE is not None:
                        data["temperature"] = float(config.TEMPERATURE)
                    
                    if hasattr(config, 'TOP_P') and config.TOP_P is not None:
                        data["top_p"] = float(config.TOP_P)
                    
                    # 日志记录开始信息 - 避免记录完整提示内容，只记录前30个字符
                    prompt_preview = prompt[:30].replace('\n', ' ') + "..." if len(prompt) > 30 else prompt
                    logger.debug(f"[{request_id}] 发送请求到 {self.model_name} (尝试 {attempt+1}/{total_attempts})")
                    logger.debug(f"[{request_id}] 提示预览: {prompt_preview}")
                    
                    start_time = time.time()
                    
                    # 使用异步上下文管理器创建客户端
                    async with httpx.AsyncClient(timeout=current_timeout) as client:
                        try:
                            # 发送请求
                            resp = await client.post(
                                f"{self.base_url}/chat/completions", 
                                headers=self.headers, 
                                json=data, 
                                follow_redirects=True
                            )
                            
                            # 检查状态码
                            resp.raise_for_status()
                            elapsed = time.time() - start_time
                            
                            # 解析响应
                            response_json = resp.json()
                            # print(f"完整响应: {response_json}")
                            
                            # 处理响应格式，返回完整的响应JSON，方便处理推理内容
                            logger.debug(f"[{request_id}] 请求成功，耗时 {elapsed:.2f}秒")
                            
                            # 返回完整响应JSON
                            return response_json
                            
                        except httpx.HTTPStatusError as e:
                            elapsed = time.time() - start_time
                            status_code = e.response.status_code
                            error_code = self._extract_error_code(e.response)
                            error_text = e.response.text[:200] + "..." if len(e.response.text) > 200 else e.response.text
                            
                            logger.error(f"[{request_id}] HTTP 错误 ({elapsed:.2f}秒): {status_code} - {error_text}")
                            
                            if status_code == 401:
                                error = LLMRequestError(f"API 密钥错误或未授权: {error_text}")
                                logger.error(f"[{request_id}] {error}")
                                if return_exceptions:
                                    return error
                                raise error
                                
                            elif status_code == 429:
                                logger.warning(f"[{request_id}] 请求频率限制，将重试")
                                wait_time = retry_interval
                                logger.warning(f"[{request_id}] 等待 {wait_time:.1f} 秒后重试...")
                                await asyncio.sleep(wait_time)
                                continue
                                
                            elif status_code >= 500:
                                logger.warning(f"[{request_id}] 服务器错误 ({status_code})，将重试")
                                wait_time = retry_interval
                                logger.warning(f"[{request_id}] 等待 {wait_time:.1f} 秒后重试...")
                                await asyncio.sleep(wait_time)
                                continue

                            elif status_code == 400 and error_code == 1601:
                                error = LLMRequestError(f"请求内容未通过审核 (HTTP 400): {error_text}")
                                logger.error(f"[{request_id}] {error}")
                                if return_exceptions:
                                    return error
                                raise error

                            elif 400 <= status_code < 500:
                                error = LLMRequestError(f"客户端请求错误 ({status_code}): {error_text}")
                                logger.error(f"[{request_id}] {error}")
                                if return_exceptions:
                                    return error
                                raise error
                                
                            # 其他HTTP错误
                            last_error = e
                            if attempt < total_attempts - 1:
                                wait_time = retry_interval
                                logger.warning(f"[{request_id}] 等待 {wait_time:.1f} 秒后重试...")
                                await asyncio.sleep(wait_time)
                            else:
                                logger.error(f"[{request_id}] 已达到最大重试次数")
                                if return_exceptions:
                                    return e
                                raise LLMRequestError(f"LLM 请求失败，已达到最大重试次数: {error_text}") from e
                        
                except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as e:
                    elapsed = time.time() - start_time if 'start_time' in locals() else 0
                    logger.warning(f"[{request_id}] 连接/读取错误 ({type(e).__name__}): {str(e)} ({elapsed:.1f}秒)")
                    last_error = e

                    if self._is_local_base_url(base_url):
                        error = LLMRequestError(f"本地 LLM 服务不可达 ({base_url}): {str(e)}")
                        logger.error(f"[{request_id}] {error}")
                        if return_exceptions:
                            return error
                        raise error
                    
                    if attempt < total_attempts - 1:
                        wait_time = retry_interval
                        logger.warning(f"[{request_id}] 将在 {wait_time:.1f} 秒后重试...")
                        await asyncio.sleep(wait_time)
                    else:
                        error = LLMRequestError(f"连接失败，已达到最大重试次数: {str(e)}")
                        logger.error(f"[{request_id}] {error}")
                        if return_exceptions:
                            return error
                        raise error
                        
                except LLMRequestError:
                    raise

                except Exception as e:
                    elapsed = time.time() - start_time if 'start_time' in locals() else 0
                    logger.error(f"[{request_id}] 调用 LLM API 失败 ({elapsed:.1f}秒): {str(e)}")
                    last_error = e
                    
                    if isinstance(logging.getLogger().level, int) and logging.getLogger().level <= logging.DEBUG:
                        logger.debug(f"[{request_id}] 异常详情: {traceback.format_exc()}")
                    
                    if attempt < total_attempts - 1:
                        wait_time = retry_interval
                        logger.warning(f"[{request_id}] 将在 {wait_time:.1f} 秒后重试...")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"[{request_id}] 已达到最大重试次数")
                        if return_exceptions:
                            return e
                        raise LLMRequestError(f"调用 LLM API 失败，已达到最大重试次数: {str(e)}") from e
            
            # 所有重试都失败，抛出最终异常
            logger.error(f"[{request_id}] 所有 API 调用尝试都失败")
            if return_exceptions:
                return LLMRequestError(f"所有API调用尝试都失败({total_attempts}次)")
            if isinstance(last_error, Exception):
                raise LLMRequestError(f"所有API调用尝试都失败({total_attempts}次): {str(last_error)}") from last_error
            raise LLMRequestError(f"所有API调用尝试都失败({total_attempts}次)")
    
    def _fallback_response(self, prompt: str) -> str:
        """当 LLM API 调用失败时的后备响应"""
        logger.warning("使用模拟回复代替 LLM 响应")
        prompt_lower = prompt.lower()

        if "优化建议" in prompt or "optimization suggestions" in prompt_lower:
            for marker in ("## 原始答案：", "## Original Answer:", "## 原始思维链：", "## Original COT:"):
                if marker in prompt:
                    original = prompt.split(marker, 1)[1]
                    original = re.split(r"\n\s*##\s+", original, maxsplit=1)[0]
                    return original.strip() or "模拟 LLM 响应"
            return "模拟 LLM 响应"

        if "生成不少于" in prompt or "generate no less than" in prompt_lower:
            match = re.search(r"生成不少于\s*(\d+)\s*个", prompt)
            if not match:
                match = re.search(r"no less than\s*(\d+)", prompt_lower)
            count = int(match.group(1)) if match else 1
            questions = [f"问题{i + 1}：请概括这段内容的关键信息？" for i in range(max(1, count))]
            return json.dumps(questions, ensure_ascii=False)

        if "为该问题生成2-3个相关领域标签" in prompt or "generate 2-3 relevant domain labels" in prompt_lower:
            return '["其他"]'

        if "## 问题" in prompt or "\n## question" in prompt_lower:
            return "这是基于输入内容生成的离线示例答案。当前 LLM 服务不可达，因此使用本地回退内容完成数据集导出。"

        return "模拟 LLM 响应"

# For sync usage or testing
class DummyLLM:
    def generate(self, prompt: str) -> str:
        return f"LLM output: {prompt}"

# 便于后续切换不同LLM
llm = AsyncLLM()
