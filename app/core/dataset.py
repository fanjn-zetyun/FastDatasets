import os
import json
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import asyncio
import httpx
from tqdm.asyncio import tqdm as tqdm_async
from app.core.logger import logger
from app.core.config import config
import random
import time
import logging
from app.core.llm import AsyncLLM

class DatasetBuilder:
    """数据集构建器，用于从文档块构建训练数据集"""
    
    def __init__(self):
        self.model_name = config.MODEL_NAME
        self.base_url = config.BASE_URL
        self.api_key = config.API_KEY
        self.language = config.LANGUAGE
        self.system_prompt = config.SYSTEM_PROMPT
        self.enable_cot = config.ENABLE_COT
        self.enable_label = config.ENABLE_LABEL
        self.enable_optimize = config.ENABLE_OPTIMIZE
        self.max_concurrency = config.MAX_LLM_CONCURRENCY
        self.questions_per_chunk = max(1, int(getattr(config, "DEFAULT_SAMPLE_SIZE", 1)))
        self.semaphore = asyncio.Semaphore(self.max_concurrency)
        self.headers = {"Authorization": f"Bearer {self.api_key}"}
        # 初始化LLM客户端
        self.llm = AsyncLLM(
            model_name=self.model_name,
            base_url=self.base_url,
            api_key=self.api_key,
            language=self.language,
            max_concurrency=self.max_concurrency,
            system_prompt=self.system_prompt
        )
        logger.info("DatasetBuilder 初始化")

    async def build_dataset(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        构建数据集 - 全异步处理
        
        Args:
            chunks: 文档块列表
            
        Returns:
            List[Dict[str, Any]]: 数据集
        """
        if not chunks:
            logger.warning("没有文档块，无法构建数据集")
            return []
            
        logger.info(f"开始构建数据集，共 {len(chunks)} 个文档块")
        
        # 步骤 1: 并行为每个文档块生成问题
        async def generate_questions_for_chunk(chunk):
            chunk_id = chunk.get('chunk_id', '')
            file_name = chunk.get('file', '')
            content = chunk.get('content', '')
            summary = chunk.get('summary', '')
            
            # 优先使用显式配置的问题数量；缺失时再按内容长度兜底
            question_count = self.questions_per_chunk or max(1, len(content) // 240)
            
            # 生成问题
            questions = await self._generate_questions(content, question_count)
            
            # 返回问题列表，每个问题包含完整的块信息
            return [{
                "chunk_id": chunk_id,
                "file": file_name,
                "summary": summary,
                "content": content,
                "question": q
            } for q in questions]
        
        # 并行处理所有文档块
        chunk_tasks = [generate_questions_for_chunk(chunk) for chunk in chunks]
        chunk_results = await tqdm_async.gather(*chunk_tasks, desc="生成问题")
        
        # 合并所有问题
        all_questions = []
        for result in chunk_results:
            all_questions.extend(result)
        
        logger.info(f"已生成 {len(all_questions)} 个问题，开始生成答案...")
        
        # 步骤 2: 并行为每个问题生成答案和相关内容
        async def process_question(item):
            # 获取问题和上下文
            question = item["question"]
            context = item["content"]
            
            # 定义要执行的任务
            tasks = [self._generate_answer(question, context)]
            
            # 注意: 思维链现在由 _generate_answer 根据 enable_cot 自动处理
            # 不再需要单独调用 _generate_cot
            
            # 如果启用了标签生成，添加到任务中
            if self.enable_label:
                tasks.append(self._generate_labels(question))
            
            # 同时执行所有任务
            results = await asyncio.gather(*tasks)
            
            data_point = dict(item)
            
            # 处理答案结果 (第一个任务)
            if isinstance(results[0], dict):
                if 'choices' in results[0]:
                    message = results[0]['choices'][0]['message']
                    data_point["answer"] = (message.get('content') or '').strip()
                    if 'reasoning_content' in message and message['reasoning_content']:
                        data_point["reasoning_content"] = message['reasoning_content'].strip()
                elif 'content' in results[0]:
                    data_point["answer"] = results[0]['content']
                    if 'reasoning_content' in results[0]:
                        data_point["reasoning_content"] = results[0]['reasoning_content']
                else:
                    data_point["answer"] = str(results[0])
            else:
                data_point["answer"] = results[0]
            
            # 处理其他任务结果
            task_index = 1
            
            if self.enable_label:
                data_point["labels"] = results[task_index]
                task_index += 1
            
            if self.enable_optimize:
                optimize_tasks = []
                
                if "answer" in data_point:
                    optimize_tasks.append(self._optimize_answer(data_point["answer"]))
                
                if self.enable_cot and "reasoning_content" in data_point:
                    optimize_tasks.append(self._optimize_cot(data_point["reasoning_content"]))
                
                if optimize_tasks:
                    optimize_results = await asyncio.gather(*optimize_tasks)
                    
                    result_index = 0
                    if "answer" in data_point:
                        data_point["answer"] = optimize_results[result_index]
                        result_index += 1
                    
                    if self.enable_cot and "reasoning_content" in data_point:
                        data_point["reasoning_content"] = optimize_results[result_index]
            
            return data_point
        
        # 动态计算最佳批处理大小
        max_concurrency = getattr(self, 'max_concurrency', 10)
        total_questions = len(all_questions)
        
        # 根据问题总数和并发数动态调整批处理大小
        if total_questions <= max_concurrency:
            # 问题数少于并发数，一次处理所有问题
            batch_size = total_questions
        elif total_questions <= max_concurrency * 3:
            # 问题数适中，设置较大的批处理大小
            batch_size = max(max_concurrency, total_questions // 2)
        else:
            # 问题数较多，使用较小的批处理大小避免内存问题
            batch_size = min(30, max(5, max_concurrency))
        
        logger.info(f"使用批处理大小: {batch_size}, 总并发数: {max_concurrency}")
        dataset = []
        
        # 计算总批次数
        total_batches = (total_questions + batch_size - 1) // batch_size
        for i in range(0, total_questions, batch_size):
            batch = all_questions[i:i+batch_size]
            batch_tasks = [process_question(item) for item in batch]
            
            # 使用tqdm显示进度
            batch_desc = f"生成答案 [批次 {i//batch_size+1}/{total_batches}]"
            batch_results = await tqdm_async.gather(*batch_tasks, desc=batch_desc)
            
            dataset.extend(batch_results)
            
            if batch_results:
                logger.info(f"批次 {i//batch_size+1}/{total_batches} 完成: 成功 {len(batch_results)}/{len(batch_results)}")
            
            # 短暂休息，避免API限制
            if i + batch_size < total_questions:
                await asyncio.sleep(0.5)
        
        logger.info(f"数据集构建完成，共 {len(dataset)} 个数据点")
        return dataset
    
    def save_dataset(self, dataset: List[Dict[str, Any]], output_path: str):
        """
        保存数据集
        
        Args:
            dataset: 数据集
            output_path: 输出路径
        """
        try:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 根据文件格式保存
            suffix = Path(output_path).suffix.lower()
            if suffix == ".json":
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(dataset, f, ensure_ascii=False, indent=2)
            elif suffix == ".jsonl":
                with open(output_path, 'w', encoding='utf-8') as f:
                    for item in dataset:
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')
            else:
                # 默认使用 JSONL 格式
                output_path = str(Path(output_path).with_suffix(".jsonl"))
                with open(output_path, 'w', encoding='utf-8') as f:
                    for item in dataset:
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')
                        
            logger.info(f"数据集已保存: {output_path}")
        except Exception as e:
            logger.error(f"保存数据集失败: {str(e)}")

    def _get_unique_path(self, base_path: str) -> str:
        """
        获取唯一的文件路径，如果文件存在则添加后缀 _1, _2 等
        
        Args:
            base_path: 基础路径
            
        Returns:
            唯一的文件路径
        """
        if not os.path.exists(base_path):
            return base_path
        
        # 分离路径和扩展名
        dir_path = os.path.dirname(base_path)
        filename = os.path.basename(base_path)
        name, ext = os.path.splitext(filename)
        
        # 尝试添加后缀
        counter = 1
        while True:
            new_filename = f"{name}_{counter}{ext}"
            new_path = os.path.join(dir_path, new_filename)
            if not os.path.exists(new_path):
                return new_path
            counter += 1

    def export_dataset(self, dataset: List[Dict[str, Any]], output_path: str, formats: List[str], file_format: str = "json", name: str = None):
        """
        导出数据集为多种格式
        
        Args:
            dataset: 数据集
            output_path: 输出路径，可以是目录或具体文件路径
            formats: 导出格式列表，如 ["alpaca", "sharegpt"]
            file_format: 文件格式，如 "json" 或 "jsonl"
            name: 数据集名称，默认为 "dataset"
        """
        # 如果提供了name参数，使用name作为数据集名称
        # 否则，从output_path中提取数据集名称
        if name:
            dataset_name = name
        else:
            # 从output_path中提取数据集名称
            # 例如：/workspace/user-data/datasets/fd-2026-03-31 -> fd-2026-03-31
            dataset_name = os.path.basename(output_path)
            # 如果dataset_name为空（例如output_path以/结尾），使用默认名称
            if not dataset_name:
                dataset_name = "dataset"
        
        # 判断 output_path 是文件路径还是目录
        _, ext = os.path.splitext(output_path)
        is_file_path = ext in [".json", ".jsonl"]
        
        if is_file_path:
            # output_path 是具体文件路径，使用 name 作为文件名基础
            output_dir = os.path.dirname(output_path) or "."
            os.makedirs(output_dir, exist_ok=True)
        else:
            # output_path 是目录，但我们将它视为数据集名称的基础
            # 例如：output_path = /workspace/user-data/datasets/fd-2026-03-31
            # 我们将在这个目录下生成文件，文件名基于dataset_name
            output_dir = os.path.dirname(output_path) or "."
            # 确保输出目录存在
            os.makedirs(output_dir, exist_ok=True)

        exported_paths: List[str] = []
        for fmt in formats:
            if fmt == "alpaca":
                export_data = self._export_alpaca(dataset)
                out_path = os.path.join(output_dir, f"{dataset_name}-alpaca.{file_format}")
            elif fmt == "sharegpt":
                export_data = self._export_sharegpt(dataset)
                out_path = os.path.join(output_dir, f"{dataset_name}-sharegpt.{file_format}")
            else:
                logger.warning(f"不支持的导出格式: {fmt}")
                continue
            
            # 获取唯一路径（处理文件名冲突）
            out_path = self._get_unique_path(out_path)
                
            self._save_output(export_data, out_path, file_format)
            logger.info(f"已导出 {fmt} 格式: {out_path}")
            exported_paths.append(out_path)

        self._notify_export_results(exported_paths)

    def _notify_export_results(self, exported_paths: List[str]) -> None:
        """在数据集文件成功生成后回调结果路径。"""
        callback_url = (os.getenv("CALLBACK_URL") or "").strip()
        task_id = (os.getenv("TASK_ID") or "").strip()

        if not callback_url:
            logger.info("未配置 CALLBACK_URL，跳过数据结果路径回调")
            self._append_callback_log("未配置 CALLBACK_URL，跳过数据结果路径回调")
            return

        if not task_id:
            logger.warning("未配置 TASK_ID，跳过数据结果路径回调")
            self._append_callback_log("未配置 TASK_ID，跳过数据结果路径回调")
            return

        successful_paths = [path for path in exported_paths if path and os.path.exists(path)]
        if not successful_paths:
            logger.warning("没有成功生成的数据集文件，跳过数据结果路径回调")
            self._append_callback_log("没有成功生成的数据集文件，跳过数据结果路径回调")
            return

        payload = [{"id": task_id, "resultPath": path} for path in successful_paths]
        self._append_callback_log(
            f"开始数据结果路径回调: url={callback_url}, payload={json.dumps(payload, ensure_ascii=False)}"
        )

        try:
            response = httpx.post(callback_url, json=payload, timeout=30.0)
            response.raise_for_status()
            logger.info(f"数据结果路径回调成功")
            self._append_callback_log(
                f"数据结果路径回调成功: url={callback_url}, status_code={response.status_code}, response={response.text}"
            )
        except Exception as exc:
            logger.error(f"数据结果路径回调失败 error: {exc}")
            self._append_callback_log(f"数据结果路径回调失败: url={callback_url}, error={exc}")

    def _append_callback_log(self, message: str) -> None:
        """将回调日志追加到 entrypoint.sh 使用的同一日志文件。"""
        log_file = (os.getenv("FASTDATASETS_LOG_FILE") or "").strip()
        if not log_file:
            return

        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {message}\n"
        try:
            with open(log_file, "a", encoding="utf-8") as file_obj:
                file_obj.write(line)
        except Exception as exc:
            logger.error(f"写入回调日志文件失败: {log_file}, error: {exc}")
    
    def _export_alpaca(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """导出为 Alpaca 格式"""
        result = []
        for item in data:
            # 打印调试信息
            print(f"\n=== 处理数据点 ===")
            print(f"问题: {item['question'][:50]}...")
            print(f"回答: {item['answer'][:50]}...")
            print(f"enable_cot: {self.enable_cot}")
            
            # 构建输出
            output = ""
            if self.enable_cot and 'reasoning_content' in item and item.get('reasoning_content'):
                print("添加推理内容到输出")
                output = f"<think>\n{item.get('reasoning_content', '')}\n</think>\n\n{self._clean_markdown_json(item['answer'])}"
            else:
                # enable_cot=False 或没有 reasoning_content
                # 根据 enable_cot 决定是否移除思考标签
                output = self._clean_markdown_json(item["answer"], remove_think_tags=not self.enable_cot)
            
            result.append({
                "instruction": self._clean_markdown_json(item["question"]),
                "input": "",
                "output": self._clean_optimized_output(output)
                # "system": self.system_prompt or ""
            })
            
        return result
    
    def _export_sharegpt(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """导出为 ShareGPT 格式"""
        result = []
        for item in data:
            messages = []
            # if self.system_prompt:
            #     messages.append({"role": "system", "content": self.system_prompt})
            messages.append({"role": "user", "content": self._clean_markdown_json(item["question"])})
            
            # 根据 enable_cot 构建回答内容
            if self.enable_cot and 'reasoning_content' in item and item.get('reasoning_content'):
                reasoning = item.get('reasoning_content', '')
                answer = self._clean_markdown_json(item['answer'])
                assistant_content = f"<think>\n{reasoning}\n</think>\n\n{answer}"
            else:
                # enable_cot=False 或没有 reasoning_content
                # 根据 enable_cot 决定是否移除思考标签
                assistant_content = self._clean_optimized_output(self._clean_markdown_json(item["answer"], remove_think_tags=not self.enable_cot))
            
            messages.append({"role": "assistant", "content": assistant_content})
            result.append({"messages": messages})
        return result
    
    def _save_output(self, data: List[Dict[str, Any]], output_path: str, file_format: str = "json"):
        """保存输出文件"""
        if file_format == "jsonl":
            with open(output_path, "w", encoding="utf-8") as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
        else:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

    def _clean_markdown_json(self, text: str, remove_think_tags: bool = True) -> str:
        """清理 Markdown 中的 JSON 格式
        
        Args:
            text: 要清理的文本
            remove_think_tags: 是否移除思考标签（默认True）
        """
        if remove_think_tags:
            # 去除思考标签 - 多种格式
            # DeepSeek 风格: <|im_start|>think<|im_end|>...<|im_start|>/think<|im_end|>
            text = re.sub(r"<\|im_start\|>think<\|im_end\|>.*?<\|im_start\|>/think<\|im_end\|>", "", text, flags=re.DOTALL)
            # Markdown 代码块风格: ```think...```
            text = re.sub(r"```think.*?```", "", text, flags=re.DOTALL)
            # 简单标签风格: <think...>
            text = re.sub(r"<think[^>]*>.*?</think\s*>", "", text, flags=re.DOTALL | re.IGNORECASE)
            # 可能的残留标签
            text = re.sub(r"<\|im_start\|>think<\|im_end\|>", "", text)
            text = re.sub(r"<\|im_start\|>/think<\|im_end\|>", "", text)
            text = re.sub(r"<think[^>]*/>", "", text)
            text = re.sub(r"</think\s*>", "", text)
            
            # 处理 ohl 格式（可能是特殊编码的思考内容）
            # 使用贪婪匹配，直到找到标准格式开头
            text = re.sub(r"^ohl\s*\n[\s\S]*?\n{2,}(?=\*\*问题|\*\*答案|\*\*Question|\*\*Answer|\[|\{)", "", text, flags=re.DOTALL)
            text = re.sub(r"^ohl\s*\n", "", text)
        
        # 去除开头和结尾的```、```json、首尾空行
        text = re.sub(r"^\s*```json\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"^\s*```\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
        text = re.sub(r"\s*```json\s*$", "", text, flags=re.IGNORECASE)
        text = text.strip()
        # 如果是json数组，尝试解析后再转为字符串
        try:
            obj = json.loads(text)
            if isinstance(obj, list):
                return "\n".join(str(q).strip() for q in obj)
            if isinstance(obj, str):
                return obj.strip()
        except Exception:
            pass
        return text




    def _clean_optimized_output(self, text: str) -> str:
        """清理优化后的输出"""
        # 处理 ohl 开头的格式（可能是特殊编码的思考内容）
        # 匹配从 ohl 开始到实际内容之间的所有内容（使用贪婪匹配直到找到标准格式）
        text = re.sub(r"^ohl\s*\n[\s\S]*?\n{2,}(?=\*\*问题|\*\*答案|\*\*Question|\*\*Answer)", "", text, flags=re.DOTALL)
        text = re.sub(r"^ohl\s*\n", "", text)
        # 去除常见冗余前缀
        text = re.sub(r"^#+\s*优化后的答案内容[:：]?\s*", "", text)
        text = re.sub(r"^优化后的答案内容[:：]?\s*", "", text)
        text = re.sub(r"^#+\s*Optimized answer content[:：]?\s*", "", text)
        text = re.sub(r"^Optimized answer content[:：]?\s*", "", text)
        text = re.sub(r"^#+\s*优化后的思维链内容[:：]?\s*", "", text)
        text = re.sub(r"^优化后的思维链内容[:：]?\s*", "", text)
        text = re.sub(r"^#+\s*Optimized COT content[:：]?\s*", "", text)
        text = re.sub(r"^Optimized COT content[:：]?\s*", "", text)
        return text.strip()

    def _extract_json_array_text(self, text: str) -> Optional[str]:
        """从模型输出中提取首个 JSON 数组文本。"""
        if not text:
            return None

        fenced_match = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", text, flags=re.IGNORECASE)
        if fenced_match:
            return fenced_match.group(1).strip()

        bracket_match = re.search(r"(\[[\s\S]*?\])", text)
        if bracket_match:
            return bracket_match.group(1).strip()

        return None


    
    async def _generate_questions(self, context: str, number: int = 5) -> List[str]:
        """生成问题"""
        logger.info(f"生成问题: 生成 {number} 个问题...")
        
        # 构建 prompt
        if self.language == '中文':
            prompt = f"""
# 角色使命
你是一位专业的文本分析专家，擅长从复杂文本中提取关键信息并生成可用于模型微调的结构化数据（仅生成问题）。

## 核心任务
根据用户提供的文本（长度：{len(context)} 字），生成不少于 {number} 个高质量问题。

## 约束条件（重要！）
- 必须基于文本内容直接生成
- 问题应具有明确答案指向性
- 需覆盖文本的不同方面
- 禁止生成假设性、重复或相似问题

## 处理流程
1. 【文本解析】分段处理内容，识别关键实体和核心概念
2. 【问题生成】基于信息密度选择最佳提问点
3. 【质量检查】确保：
   - 问题答案可在原文中找到依据
   - 标签与问题内容强相关
   - 无格式错误

## 输出格式
- JSON 数组格式必须正确
- 字段名使用英文双引号
- 输出的 JSON 数组必须严格符合以下结构：
```json
["问题1", "问题2", "..."]
```

## 待处理文本
{context}

## 限制
- 必须按照规定的 JSON 格式输出，不要输出任何其他不相关内容
- 生成不少于{number}个高质量问题
- 问题不要和材料本身相关，例如禁止出现作者、章节、目录等相关问题
- 问题不得包含【报告、文章、文献、表格】中提到的这种话术，必须是一个自然的问题
"""
        else:
            prompt = f"""
# Role Mission
You are a professional text analysis expert, skilled at extracting key information from complex texts and generating structured data(only generate questions) that can be used for model fine-tuning.

## Core Task
Based on the text provided by the user(length: {len(context)} characters), generate no less than {number} high-quality questions.

## Constraints(Important!)
✔️ Must be directly generated based on the text content.
✔️ Questions should have a clear answer orientation.
✔️ Should cover different aspects of the text.
❌ It is prohibited to generate hypothetical, repetitive, or similar questions.

## Processing Flow
1. 【Text Parsing】Process the content in segments, identify key entities and core concepts.
2. 【Question Generation】Select the best questioning points based on the information density.
3. 【Quality Check】Ensure that:
   - The answers to the questions can be found in the original text.
   - The labels are strongly related to the question content.
   - There are no formatting errors.

## Output Format
- The JSON array format must be correct.
- Use English double-quotes for field names.
- The output JSON array must strictly follow the following structure:
```json
["Question 1", "Question 2", "..."]
```

## Text to be Processed
{context}

## Restrictions
- Must output in the specified JSON format and do not output any other irrelevant content.
- Generate no less than {number} high-quality questions.
- Questions should not be related to the material itself. For example, questions related to the author, chapters, table of contents, etc. are prohibited.
"""
        
        # 调用统一的LLM服务
        response = await self.llm.call_llm_advanced(prompt)
        
        print(f"\n=== 生成问题API响应 ===\n{response}\n=================\n")
        
        try:
            # 处理API响应
            if isinstance(response, dict) and 'choices' in response:
                content = response['choices'][0]['message']['content']
                json_array_text = self._extract_json_array_text(content)
                cleaned_content = json_array_text or self._clean_markdown_json(content)
                # 尝试解析JSON
                try:
                    questions = json.loads(cleaned_content)
                    if isinstance(questions, list):
                        return [str(q).strip() for q in questions if str(q).strip()]
                    else:
                        # 可能返回的是包含问题的对象
                        return [cleaned_content]
                except Exception as e:
                    print(f"解析问题JSON失败: {str(e)}")
                    print(f"原始内容: {content}")
                    print(f"清理后内容: {cleaned_content}")
                    lines = [line.strip() for line in cleaned_content.split('\n') if line.strip()]
                    return lines if lines else [cleaned_content]
            else:
                # 旧版响应处理
                content = str(response)
                json_array_text = self._extract_json_array_text(content)
                cleaned_content = json_array_text or self._clean_markdown_json(content)
                try:
                    # 尝试解析 JSON
                    questions = json.loads(cleaned_content)
                    if isinstance(questions, list):
                        return [str(q).strip() for q in questions if str(q).strip()]
                    else:
                        # 可能返回的是包含问题的对象
                        return [cleaned_content]
                except Exception:
                    lines = [line.strip() for line in cleaned_content.split('\n') if line.strip()]
                    return lines if lines else [cleaned_content]
        except Exception as e:
            logger.error(f"处理生成问题响应失败: {str(e)}")
            raise

    async def _generate_answer(self, question: str, context: str) -> str:
        """生成答案"""
        logger.info(f"生成答案: 问题: {question[:20]}...")
        
        # 构建 prompt
        if self.language == '中文':
            prompt = f"""
# Role: 微调数据集生成专家
## Profile:
- Description: 你是一名微调数据集生成专家，擅长从给定的内容中生成准确的问题答案，确保答案的准确性和相关性，你要直接回答用户问题，所有信息已内化为你的专业知识。

## Skills   :
1. 答案必须基于给定的内容
2. 答案必须准确，不能胡编乱造
3. 答案必须与问题相关
4. 答案必须符合逻辑
5. 基于给定参考内容，用自然流畅的语言整合成一个完整答案，不需要提及文献来源或引用标记
   
## Workflow:
1. Take a deep breath and work on this problem step-by-step.
2. 首先，分析给定的文件内容
3. 然后，从内容中提取关键信息
4. 接着，生成与问题相关的准确答案
5. 最后，确保答案的准确性和相关性

## 参考内容：
{context}

## 问题
{question}

## Constrains:
1. 答案必须基于给定的内容
2. 答案必须准确，必须与问题相关，不能胡编乱造
"""
        else:
            prompt = f"""
# Role: Fine-tuning Dataset Generation Expert
## Profile:
- Description: You are a fine-tuning dataset generation expert, skilled at generating accurate question-answer pairs from given content, ensuring answer accuracy and relevance. You should directly answer user questions, with all information internalized as your professional knowledge.

## Skills:
1. Answers must be based on the given content
2. Answers must be accurate, no fabrication
3. Answers must be relevant to the question
4. Answers must be logical
5. Based on the given reference content, integrate into a complete answer using natural and fluent language, no need to mention source or citation marks

## Workflow:
1. Take a deep breath and work on this problem step-by-step.
2. First, analyze the given content
3. Then, extract key information from the content
4. Next, generate accurate answers related to the question
5. Finally, ensure answer accuracy and relevance

## Reference Content:
{context}

## Question
{question}

## Constraints:
1. Answers must be based on the given content
2. Answers must be accurate and relevant to the question, no fabrication
"""
        
        # 调用统一的LLM服务
        response = await self.llm.call_llm_advanced(prompt)
        
        # 打印原始响应，用于调试
        print(f"\n=== API响应 ===\n{response}\n=================\n")
        
        # 处理响应
        if isinstance(response, dict) and 'choices' in response:
            message = response['choices'][0]['message']
            content = (message.get('content') or '').strip()
            reasoning_content = (message.get('reasoning_content') or '').strip()

            # 根据 enable_cot 决定是否清理思考标签
            # enable_cot=True: 保留思考标签（思维链在答案中）
            # enable_cot=False: 移除思考标签
            if not self.enable_cot:
                content = self._remove_think_tags(content)
                reasoning_content = ""  # 不保留推理内容

            # 如果启用了推理内容且存在推理内容，则返回包含推理内容的字典
            if self.enable_cot and reasoning_content:
                return {
                    'content': content,
                    'reasoning_content': reasoning_content
                }
            return content
        return str(response)

    def _remove_think_tags(self, text: str) -> str:
        """移除思考标签（如 <|im_start|>think<|im_end|>... 或其他格式）"""
        # 处理 DeepSeek 风格的思考标签
        text = re.sub(r"<\|im_start\|>think<\|im_end\|>.*?<\|im_start\|>/think<\|im_end\|>", "", text, flags=re.DOTALL)
        # 处理 ohl 开头的格式（可能是特殊编码的思考内容）
        text = re.sub(r"^ohl\s+\n.*?\n{2,}(?=\*\*问题|\*\*答案|问题\d)", "", text, flags=re.DOTALL)
        # 处理其他常见思考标签格式
        text = re.sub(r"<think[^>]*>.*?</think\s*>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"```think.*?```", "", text, flags=re.DOTALL)
        # 清理可能残留的开始或结束标签
        text = re.sub(r"<\|im_start\|>think<\|im_end\|>", "", text)
        text = re.sub(r"<\|im_start\|>/think<\|im_end\|>", "", text)
        text = re.sub(r"<think[^>]*/>", "", text)
        text = re.sub(r"</think\s*>", "", text)
        return text.strip()



    async def _generate_cot(self, question: str) -> str:
        """生成思维链"""
        logger.info(f"生成思维链: 问题: {question[:20]}...")
        
        # 构建 prompt
        if self.language == '中文':
            prompt = f"""
# Role: 思维链生成专家
- Description: 你是一名思维链生成专家，擅长为问题生成详细的推理过程。

## 问题：
{question}

## Workflow:
1. 分析问题，分解为多个推理步骤。
2. 详细描述每一步推理过程。
3. 输出完整的思维链。

## Output Example:
<think>首先...然后...最后...</think>
"""
        else:
            prompt = f"""
# Role: Chain-of-Thought Generation Expert
- Description: You are an expert in generating detailed reasoning chains for questions.

## Question:
{question}

## Workflow:
1. Analyze the question and break it down into multiple reasoning steps.
2. Describe each reasoning step in detail.
3. Output the complete chain of thought.

## Output Example:
<think>First... Then... Finally...</think>
"""
        
        # 调用统一的LLM服务
        response = await self.llm.call_llm_advanced(prompt)
        
        # 处理响应格式
        try:
            if isinstance(response, dict) and 'choices' in response:
                content = response['choices'][0]['message']['content'].strip()
            else:
                content = str(response)
        except Exception as e:
            logger.error(f"处理优化思维链响应失败: {str(e)}")
            content = str(response)
            
        return self._clean_optimized_output(content)

    async def _generate_labels(self, question: str) -> List[str]:
        """生成标签"""
        logger.info(f"生成标签: 问题: {question[:20]}...")
        
        # 构建 prompt
        if self.language == '中文':
            prompt = f"""
# Role: 领域分类专家
- Description: 你是一名标签分类专家，能够根据问题内容生成相关的标签。

## 问题：
{question}

## 任务：
为该问题生成2-3个相关领域标签，这些标签应该能够概括问题所属的知识领域。

## 输出格式：
[
  "标签1",
  "标签2",
  "标签3"
]
"""
        else:
            prompt = f"""
# Role: Domain Classification Expert
- Description: You are a label classification expert who can generate relevant labels based on the content of questions.

## Question:
{question}

## Task:
Generate 2-3 relevant domain labels for this question. These labels should be able to summarize the knowledge domain to which the question belongs.

## Output Format:
[
  "Label1",
  "Label2",
  "Label3"
]
"""
        
        # 调用统一的LLM服务
        response = await self.llm.call_llm_advanced(prompt)
        
        # 处理响应格式
        try:
            if isinstance(response, dict) and 'choices' in response:
                content = response['choices'][0]['message']['content'].strip()
            else:
                content = str(response)
        except Exception as e:
            logger.error(f"处理标签响应失败: {str(e)}")
            content = str(response)
            
        try:
            labels = json.loads(content)
            if isinstance(labels, list):
                return labels
            else:
                return ["其他"]
        except Exception:
            return ["其他"]

    async def _optimize_answer(self, answer: str) -> str:
        """优化答案"""
        logger.info(f"优化答案: {answer[:20]}...")
        
        # 构建 prompt
        if self.language == '中文':
            prompt = f"""
# Role: 答案优化专家
- Description: 你是一名答案优化专家，擅长优化答案。

## 原始答案：
{answer}

## 优化建议：
1. 使答案更准确、简洁、无引用性表述
2. 确保答案自然流畅，避免冗余
3. 删除所有引用性表述如"根据文章"、"参考文献表明"等
4. 确保内容与原始答案保持一致，不要添加新信息

请直接输出优化后的内容，不要包含任何多余的前缀或标题。
"""
        else:
            prompt = f"""
# Role: Answer Optimization Expert
- Description: You are an expert in optimizing answers based on suggestions.

## Original Answer:
{answer}

## Optimization Suggestions:
1. Make the answer more accurate, concise, and without citation expressions
2. Ensure the answer is natural and fluent, avoiding redundancy
3. Remove all citation expressions such as "according to the article", "the reference shows", etc.
4. Ensure the content is consistent with the original answer, do not add new information

Please output only the optimized answer content, without any extra prefix or title.
"""
        
        # 调用统一的LLM服务
        response = await self.llm.call_llm_advanced(prompt)
        
        # 处理响应格式
        try:
            if isinstance(response, dict) and 'choices' in response:
                content = response['choices'][0]['message']['content'].strip()
            else:
                content = str(response)
        except Exception as e:
            logger.error(f"处理优化答案响应失败: {str(e)}")
            content = str(response)
            
        return self._clean_optimized_output(content)

    async def _optimize_cot(self, cot: str) -> str:
        """优化思维链"""
        logger.info(f"优化思维链: {cot[:20]}...")
        
        # 构建 prompt
        if self.language == '中文':
            prompt = f"""
# Role: 思维链优化专家
- Description: 你是一名思维链优化专家，擅长优化思维链。

## 原始思维链：
{cot}

## 优化建议：
1. 使思维链更自然、流畅
2. 去除引用性表述
3. 确保逻辑清晰
4. 保持内容与原始思维链一致

请直接输出优化后的内容，不要包含任何多余的前缀或标题。
"""
        else:
            prompt = f"""
# Role: COT Optimization Expert
- Description: You are an expert in optimizing chains of thought.

## Original COT:
{cot}

## Optimization Suggestions:
1. Make the chain of thought more natural and fluent
2. Remove citation expressions
3. Ensure clear logic
4. Maintain consistency with the original chain of thought

Please output only the optimized COT content, without any extra prefix or title.
"""
        
        # 调用统一的LLM服务
        content = await self.llm.call_llm_advanced(prompt)
        return self._clean_optimized_output(content)
