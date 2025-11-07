import os
import json
from dotenv import load_dotenv
from typing import Union, List, Dict, Type, Optional, Literal
from openai import OpenAI
import asyncio
from src.api_request_parallel_processor import process_api_requests_from_file
from openai.lib._parsing import type_to_response_format_param 
import tiktoken
import src.prompts as prompts
import requests
from json_repair import repair_json
from pydantic import BaseModel
import google.generativeai as genai
from copy import deepcopy
import dashscope
from typing import Union, List
import re
import requests
import json

from tenacity import retry, stop_after_attempt, wait_fixed

class BaseQwenProcessor:
        def __init__(self):
            load_dotenv()
            # 优先用 DASHSCOPE_API_KEY，兼容 QWEN_API_KEY
            self.api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY")
            self.base_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
            self.default_model = "qwen-turbo"  # 可根据实际模型名调整
            self.response_data = None  # 新增：用于保存最近一次API响应内容

        @retry(
            wait=wait_fixed(50),
            stop=stop_after_attempt(3),
        )
        def get_embeddings(self, texts: Union[str, List[str], List[dict]], model="text-embedding-v4", max_length=4000, batch_size=10) -> List[dict]:
            """
            获取文本的 Dashscope 向量表示（批量），返回格式兼容 `_get_embeddings`
            兼容输入为字符串、字符串列表、或问题字典列表（自动提取 text/question 字段）
            """
            def extract_text(q):
                # 兼容新旧格式
                if isinstance(q, str):
                    return q
                if isinstance(q, dict):
                    return q.get("question") or q.get("text")
                return None

            # 清洗文本
            def clean_text(text: str) -> str:
                text = re.sub(r"\s+", " ", text).strip()
                return "".join(ch for ch in text if ch.isprintable())

            # 截断文本
            def truncate_text(text: str, max_len: int) -> str:
                if len(text) > max_len:
                    print(f"[DEBUG] Truncating text from {len(text)}")
                return text[:max_len] if len(text) > max_len else text

            # 设置 Dashscope API
            dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'  # 国内版
            dashscope.api_key = "sk-33633db8e3574123b255ed9f0b6b09d8"

            # 统一为字符串列表
            if isinstance(texts, str):
                texts = [texts]

            # 自动提取文本字段，过滤 None 和空串
            texts_extracted = [extract_text(t) for t in texts]
            texts_extracted = [t for t in texts_extracted if isinstance(t, str) and t.strip()]

            # 清洗 + 截断
            texts_clean = [truncate_text(clean_text(t), max_length) for t in texts_extracted]

            results = []
            batch_counter = 0

            import sys
            for i in range(0, len(texts_clean), batch_size):
                batch = texts_clean[i:i + batch_size]
                print(f"[DEBUG] [Dashscope Embedding] Batch {i//batch_size + 1}, size: {len(batch)}")
                sys.stdout.flush()
                try:
                    resp = dashscope.TextEmbedding.call(
                        model=model,
                        input=batch,
                        dimension=1024,  # 根据模型选定向量维度
                        output_type="dense"
                    )
                    print(f"[DEBUG] [Dashscope Embedding] Status: {resp.get('status_code')}")
                    sys.stdout.flush()

                    if resp.get('status_code') == 200:
                        embeddings = resp.get('output', {}).get('embeddings', [])
                        # 将每条 embedding 对应原文本
                        for j, emb in enumerate(embeddings):
                            text_index = j + batch_counter
                            if text_index < len(texts_clean):
                                results.append({"embedding": emb.get('embedding', []), "text": texts_clean[text_index]})
                    else:
                        print(f"[Dashscope error] status: {resp.get('status_code')}, message: {resp}")

                except Exception as e:
                    print(f"[Dashscope exception] {e}")

                batch_counter += len(batch)

            return results

        # def __init__(self):
        #     load_dotenv()
        #     # 优先用 DASHSCOPE_API_KEY，兼容 QWEN_API_KEY
        #     self.api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY")
        #     self.base_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
        #     self.default_model = "qwen-turbo"  # 可根据实际模型名调整

        def send_message(
            self,
            model=None,
            temperature=0.5,
            seed=None,
            system_content="You are a helpful assistant.",
            human_content="Hello!",
            is_structured=False,
            response_format=None,
            **kwargs
        ):
            if model is None:
                model = self.default_model

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": model,
                "input": {
                    "messages": [
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": human_content}
                    ]
                },
                "parameters": {
                    "temperature": temperature
                }
            }
            if seed is not None:
                payload["parameters"]["seed"] = seed

            import sys
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=60)
            print(f"[DEBUG] [Qwen API] Status: {response.status_code}, Response length: {len(response.text)} bytes")
            sys.stdout.flush()
            # Uncomment below to see full response (verbose):
            # print(f"[DEBUG] [Qwen API] Full response: {response.text}")
            response.raise_for_status()
            result = response.json()
            # 保存最近一次API响应内容
            self.response_data = {
                "status_code": response.status_code,
                "response_text": response.text,
                "response_json": result,
                "payload": payload
            }
            # Qwen返回格式可能不同，需根据实际API文档调整
            content = result.get("output", {}).get("text", "")

            if is_structured and response_format is not None:
                try:
                    repaired_json = repair_json(content)
                    parsed_dict = json.loads(repaired_json)
                    
                    # 防御性检查：如果解析出来的是列表，尝试提取第一个元素
                    if isinstance(parsed_dict, list):
                        print(f"[WARNING] Parsed JSON is a list, extracting first element")
                        if parsed_dict and isinstance(parsed_dict[0], dict):
                            parsed_dict = parsed_dict[0]
                        else:
                            raise ValueError("Parsed JSON is a list but first element is not a dict")
                    
                    validated_data = response_format.model_validate(parsed_dict)
                    content = validated_data.model_dump()
                except Exception as e:
                    print(f"Qwen structured output parse error: {e}")
                    # 尝试返回解析后的字典，即使验证失败
                    try:
                        repaired_json = repair_json(content)
                        parsed_dict = json.loads(repaired_json)
                        
                        # 再次检查列表情况
                        if isinstance(parsed_dict, list):
                            print(f"[WARNING] Parsed JSON is a list in fallback, extracting first element")
                            if parsed_dict and isinstance(parsed_dict[0], dict):
                                parsed_dict = parsed_dict[0]
                        
                        print(f"[DEBUG] Returning parsed dict despite validation error")
                        content = parsed_dict
                    except Exception as parse_err:
                        print(f"[ERROR] Failed to parse JSON: {parse_err}")
                        # 返回一个安全的默认字典
                        content = {
                            "step_by_step_analysis": "解析错误",
                            "reasoning_summary": "无法解析LLM响应",
                            "relevant_pages": [],
                            "final_answer": "解析失败"
                        }
            return content

class BaseOpenaiProcessor:
    def __init__(self):
        self.llm = self.set_up_llm()
        self.default_model = 'gpt-4o-2024-08-06'
        # self.default_model = 'gpt-4o-mini-2024-07-18',

    def set_up_llm(self):
        load_dotenv()
        llm = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=None,
            max_retries=2
            )
        return llm

    def send_message(
        self,
        model=None,
        temperature=0.5,
        seed=None, # For deterministic ouptputs
        system_content='You are a helpful assistant.',
        human_content='Hello!',
        is_structured=False,
        response_format=None
        ):
        if model is None:
            model = self.default_model
        params = {
            "model": model,
            "seed": seed,
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": human_content}
            ]
        }
        
        # Reasoning models do not support temperature
        if "o3-mini" not in model:
            params["temperature"] = temperature
            
        if not is_structured:
            completion = self.llm.chat.completions.create(**params)
            content = completion.choices[0].message.content

        elif is_structured:
            params["response_format"] = response_format
            completion = self.llm.beta.chat.completions.parse(**params)

            response = completion.choices[0].message.parsed
            content = response.dict()

        self.response_data = {"model": completion.model, "input_tokens": completion.usage.prompt_tokens, "output_tokens": completion.usage.completion_tokens}
        print(self.response_data)

        return content

    @staticmethod
    def count_tokens(string, encoding_name="o200k_base"):
        encoding = tiktoken.get_encoding(encoding_name)

        # Encode the string and count the tokens
        tokens = encoding.encode(string)
        token_count = len(tokens)

        return token_count


class BaseIBMAPIProcessor:
    def __init__(self):
        load_dotenv()
        self.api_token = os.getenv("IBM_API_KEY")
        self.base_url = "https://rag.timetoact.at/ibm"
        self.default_model = 'meta-llama/llama-3-3-70b-instruct'
    def check_balance(self):
        """Check the current balance for the provided token."""
        balance_url = f"{self.base_url}/balance"
        headers = {"Authorization": f"Bearer {self.api_token}"}
        
        try:
            response = requests.get(balance_url, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as err:
            print(f"Error checking balance: {err}")
            return None
    
    def get_available_models(self):
        """Get a list of available foundation models."""
        models_url = f"{self.base_url}/foundation_model_specs"
        
        try:
            response = requests.get(models_url)
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as err:
            print(f"Error getting available models: {err}")
            return None
    
    def get_embeddings(self, texts, model_id="ibm/granite-embedding-278m-multilingual"):
        """Get vector embeddings for the provided text inputs."""
        embeddings_url = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        payload = {
            "inputs": texts,
            "model_id": model_id
        }
        
        try:
            response = requests.post(embeddings_url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as err:
            print(f"Error getting embeddings: {err}")
            return None
    
    def send_message(
        self,
        # model='meta-llama/llama-3-1-8b-instruct',
        model=None,
        temperature=0.5,
        seed=None,  # For deterministic outputs
        system_content='You are a helpful assistant.',
        human_content='Hello!',
        is_structured=False,
        response_format=None,
        max_new_tokens=5000,
        min_new_tokens=1,
        **kwargs
    ):
        if model is None:
            model = self.default_model
        text_generation_url = f"{self.base_url}/text_generation"
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        
        # Prepare the input messages
        input_messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": human_content}
        ]
        
        # Prepare parameters with defaults and any additional parameters
        parameters = {
            "temperature": temperature,
            "random_seed": seed,
            "max_new_tokens": max_new_tokens,
            "min_new_tokens": min_new_tokens,
            **kwargs
        }
        
        payload = {
            "input": input_messages,
            "model_id": model,
            "parameters": parameters
        }
        
        try:
            response = requests.post(text_generation_url, headers=headers, json=payload)
            response.raise_for_status()
            completion = response.json()

            content = completion.get("results")[0].get("generated_text")
            self.response_data = {"model": completion.get("model_id"), "input_tokens": completion.get("results")[0].get("input_token_count"), "output_tokens": completion.get("results")[0].get("generated_token_count")}
            print(self.response_data)
            if is_structured and response_format is not None:
                try:
                    repaired_json = repair_json(content)
                    parsed_dict = json.loads(repaired_json)
                    validated_data = response_format.model_validate(parsed_dict)
                    content = validated_data.model_dump()
                    return content
                
                except Exception as err:
                    print("Error processing structured response, attempting to reparse the response...")
                    reparsed = self._reparse_response(content, system_content)
                    try:
                        repaired_json = repair_json(reparsed)
                        reparsed_dict = json.loads(repaired_json)
                        try:
                            validated_data = response_format.model_validate(reparsed_dict)
                            print("Reparsing successful!")
                            content = validated_data.model_dump()
                            return content
                        
                        except Exception:
                            return reparsed_dict
                        
                    except Exception as reparse_err:
                        print(f"Reparse failed with error: {reparse_err}")
                        print(f"Reparsed response: {reparsed}")
                        return content
            
            return content

        except requests.HTTPError as err:
            print(f"Error generating text: {err}")
            return None

    def _reparse_response(self, response, system_content):

        user_prompt = prompts.AnswerSchemaFixPrompt.user_prompt.format(
            system_prompt=system_content,
            response=response
        )
        
        reparsed_response = self.send_message(
            system_content=prompts.AnswerSchemaFixPrompt.system_prompt,
            human_content=user_prompt,
            is_structured=False
        )
        
        return reparsed_response

     
class BaseGeminiProcessor:
    def __init__(self):
        self.llm = self._set_up_llm()
        self.default_model = 'gemini-2.0-flash-001'
        # self.default_model = "gemini-2.0-flash-thinking-exp-01-21",
        
    def _set_up_llm(self):
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=api_key)
        return genai

    def list_available_models(self) -> None:
        """
        Prints available Gemini models that support text generation.
        """
        print("Available models for text generation:")
        for model in self.llm.list_models():
            if "generateContent" in model.supported_generation_methods:
                print(f"- {model.name}")
                print(f"  Input token limit: {model.input_token_limit}")
                print(f"  Output token limit: {model.output_token_limit}")
                print()

    def _log_retry_attempt(retry_state):
        """Print information about the retry attempt"""
        exception = retry_state.outcome.exception()
        print(f"\nAPI Error encountered: {str(exception)}")
        print("Waiting 50 seconds before retry...\n")

    @retry(
        wait=wait_fixed(50),
        stop=stop_after_attempt(3),
        before_sleep=_log_retry_attempt,
    )
    def _generate_with_retry(self, model, human_content, generation_config):
        """Wrapper for generate_content with retry logic"""
        try:
            return model.generate_content(
                human_content,
                generation_config=generation_config
            )
        except Exception as e:
            if getattr(e, '_attempt_number', 0) == 3:
                print(f"\nRetry failed. Error: {str(e)}\n")
            raise

    def _parse_structured_response(self, response_text, response_format):
        try:
            repaired_json = repair_json(response_text)
            parsed_dict = json.loads(repaired_json)
            validated_data = response_format.model_validate(parsed_dict)
            return validated_data.model_dump()
        except Exception as err:
            print(f"Error parsing structured response: {err}")
            print("Attempting to reparse the response...")
            reparsed = self._reparse_response(response_text, response_format)
            return reparsed

    def _reparse_response(self, response, response_format):
        """Reparse invalid JSON responses using the model itself."""
        user_prompt = prompts.AnswerSchemaFixPrompt.user_prompt.format(
            system_prompt=prompts.AnswerSchemaFixPrompt.system_prompt,
            response=response
        )
        
        try:
            reparsed_response = self.send_message(
                model="gemini-2.0-flash-001",
                system_content=prompts.AnswerSchemaFixPrompt.system_prompt,
                human_content=user_prompt,
                is_structured=False
            )
            
            try:
                repaired_json = repair_json(reparsed_response)
                reparsed_dict = json.loads(repaired_json)
                try:
                    validated_data = response_format.model_validate(reparsed_dict)
                    print("Reparsing successful!")
                    return validated_data.model_dump()
                except Exception:
                    return reparsed_dict
            except Exception as reparse_err:
                print(f"Reparse failed with error: {reparse_err}")
                print(f"Reparsed response: {reparsed_response}")
                return response
        except Exception as e:
            print(f"Reparse attempt failed: {e}")
            return response

    def send_message(
        self,
        model=None,
        temperature: float = 0.5,
        seed=12345,  # For back compatibility
        system_content: str = "You are a helpful assistant.",
        human_content: str = "Hello!",
        is_structured: bool = False,
        response_format: Optional[Type[BaseModel]] = None,
    ) -> Union[str, Dict, None]:
        import sys
        print(f"[DEBUG] [Gemini] send_message 调用开始，model={model}")
        sys.stdout.flush()
        
        if model is None:
            model = self.default_model

        generation_config = {"temperature": temperature}
        
        prompt = f"{system_content}\n\n---\n\n{human_content}"
        print(f"[DEBUG] [Gemini] 准备调用 API，prompt 长度: {len(prompt)}")
        sys.stdout.flush()

        model_instance = self.llm.GenerativeModel(
            model_name=model,
            generation_config=generation_config
        )

        try:
            print(f"[DEBUG] [Gemini] 开始 API 调用...")
            sys.stdout.flush()
            response = self._generate_with_retry(model_instance, prompt, generation_config)
            print(f"[DEBUG] [Gemini] API 调用成功")
            sys.stdout.flush()

            self.response_data = {
                "model": response.model_version,
                "input_tokens": response.usage_metadata.prompt_token_count,
                "output_tokens": response.usage_metadata.candidates_token_count
            }
            print(f"[DEBUG] [Gemini] Response data: {self.response_data}")
            sys.stdout.flush()
            
            if is_structured and response_format is not None:
                return self._parse_structured_response(response.text, response_format)
            
            print(f"[DEBUG] [Gemini] 返回响应，长度: {len(response.text)}")
            sys.stdout.flush()
            return response.text
        except Exception as e:
            print(f"[ERROR] [Gemini] API request failed: {str(e)}")
            sys.stdout.flush()
            raise Exception(f"API request failed after retries: {str(e)}")


class APIProcessor:
    def __init__(self, provider: Literal["openai", "ibm", "gemini", "qwen"] ="openai"):
        self.provider = provider.lower()
        self.response_data = None  # 新增，保证所有APIProcessor实例都可安全访问
        if self.provider == "openai":
            self.processor = BaseOpenaiProcessor()
        elif self.provider == "ibm":
            self.processor = BaseIBMAPIProcessor()
        elif self.provider == "gemini":
            self.processor = BaseGeminiProcessor()
        elif self.provider == "qwen":
            self.processor = BaseQwenProcessor()

    def send_message(
        self,
        model=None,
        temperature=0.5,
        seed=None,
        system_content="You are a helpful assistant.",
        human_content="Hello!",
        is_structured=False,
        response_format=None,
        **kwargs
    ):
        """
        Routes the send_message call to the appropriate processor.
        The underlying processor's send_message method is responsible for handling the parameters.
        """
        if model is None:
            model = self.processor.default_model
        return self.processor.send_message(
            model=model,
            temperature=temperature,
            seed=seed,
            system_content=system_content,
            human_content=human_content,
            is_structured=is_structured,
            response_format=response_format,
            **kwargs
        )

    def get_answer_from_rag_context(self, question, rag_context, schema, model):
        system_prompt, response_format, user_prompt = self._build_rag_context_prompts(schema)
        answer_dict = self.processor.send_message(
            model=model,
            system_content=system_prompt,
            human_content=user_prompt.format(context=rag_context, question=question),
            is_structured=True,
            response_format=response_format
        )
        # 移除对 self.processor.response_data 的访问，避免 AttributeError
        return answer_dict


    def _build_rag_context_prompts(self, schema):
        """Return prompts tuple for the given schema."""
        use_schema_prompt = True if self.provider == "ibm" or self.provider == "gemini" else False
        
        if schema == "name":
            system_prompt = (prompts.AnswerWithRAGContextNamePrompt.system_prompt_with_schema 
                            if use_schema_prompt else prompts.AnswerWithRAGContextNamePrompt.system_prompt)
            response_format = prompts.AnswerWithRAGContextNamePrompt.AnswerSchema
            user_prompt = prompts.AnswerWithRAGContextNamePrompt.user_prompt
        elif schema == "number":
            system_prompt = (prompts.AnswerWithRAGContextNumberPrompt.system_prompt_with_schema
                            if use_schema_prompt else prompts.AnswerWithRAGContextNumberPrompt.system_prompt)
            response_format = prompts.AnswerWithRAGContextNumberPrompt.AnswerSchema
            user_prompt = prompts.AnswerWithRAGContextNumberPrompt.user_prompt
        elif schema == "boolean":
            system_prompt = (prompts.AnswerWithRAGContextBooleanPrompt.system_prompt_with_schema
                            if use_schema_prompt else prompts.AnswerWithRAGContextBooleanPrompt.system_prompt)
            response_format = prompts.AnswerWithRAGContextBooleanPrompt.AnswerSchema
            user_prompt = prompts.AnswerWithRAGContextBooleanPrompt.user_prompt
        elif schema == "names":
            system_prompt = (prompts.AnswerWithRAGContextNamesPrompt.system_prompt_with_schema
                            if use_schema_prompt else prompts.AnswerWithRAGContextNamesPrompt.system_prompt)
            response_format = prompts.AnswerWithRAGContextNamesPrompt.AnswerSchema
            user_prompt = prompts.AnswerWithRAGContextNamesPrompt.user_prompt
        elif schema == "comparative":
            system_prompt = (prompts.ComparativeAnswerPrompt.system_prompt_with_schema
                            if use_schema_prompt else prompts.ComparativeAnswerPrompt.system_prompt)
            response_format = prompts.ComparativeAnswerPrompt.AnswerSchema
            user_prompt = prompts.ComparativeAnswerPrompt.user_prompt
        elif schema == "jingpan":
            system_prompt = (prompts.AnswerWithRAGContextJingpanPrompt.system_prompt_with_schema
                            if use_schema_prompt else prompts.AnswerWithRAGContextJingpanPrompt.system_prompt)
            response_format = prompts.AnswerWithRAGContextJingpanPrompt.AnswerSchema
            user_prompt = prompts.AnswerWithRAGContextJingpanPrompt.user_prompt
        else:
            raise ValueError(f"Unsupported schema: {schema}")
        return system_prompt, response_format, user_prompt

    def get_rephrased_questions(self, original_question: str, companies: List[str] = None) -> Dict[str, str]:
        """将问题分解为子问题。如果指定了companies，尝试将子问题映射到公司。
        
        Args:
            original_question: 原始问题
            companies: 可选的公司列表，如果提供则尝试将子问题映射到各公司
            
        Returns:
            Dict[str, str]: 公司名（或子问题ID）到问题的映射
        """
        answer_dict = self.processor.send_message(
            system_content=prompts.RephrasedQuestionsPrompt.system_prompt,
            human_content=prompts.RephrasedQuestionsPrompt.user_prompt.format(
                question=original_question
            ),
            is_structured=True,
            response_format=prompts.RephrasedQuestionsPrompt.DecomposedQuestions
        )
        
        sub_questions = answer_dict["sub_questions"]
        
        # 如果指定了companies，尝试将子问题映射到公司
        if companies:
            questions_dict = {}
            for company in companies:
                # 找到包含该公司名的子问题
                matched = [sq["question"] for sq in sub_questions if company in sq["question"]]
                if matched:
                    questions_dict[company] = matched[0]
                else:
                    # 如果没有匹配到，使用第一个子问题（fallback）
                    questions_dict[company] = sub_questions[0]["question"] if sub_questions else original_question
            return questions_dict
        else:
            # 不指定companies时，返回所有子问题（用索引作为key）
            return {f"sub_q_{i}": sq["question"] for i, sq in enumerate(sub_questions)}


class AsyncOpenaiProcessor:
    
    def _get_unique_filepath(self, base_filepath):
        """Helper method to get unique filepath"""
        if not os.path.exists(base_filepath):
            return base_filepath
        
        base, ext = os.path.splitext(base_filepath)
        counter = 1
        while os.path.exists(f"{base}_{counter}{ext}"):
            counter += 1
        return f"{base}_{counter}{ext}"

    async def process_structured_ouputs_requests(
        self,
        model="gpt-4o-mini-2024-07-18",
        temperature=0.5,
        seed=None,
        system_content="You are a helpful assistant.",
        queries=None,
        response_format=None,
        requests_filepath='./temp_async_llm_requests.jsonl',
        save_filepath='./temp_async_llm_results.jsonl',
        preserve_requests=False,
        preserve_results=True,
        request_url="https://api.openai.com/v1/chat/completions",
        max_requests_per_minute=3_500,
        max_tokens_per_minute=3_500_000,
        token_encoding_name="o200k_base",
        max_attempts=5,
        logging_level=20,
        progress_callback=None
    ):
        # Create requests for jsonl
        jsonl_requests = []
        for idx, query in enumerate(queries):
            request = {
                "model": model,
                "temperature": temperature,
                "seed": seed,
                "messages": [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": query},
                ],
                'response_format': type_to_response_format_param(response_format),
                'metadata': {'original_index': idx}
            }
            jsonl_requests.append(request)
            
        # Get unique filepaths if files already exist
        requests_filepath = self._get_unique_filepath(requests_filepath)
        save_filepath = self._get_unique_filepath(save_filepath)

        # Write requests to JSONL file
        with open(requests_filepath, "w") as f:
            for request in jsonl_requests:
                json_string = json.dumps(request)
                f.write(json_string + "\n")

        # Process API requests
        total_requests = len(jsonl_requests)

        async def monitor_progress():
            last_count = 0
            while True:
                try:
                    with open(save_filepath, 'r') as f:
                        current_count = sum(1 for _ in f)
                        if current_count > last_count:
                            if progress_callback:
                                for _ in range(current_count - last_count):
                                    progress_callback()
                            last_count = current_count
                        if current_count >= total_requests:
                            break
                except FileNotFoundError:
                    pass
                await asyncio.sleep(0.1)

        async def process_with_progress():
            await asyncio.gather(
                process_api_requests_from_file(
                    requests_filepath=requests_filepath,
                    save_filepath=save_filepath,
                    request_url=request_url,
                    api_key=os.getenv("OPENAI_API_KEY"),
                    max_requests_per_minute=max_requests_per_minute,
                    max_tokens_per_minute=max_tokens_per_minute,
                    token_encoding_name=token_encoding_name,
                    max_attempts=max_attempts,
                    logging_level=logging_level
                ),
                monitor_progress()
            )

        await process_with_progress()

        with open(save_filepath, "r") as f:
            validated_data_list = []
            results = []
            for line_number, line in enumerate(f, start=1):
                raw_line = line.strip()
                try:
                    result = json.loads(raw_line)
                except json.JSONDecodeError as e:
                    print(f"[ERROR] Line {line_number}: Failed to load JSON from line: {raw_line}")
                    continue

                # Check finish_reason in the API response
                finish_reason = result[1]['choices'][0].get('finish_reason', '')
                if finish_reason != "stop":
                    print(f"[WARNING] Line {line_number}: finish_reason is '{finish_reason}' (expected 'stop').")

                # Safely parse answer; if it fails, leave answer empty and report the error.
                try:
                    answer_content = result[1]['choices'][0]['message']['content']
                    answer_parsed = json.loads(answer_content)
                    answer = response_format(**answer_parsed).model_dump()
                except Exception as e:
                    print(f"[ERROR] Line {line_number}: Failed to parse answer JSON. Error: {e}.")
                    answer = ""

                results.append({
                    'index': result[2],
                    'question': result[0]['messages'],
                    'answer': answer
                })
            
            # Sort by original index and build final list
            validated_data_list = [
                {'question': r['question'], 'answer': r['answer']} 
                for r in sorted(results, key=lambda x: x['index']['original_index'])
            ]

        if not preserve_requests:
            os.remove(requests_filepath)

        if not preserve_results:
            os.remove(save_filepath)
        else:  # Fix requests order
            with open(save_filepath, "r") as f:
                results = [json.loads(line) for line in f]
            
            sorted_results = sorted(results, key=lambda x: x[2]['original_index'])
            
            with open(save_filepath, "w") as f:
                for result in sorted_results:
                    json_string = json.dumps(result)
                    f.write(json_string + "\n")
            
        return validated_data_list
