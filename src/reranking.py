import os
from dotenv import load_dotenv
from openai import OpenAI
import requests
import src.prompts as prompts
from concurrent.futures import ThreadPoolExecutor
from tenacity import retry, wait_fixed, stop_after_attempt


class JinaReranker:
    def __init__(self):
        self.url = 'https://api.jina.ai/v1/rerank'
        self.headers = self.get_headers()
        
    def get_headers(self):
        load_dotenv()
        jina_api_key = os.getenv("JINA_API_KEY")    
        headers = {'Content-Type': 'application/json',
                   'Authorization': f'Bearer {jina_api_key}'}
        return headers
    
    def rerank(self, query, documents, top_n = 10):
        data = {
            "model": "jina-reranker-v2-base-multilingual",
            "query": query,
            "top_n": top_n,
            "documents": documents
        }

        response = requests.post(url=self.url, headers=self.headers, json=data)

        return response.json()


def _log_retry_attempt(retry_state):
    exception = retry_state.outcome.exception()
    print(f"\nAPI Error encountered: {str(exception)}")
    print("Waiting 50 seconds before retry...\n")


class LLMReranker:
    def __init__(self, provider: str = None, model: str = None, max_concurrent_requests: int = 10):
        load_dotenv()
        self.provider = provider or os.getenv("LLM_RERANK_PROVIDER", "qwen")
        print(f"[LLMReranker] Using provider: {self.provider}")
        #self.model = model or os.getenv("LLM_RERANK_MODEL", "qwen-turbo")
        self.model = model or os.getenv("LLM_RERANK_MODEL", "qwen-max-latest")
        print(f"[LLMReranker] Using model: {self.model}")
        # 并发限制：阿里云 Dashscope API 通常限制在 10-20 QPS
        self.max_concurrent_requests = max_concurrent_requests
        print(f"[LLMReranker] Max concurrent requests: {self.max_concurrent_requests}")
        self.qwen_api_key = os.getenv("QWEN_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.system_prompt_rerank_single_block = prompts.RerankingPrompt.system_prompt_rerank_single_block
        self.system_prompt_rerank_multiple_blocks = prompts.RerankingPrompt.system_prompt_rerank_multiple_blocks
        self.schema_for_single_block = prompts.RetrievalRankingSingleBlock
        self.schema_for_multiple_blocks = prompts.RetrievalRankingMultipleBlocks
        if self.provider == "openai":
            from openai import OpenAI
            self.llm = OpenAI(api_key=self.openai_api_key)
        else:
            self.llm = None  # Qwen uses HTTP API

    @retry(wait=wait_fixed(50), stop=stop_after_attempt(3), before_sleep=_log_retry_attempt)
    def _qwen_send(self, system_content, user_content, response_format=None):
        import os
        from dashscope import Generation
        import dashscope
        dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'

        api_key = os.getenv("DASHSCOPE_API_KEY") or self.qwen_api_key
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]
        import sys
        print(f"[DEBUG] [Reranker] 调用 Dashscope API...")
        sys.stdout.flush()
        try:
            response = Generation.call(
                api_key=api_key,
                model=self.model,
                messages=messages,
                result_format="message",
            )
            print(f"[DEBUG] [Reranker] API 调用完成")
            sys.stdout.flush()
        except Exception as e:
            print(f"[ERROR] dashscope Generation.call exception: {e}")
            sys.stdout.flush()
            raise  # 让retry捕获异常并重试

        if hasattr(response, 'status_code') and response.status_code == 200:
            #print("[DEBUG] Qwen API call response is:", response)
            content = response.output.choices[0].message.content
            if response_format is not None:
                from json_repair import repair_json
                import json
                try:
                    repaired_json = repair_json(content)
                    return json.loads(repaired_json)
                except Exception as e:
                    print(f"[ERROR] json_repair failed: {e}, content: {content}")
                    return {"block_rankings": []}
            else:
                return content
        else:
            print(f"HTTP return code: {getattr(response, 'status_code', None)}")
            print(f"Error code: {getattr(response, 'code', None)}")
            print(f"Error message: {getattr(response, 'message', None)}")
            print("For more information, see: https://www.alibabacloud.com/help/en/model-studio/error-code")
            # 如果是限流错误则抛出异常，触发retry
            if getattr(response, 'status_code', None) == 429 or getattr(response, 'code', None) == 'Throttling.AllocationQuota':
                raise Exception(f"Qwen API throttling: {getattr(response, 'message', None)}")
            return {"block_rankings": []}



    def get_rank_for_single_block(self, query, retrieved_document):
        user_prompt = f'\nHere is the query:\n"{query}"\n\nHere is the retrieved text block:\n"""\n{retrieved_document}\n"""\n'
        if self.provider == "openai":
            completion = self.llm.beta.chat.completions.parse(
                model=self.model,
                temperature=0,
                messages=[
                    {"role": "system", "content": self.system_prompt_rerank_single_block},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=self.schema_for_single_block
            )
            response = completion.choices[0].message.parsed
            response_dict = response.model_dump()
            return response_dict
        else:
            # Qwen
            content = self._qwen_send(
                self.system_prompt_rerank_single_block,
                user_prompt,
                is_structured=True,
                response_format=self.schema_for_single_block
            )
            return content

    # reranker 调用示例
    from tenacity import retry, wait_fixed, stop_after_attempt
    @retry(wait=wait_fixed(3), stop=stop_after_attempt(2), before_sleep=_log_retry_attempt)
    def get_rank_for_multiple_blocks(self, query, retrieved_documents):
        formatted_blocks = "\n\n---\n\n".join([
            f'Block {i+1}:\n\n"""\n{text}\n"""' 
            for i, text in enumerate(retrieved_documents)
        ])
        user_prompt = (
            f"Here is the query: \"{query}\"\n\n"
            "Here are the retrieved text blocks:\n"
            f"{formatted_blocks}\n\n"
            f"You should provide exactly {len(retrieved_documents)} rankings, in order and in json format like this:"
            """
            {
  "block_rankings": [
    {
      "block_id": 1,
      "relevance_score": 0.95,
      "reasoning": "Directly explains the origin of 火山碎屑岩."
    },
    {
      "block_id": 2,
      "relevance_score": 0.3,
      "reasoning": "Discusses 砂岩, which is a different type of rock."
    },
    {
      "block_id": 3,
      "relevance_score": 0.1,
      "reasoning": "Unrelated to 火山碎屑岩 formation."
    }
  ]
}
            """
        )

        # 调用 Qwen structured 输出
        response = self._qwen_send(
            system_content=self.system_prompt_rerank_multiple_blocks,
            user_content=user_prompt,
            response_format=self.schema_for_multiple_blocks
        )
        # 保证返回值为dict，否则兜底
        if not isinstance(response, dict):
            print(f"[ERROR] get_rank_for_multiple_blocks: Qwen返回非dict: {response}")
            print("######")
            return {"block_rankings": []}
        block_rankings = response.get('block_rankings', [])
        if len(block_rankings) < len(retrieved_documents):
            print("[DEBUG] ###########")
            print(f"         Warning: Expected {len(retrieved_documents)} rankings but got {len(block_rankings)}")
            for i in range(len(block_rankings), len(retrieved_documents)):
                doc = retrieved_documents[i]
                preview = doc[:70].replace('\n', ' ').replace('\r', ' ') if isinstance(doc, str) else str(doc)
                print(f"         Missing ranking for document idx {i}:")
                print(f"         Text preview: {preview}...")
            print("[DEBUG] ###########")
            # 抛出异常让retry重试
            raise Exception("block_rankings length mismatch, will retry.")
        return response

    def rerank_documents(self, query: str, documents: list, documents_batch_size: int = 4, llm_weight: float = 0.7, progress_callback=None):
        """
        Rerank multiple documents using parallel processing with controlled concurrency.
        Combines vector similarity and LLM relevance scores using weighted average.
        
        Args:
            query: Search query
            documents: List of documents to rerank
            documents_batch_size: Number of documents per batch (default: 4)
            llm_weight: Weight for LLM score (default: 0.7)
            progress_callback: Optional callback for progress updates
        """
        # Create batches of documents
        doc_batches = [documents[i:i + documents_batch_size] for i in range(0, len(documents), documents_batch_size)]
        vector_weight = 1 - llm_weight
        
        print(f"[LLMReranker] Processing {len(documents)} documents in {len(doc_batches)} batches with max {self.max_concurrent_requests} concurrent requests")
        
        if documents_batch_size == 1:
            def process_single_doc(doc):
                # Add source information to text for reranking context
                source_sha1 = doc.get('source_sha1', 'Unknown')
                text_with_source = f"[来源: {source_sha1}]\n{doc['text']}"
                
                # Get ranking for single document
                ranking = self.get_rank_for_single_block(query, text_with_source)
                
                doc_with_score = doc.copy()
                doc_with_score["relevance_score"] = ranking["relevance_score"]
                # Calculate combined score - note that distance is inverted since lower is better
                doc_with_score["combined_score"] = round(
                    llm_weight * ranking["relevance_score"] + 
                    vector_weight * doc['distance'],
                    4
                )
                return doc_with_score

            # Process all documents in parallel using single-block method
            # 使用受控的并发数量
            with ThreadPoolExecutor(max_workers=self.max_concurrent_requests) as executor:
                all_results = list(executor.map(process_single_doc, documents))
                
        else:
            def process_batch(batch_idx_tuple):
                """Process a batch and update progress"""
                batch_idx, batch = batch_idx_tuple
                
                # 更新进度
                if progress_callback:
                    progress_percentage = 60 + int((batch_idx / len(doc_batches)) * 10)  # 60-70%
                    progress_callback(f"🎯 重排序中 ({batch_idx + 1}/{len(doc_batches)} 批次)...", progress_percentage)
                
                # Add source information to texts for reranking context
                texts = [f"[来源: {doc.get('source_sha1', 'Unknown')}]\n{doc['text']}" for doc in batch]
                try:
                    rankings = self.get_rank_for_multiple_blocks(query, texts)
                except Exception as e:
                    # 两次重试后依然失败，补全缺失项
                    print("[DEBUG] ###########")
                    print(f"         [Final] Warning: Expected {len(batch)} rankings but got less. Filling defaults.")
                    rankings = {"block_rankings": []}
                results = []
                block_rankings = rankings.get('block_rankings', [])
                if len(block_rankings) < len(batch):
                    for _ in range(len(batch) - len(block_rankings)):
                        block_rankings.append({
                            "relevance_score": 0.0,
                            "reasoning": "Default ranking due to missing LLM response"
                        })
                for doc, rank in zip(batch, block_rankings):
                    doc_with_score = doc.copy()
                    # 健壮性处理，缺失字段补默认值
                    doc_with_score["relevance_score"] = rank.get("relevance_score", 0.0)
                    doc_with_score["reasoning"] = rank.get("reasoning", "No reasoning provided.")
                    doc_with_score["combined_score"] = round(
                        llm_weight * doc_with_score["relevance_score"] + 
                        vector_weight * doc['distance'],
                        4
                    )
                    results.append(doc_with_score)
                return results

            # Process batches in parallel with controlled concurrency
            # 使用受控的并发数量：max_workers 限制同时运行的线程数
            with ThreadPoolExecutor(max_workers=self.max_concurrent_requests) as executor:
                # 为每个 batch 添加索引以便追踪进度
                batch_results = list(executor.map(process_batch, enumerate(doc_batches)))
            
            # Flatten results
            all_results = []
            for batch in batch_results:
                all_results.extend(batch)
        
        # Sort results by combined score in descending order
        all_results.sort(key=lambda x: x["combined_score"], reverse=True)
        return all_results
