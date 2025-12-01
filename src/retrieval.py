import json
import logging
from typing import List, Tuple, Dict, Union, Optional
from rank_bm25 import BM25Okapi
import pickle
from pathlib import Path
import faiss
from src.api_requests import BaseQwenProcessor,BaseGeminiProcessor
from dotenv import load_dotenv
import os
import re
import numpy as np
from src.reranking import LLMReranker
from src.financial_glossary import (
    find_financial_concepts,
    format_concepts_for_prompt,
)
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

_log = logging.getLogger(__name__)


def extract_years_from_question(question: str, expand_window: bool = True) -> List[int]:
    """
    从问题中提取年份信息，并可选地扩展时间窗口
    支持格式：
    - 明确年份: "2025年", "2023年第一季度"
    - 日期格式: "2025年9月30日"
    - 多年份比较: "2024年相比2023年" → [2023, 2024]
    
    Args:
        question: 用户问题
        expand_window: 是否扩展时间窗口（在年份范围前后各加1年）
    
    Returns:
        List[int]: 提取到的年份列表（去重并排序）
        
    示例：
        问"2024年xxx" + expand_window=True → [2023, 2024, 2025]
        问"2024年相比2023年" + expand_window=True → [2022, 2023, 2024, 2025] （范围扩展而非逐个扩展）
        问"2024年xxx" + expand_window=False → [2024]
    """
    # 正则匹配 20XX年 格式
    year_pattern = r'(20\d{2})年'
    matches = re.findall(year_pattern, question)
    extracted_years = [int(y) for y in matches]
    
    if not extracted_years:
        return []
    
    if expand_window:
        # 找到年份范围的最小和最大值
        min_year = min(extracted_years)
        max_year = max(extracted_years)
        
        # 在范围前后各扩展1年，生成连续年份列表
        years = list(range(min_year - 1, max_year + 2))  # +2 因为 range 不包含结束值
        
        print(f"[DEBUG] 📅 提取年份: {sorted(set(extracted_years))} → 扩展范围: [{min_year-1}, {max_year+1}]")
    else:
        years = extracted_years
    
    return sorted(list(set(years)))  # 去重并排序


def route_reports_by_time(
    company_name: str, 
    question: str, 
    all_reports: List[Dict],
    fallback_strategy: str = "all",  # "all" 或 "latest"
    selected_years: List[int] = None  # 可选：前端指定的年份列表
) -> List[Dict]:
    """
    基于公司名和可选年份信息路由到合适的文档
    
    Args:
        company_name: 公司名称
        question: 用户问题（不再用于提取年份）
        all_reports: 所有可用的报告
        fallback_strategy: 当没有指定年份时的回退策略
            - "all": 返回该公司所有文档（默认）
            - "latest": 只返回最新年份的文档
        selected_years: 可选，前端指定的年份列表。如果提供，只返回这些年份的文档
    
    Returns:
        List[Dict]: 匹配的报告列表
    """
    # 1. 先按公司名过滤
    company_reports = []
    for report in all_reports:
        document = report.get("document", {})
        metainfo = document.get("metainfo", {})
        if metainfo.get("company_name") == company_name:
            company_reports.append(report)
    
    if not company_reports:
        return []
    
    # 2. 如果指定了年份，按年份过滤
    if selected_years and len(selected_years) > 0:
        filtered_reports = []
        for report in company_reports:
            document = report.get("document", {})
            metainfo = document.get("metainfo", {})
            
            # 优先从 year 字段获取，否则从 sha1_name 中提取（如 "J2025" → 2025）
            report_year = metainfo.get("year")
            if report_year is None:
                sha1_name = metainfo.get("sha1_name", "")
                # 从 sha1_name 中提取年份：匹配 J20XX 或 20XX 格式
                year_match = re.search(r'[J]?(20\d{2})', sha1_name)
                if year_match:
                    report_year = int(year_match.group(1))
            
            # 支持字符串或整数格式的 year
            if report_year is not None:
                try:
                    report_year = int(report_year)
                    if report_year in selected_years:
                        filtered_reports.append(report)
                except (ValueError, TypeError):
                    pass
        
        if filtered_reports:
            print(f"[INFO] 🎯 年份过滤: 选择年份 {selected_years}，匹配到 {len(filtered_reports)} 个文档")
            return filtered_reports
        else:
            print(f"[WARNING] ⚠️ 指定年份 {selected_years}，但未找到对应文档，回退到全部文档")
    
    # 3. 没有指定年份时的回退策略
    if fallback_strategy == "latest":
        # 返回最新年份的文档
        latest_year = None
        latest_reports = []
        for report in company_reports:
            document = report.get("document", {})
            metainfo = document.get("metainfo", {})
            
            # 优先从 year 字段获取，否则从 sha1_name 中提取
            report_year = metainfo.get("year")
            if report_year is None:
                sha1_name = metainfo.get("sha1_name", "")
                year_match = re.search(r'[J]?(20\d{2})', sha1_name)
                if year_match:
                    report_year = int(year_match.group(1))
            
            if report_year is not None:
                try:
                    report_year = int(report_year)
                    if latest_year is None or report_year > latest_year:
                        latest_year = report_year
                        latest_reports = [report]
                    elif report_year == latest_year:
                        latest_reports.append(report)
                except (ValueError, TypeError):
                    pass
        
        if latest_reports:
            print(f"[INFO] 📅 无指定年份，使用最新年份 {latest_year} 的文档")
            return latest_reports
    
    # 4. 默认返回所有该公司的文档（不再根据问题中的年份过滤）
    print(f"[INFO] 📚 使用该公司所有 {len(company_reports)} 个文档（所有年份）")
    return company_reports

class BM25Retriever:
    def __init__(self, bm25_db_dir: Path, documents_dir: Path, subset_path: Path = None):
        self.bm25_db_dir = bm25_db_dir
        self.documents_dir = documents_dir
        self.subset_path = subset_path
        self.year_lookup = self._load_year_lookup() if subset_path else {}
    
    def _load_year_lookup(self) -> dict:
        """从 subset.csv 加载 sha1 -> year 的映射"""
        import csv
        year_lookup = {}
        try:
            with open(self.subset_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    sha1 = row.get('sha1', '').strip()
                    year = row.get('year', '').strip()
                    if sha1 and year:
                        try:
                            year_lookup[sha1] = int(year)
                        except ValueError:
                            pass
            print(f"[INFO] 📅 BM25: 从 subset.csv 加载了 {len(year_lookup)} 个文档的年份信息")
        except Exception as e:
            print(f"[WARNING] ⚠️ BM25: 无法加载 subset.csv 年份信息: {e}")
        return year_lookup
        
    def retrieve_by_company_name(self, company_name: str, query: str, top_n: int = 3, return_parent_pages: bool = False, selected_years: List[int] = None) -> List[Dict]:
        print("BM25Retriever retrieve_by_company_name is called")
        
        # 🎯 先收集所有文档，然后使用路由函数
        all_documents = []
        for path in self.documents_dir.glob("*.json"):
            with open(path, 'r', encoding='utf-8') as f:
                doc = json.load(f)
                all_documents.append({
                    "path": path,
                    "document": doc,
                    "sha1": doc["metainfo"]["sha1_name"]
                })
        
        # 转换为 route_reports_by_time 需要的格式
        all_reports = []
        for doc_info in all_documents:
            all_reports.append({
                "document": doc_info["document"],
                "name": doc_info["sha1"]
            })
        
        # 使用路由函数过滤文档（默认在所有年份中检索，除非指定了 selected_years）
        matching_reports = route_reports_by_time(
            company_name=company_name,
            question=query,
            all_reports=all_reports,
            fallback_strategy="all",
            selected_years=selected_years
        )
        
        # 转换回原来的格式
        matching_documents = []
        matching_sha1s = {rep["name"] for rep in matching_reports}
        for doc_info in all_documents:
            if doc_info["sha1"] in matching_sha1s:
                matching_documents.append(doc_info)
        
        if not matching_documents:
            raise ValueError(f"No report found with '{company_name}' company name.")
        
        if selected_years and len(selected_years) > 0:
            print(f"[INFO] 🎯 BM25年份过滤: 选择年份 {selected_years}，匹配到 {len(matching_documents)} 个文档")
        elif len(matching_documents) > 1:
            print(f"[INFO] Found {len(matching_documents)} reports for '{company_name}', retrieving from all")
            
        # Retrieve from all matching documents and aggregate results
        all_retrieval_results = []
        
        for doc_info in matching_documents:
            document = doc_info["document"]
            sha1 = doc_info["sha1"]
            
            # Load corresponding BM25 index
            bm25_path = self.bm25_db_dir / f"{sha1}.pkl"
            if not bm25_path.exists():
                print(f"[WARNING] BM25 index not found for {sha1}, skipping")
                continue
                
            with open(bm25_path, 'rb') as f:
                bm25_index = pickle.load(f)
                
            # Get the document content and BM25 index
            chunks = document["content"]["chunks"]
            pages = document["content"]["pages"]
            
            # Get BM25 scores for the query
            tokenized_query = query.split()
            scores = bm25_index.get_scores(tokenized_query)
            
            actual_top_n = min(top_n, len(scores))
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:actual_top_n]
            
            seen_pages = set()
            
            for index in top_indices:
                score = round(float(scores[index]), 4)
                chunk = chunks[index]
                parent_page = next(page for page in pages if page["page"] == chunk["page"])
                
                if return_parent_pages:
                    if parent_page["page"] not in seen_pages:
                        seen_pages.add(parent_page["page"])
                        result = {
                            "distance": score,
                            "page": parent_page["page"],
                            "text": parent_page["text"],
                            "source_sha1": sha1  # Add source document identifier
                        }
                        all_retrieval_results.append(result)
                else:
                    result = {
                        "distance": score,
                        "page": chunk["page"],
                        "text": chunk["text"],
                        "source_sha1": sha1  # Add source document identifier
                    }
                    all_retrieval_results.append(result)
        
        # Sort by score and return top_n results across all documents
        all_retrieval_results.sort(key=lambda x: x["distance"], reverse=True)
        return all_retrieval_results[:top_n]

class HybridRetriever:
    def __init__(
        self,
        vector_db_dir: Path,
        documents_dir: Path,
        use_hyde: bool = True,
        use_multi_query: bool = True,
        subset_path: Path = None,
        parallel_workers: int = 4,
        multi_query_methods: Optional[Dict[str, bool]] = None,
    ):
        self.vector_retriever = VectorRetriever(
            vector_db_dir,
            documents_dir,
            use_hyde=use_hyde,
            use_multi_query=use_multi_query,
            subset_path=subset_path,
            parallel_workers=parallel_workers,
            multi_query_methods=multi_query_methods,
        )
        self.reranker = LLMReranker()
        
    def retrieve_by_company_name(
        self, 
        company_name: str, 
        query: str, 
        llm_reranking_sample_size: int = 28,
        documents_batch_size: int = 2,
        top_n: int = 6,
        llm_weight: float = 0.7,
        return_parent_pages: bool = False,
        use_hyde: bool = None,
        use_multi_query: bool = None,
        progress_callback=None,
        selected_years: List[int] = None,
        multi_query_config: Optional[Dict[str, bool]] = None
    ) -> List[Dict]:
        """
        Retrieve and rerank documents using hybrid approach.
        
        Args:
            company_name: Name of the company to search documents for
            query: Search query
            llm_reranking_sample_size: Number of initial results to retrieve from vector DB
            documents_batch_size: Number of documents to analyze in one LLM prompt
            top_n: Number of final results to return after reranking
            llm_weight: Weight given to LLM scores (0-1)
            return_parent_pages: Whether to return full pages instead of chunks
            selected_years: Optional list of years to filter documents
            
        Returns:
            List of reranked document dictionaries with scores
        """
        import time
        
        timing_info = {
            'hyde_expansion': 0.0,
            'multi_query_expansion': 0.0,
            'vector_search': 0.0,
            'llm_reranking': 0.0
        }
        
        # Get initial results from vector retriever
        vector_retrieval_result = self.vector_retriever.retrieve_by_company_name(
            company_name=company_name,
            query=query,
            top_n=llm_reranking_sample_size,
            return_parent_pages=return_parent_pages,
            use_hyde=use_hyde,
            use_multi_query=use_multi_query,
            progress_callback=progress_callback,
            selected_years=selected_years,
            multi_query_config=multi_query_config
        )
        
        # 处理返回结果（可能是dict或list）
        expansion_texts = {}
        if isinstance(vector_retrieval_result, dict) and 'timing' in vector_retrieval_result:
            timing_info.update(vector_retrieval_result['timing'])
            vector_results = vector_retrieval_result['results']
            # 提取扩展文本信息
            if 'expansion_texts' in vector_retrieval_result:
                expansion_texts = vector_retrieval_result['expansion_texts']
        else:
            vector_results = vector_retrieval_result
        
        print(f"[DEBUG] Initial vector results count: {len(vector_results)}")

        # 重排序阶段（这是最耗时的部分）
        if progress_callback:
            progress_callback("🎯 LLM 重排序中（这可能需要一些时间）...", 58)
        
        # Rerank results using LLM
        rerank_start = time.time()
        reranked_results = self.reranker.rerank_documents(
            query=query,
            documents=vector_results,
            documents_batch_size=documents_batch_size,
            llm_weight=llm_weight
        )
        timing_info['llm_reranking'] = time.time() - rerank_start

        print(f"[DEBUG] Reranked results count: {len(reranked_results)}")
        #print("[DEBUG] HybridRetriever retrieve_by_company_name is called")
        print(f"[DEBUG] Final top_n: {top_n}")
        
        final_results = reranked_results[:top_n]
        
        # 返回结果、时间信息和扩展文本
        return {
            'results': final_results,
            'timing': timing_info,
            'expansion_texts': expansion_texts,
            'reranker_stats': self.reranker.get_stats()
        }

class VectorRetriever:
    def __init__(
        self,
        vector_db_dir: Path,
        documents_dir: Path,
        use_hyde: bool = True,
        use_multi_query: bool = True,
        subset_path: Path = None,
        parallel_workers: int = 4,
        multi_query_methods: Optional[Dict[str, bool]] = None,
    ):
        self.vector_db_dir = vector_db_dir
        self.documents_dir = documents_dir
        self.subset_path = subset_path
        self.year_lookup = self._load_year_lookup() if subset_path else {}
        self.all_dbs = self._load_dbs()
        self.qwen = BaseQwenProcessor()
        self.use_hyde = use_hyde
        self.use_multi_query = use_multi_query
        self.parallel_workers = max(1, parallel_workers)
        self.multi_query_methods = multi_query_methods or {
            'synonym': True,
            'subquestion': True,
            'variant': True
        }
        #print(f"[DEBUG][VectorRetriever.__init__] use_hyde={self.use_hyde}, use_multi_query={self.use_multi_query}")
    
    def _load_year_lookup(self) -> dict:
        """从 subset.csv 加载 sha1 -> year 的映射"""
        import csv
        year_lookup = {}
        try:
            with open(self.subset_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    sha1 = row.get('sha1', '').strip()
                    year = row.get('year', '').strip()
                    if sha1 and year:
                        try:
                            year_lookup[sha1] = int(year)
                        except ValueError:
                            pass
            print(f"[INFO] 📅 从 subset.csv 加载了 {len(year_lookup)} 个文档的年份信息")
        except Exception as e:
            print(f"[WARNING] ⚠️ 无法加载 subset.csv 年份信息: {e}")
        return year_lookup

    # Qwen embedding 不需要 set_up_llm
    
    # Qwen embedding 不需要 set_up_llm

    def _load_dbs(self):
        all_dbs = []
        company_names = []  # 用于收集company_name
        # Get list of JSON document paths
        all_documents_paths = list(self.documents_dir.glob('*.json'))
        vector_db_files = {db_path.stem: db_path for db_path in self.vector_db_dir.glob('*.faiss')}

        for document_path in all_documents_paths:
            #print(f"[DEBUG] Loading document: {document_path.name}")
            stem = document_path.stem
            if stem not in vector_db_files:
                _log.warning(f"No matching vector DB found for document {document_path.name}")
                continue
            try:
                with open(document_path, 'r', encoding='utf-8') as f:
                    document = json.load(f)
            except Exception as e:
                _log.error(f"Error loading JSON from {document_path.name}: {e}")
                continue

            # Validate that the document meets the expected schema
            if not (isinstance(document, dict) and "metainfo" in document and "content" in document):
                _log.warning(f"Skipping {document_path.name}: does not match the expected schema.")
                continue

            # 收集company_name
            company_name = document.get("metainfo", {}).get("company_name", None)
            if company_name:
                company_names.append(company_name)
            
            # 🆕 从 year_lookup 注入 year 信息到 metainfo
            sha1_name = document.get("metainfo", {}).get("sha1_name", stem)
            if sha1_name in self.year_lookup:
                document["metainfo"]["year"] = self.year_lookup[sha1_name]

            try:
                vector_db = faiss.read_index(str(vector_db_files[stem]))
            except Exception as e:
                _log.error(f"Error reading vector DB for {document_path.name}: {e}")
                continue

            report = {
                "name": stem,
                "vector_db": vector_db,
                "document": document
            }
            all_dbs.append(report)

        # print("[DEBUG] 当前可用的company_name有:")
        # for name in company_names:
        #     print(f"  - {name}")
        # print("[DEBUG] 当前可用的company_name以上")

        return all_dbs

    @staticmethod
    def get_strings_cosine_similarity(str1, str2):
        qwen = BaseQwenProcessor()
        emb1 = qwen.get_embeddings([str1])["embeddings"][0]
        emb2 = qwen.get_embeddings([str2])["embeddings"][0]
        similarity_score = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        similarity_score = round(similarity_score, 4)
        return similarity_score

    
   
    def _safe_flush(self):
        """安全地刷新标准输出，忽略 BrokenPipeError"""
        import sys
        try:
            sys.stdout.flush()
        except (BrokenPipeError, OSError):
            pass  # 忽略 BrokenPipeError，在 Streamlit 环境中可能发生
    
    def _safe_print(self, *args, **kwargs):
        """安全地打印，忽略 BrokenPipeError"""
        try:
            print(*args, **kwargs)
            self._safe_flush()
        except (BrokenPipeError, OSError):
            pass  # 忽略 BrokenPipeError，在 Streamlit 环境中可能发生
    
    def retrieve_by_company_name(self, company_name: str, query: str, llm_reranking_sample_size: int = None, top_n: int = 3, return_parent_pages: bool = False, use_hyde: bool = None, use_multi_query: bool = None, progress_callback=None, selected_years: List[int] = None, multi_query_config: Optional[Dict[str, bool]] = None) -> List[Tuple[str, float]]:
        import sys
        import time
        
        # 初始化时间统计和扩展文本信息
        timing_info = {
            'hyde_expansion': 0.0,
            'multi_query_expansion': 0.0,
            'embedding_generation': 0.0,
            'vector_search': 0.0
        }
        
        # 保存扩展生成的文本
        expansion_texts = {
            'hyde_text': None,
            'multi_query_texts': [],
            'glossary_context': None,
            'multi_query_methods': {}
        }
        
        self._safe_print("[DEBUG] VectorRetriever retrieve_by_company_name is called")

        # 🎯 使用路由函数定位文档（默认在所有年份中检索，除非指定了 selected_years）
        if progress_callback:
            progress_callback("📚 定位相关文档中...", 28)
        
        matching_reports = route_reports_by_time(
            company_name=company_name,
            question=query,
            all_reports=self.all_dbs,
            fallback_strategy="all",  # 无指定年份时使用所有文档
            selected_years=selected_years  # 前端指定的年份列表
        )
        
        if not matching_reports:
            _log.error(f"No report found with '{company_name}' company name.")
            raise ValueError(f"No report found with '{company_name}' company name.")
        
        if len(matching_reports) > 1:
            self._safe_print(f"[INFO] Found {len(matching_reports)} reports for '{company_name}', retrieving from all")
            for rep in matching_reports:
                doc = rep.get("document", {})
                metainfo = doc.get("metainfo", {})
                year = metainfo.get("year", "unknown")
                self._safe_print(f"  - Report: {rep['name']} (Year: {year})")
        
        # Priority parameters
        use_hyde = self.use_hyde if use_hyde is None else use_hyde
        use_multi_query = self.use_multi_query if use_multi_query is None else use_multi_query
        multi_query_config = multi_query_config or self.multi_query_methods or {}
        expansion_texts['multi_query_methods'] = multi_query_config
        self._safe_print(f"[DEBUG] multi_query_config = {multi_query_config}")
        if use_multi_query and not any(multi_query_config.values()):
            self._safe_print("[INFO] Multi-Query enabled but no methods selected; skipping expansion.")
            use_multi_query = False
        self._safe_print(f"[DEBUG][retrieve_by_company_name] use_hyde={use_hyde}, use_multi_query={use_multi_query}")
        
        qwen = BaseQwenProcessor()
        # 控制multi_query和hyde扩充
        queries = [query]

        if use_hyde:
            if progress_callback:
                progress_callback("🔮 HYDE 查询扩展中...", 32)
            self._safe_print(f"[DEBUG] 开始 HYDE 扩展...")
            hyde_start = time.time()
            try:
                self._safe_print(f"[DEBUG] 调用 Qwen API 生成假设答案...")
                fake_answer = qwen.send_message(
                    model="qwen-turbo",
                    system_content=(
                        "You are a financial report analyst. Your task is to generate a hypothetical markdown table "
                        "that could plausibly appear in a company's annual report or financial statement to answer the given query. "
                        "\n\n"
                        "**Requirements:**\n"
                        "1. Generate a markdown-format table (using | and - for formatting)\n"
                        "2. The table should be relevant to the question and contain typical fields/columns that would appear in such a table\n"
                        "3. Include appropriate table headers (such as: 类型, 项目, 金额, 单位, 备注, 年份, 季度, 比例, etc.)\n"
                        "4. Add a unit specification if applicable (e.g., '单位：万元' or '单位：元')\n"
                        "5. Include sample data rows that demonstrate the table structure\n"
                        "6. The table structure should match what would typically appear in Chinese financial reports\n"
                        "\n"
                        "**Table Format Example:**\n"
                        "```\n"
                        "单位：万元\n\n"
                        "| 类型 | 项目 | 金额 | 备注 |\n"
                        "|------|------|------|------|\n"
                        "| ...  | ...  | ...  | ...  |\n"
                        "```\n"
                        "\n"
                        "**Important:**\n"
                        "- Focus on creating a realistic table structure, not accurate data\n"
                        "- The table should help retrieve similar tables from financial reports\n"
                        "- Use Chinese column names appropriate for financial statements\n"
                        "- Include calculation formulas or notes if relevant (e.g., '① ② ③ = +' or '⑥ ① ④ ⑤ = - -')"
                    ),
                    human_content=f"Generate a markdown-format table that could appear in a company's financial report to answer this question: {query}\n\n"
                                 f"The table should include:\n"
                                 f"- Appropriate unit specification (if applicable)\n"
                                 f"- Relevant column headers based on the question\n"
                                 f"- Sample data rows showing the table structure\n"
                                 f"- Any relevant notes or calculation formulas",
                    is_structured=False
                )
                if isinstance(fake_answer, list):
                    fake_answer_str = ''.join(fake_answer)
                else:
                    fake_answer_str = str(fake_answer)
                queries.append(fake_answer_str)
                expansion_texts['hyde_text'] = fake_answer_str  # 保存HYDE生成的文本
                self._safe_print(f"[DEBUG] HYDE 扩展成功，生成假设答案长度: {len(fake_answer_str)}")
            except Exception as e:
                self._safe_print(f"[ERROR] HYDE expansion failed: {e}")
            timing_info['hyde_expansion'] = time.time() - hyde_start

        if use_multi_query:
            if progress_callback:
                progress_callback("🔄 Multi-Query 查询扩展中...", 38)
            self._safe_print(f"[DEBUG] 开始 Multi-Query 扩展...")
            multi_query_start = time.time()
            matched_concepts = find_financial_concepts(query, limit=5)
            concept_terms = [concept["term"] for concept in matched_concepts]
            concept_context_text = format_concepts_for_prompt(matched_concepts)
            glossary_instruction = (
                "Financial glossary context.\n"
                "For每个命中的术语，请按照以下格式逐条追加解释：\n"
                "1) Term名 + 主要别名/近义词\n"
                "2) 定义（至少一句）\n"
                "3) 计算方法/典型单位/数据来源（若适用）\n"
                "格式示例：\n"
                "\"毛利率\n"
                "- 别名：综合毛利率\n"
                "- 定义：体现产品盈利空间的比例……\n"
                "- 计算方式：毛利率 = (营业收入 - 营业成本) ÷ 营业收入\"\n"
                "在生成新的查询时，将上述解释附加在原问题后方的独立段落中，而不是写在括号里。\n"
                f"{concept_context_text}"
            )
            expansion_texts['glossary_context'] = concept_context_text
            expansion_texts['multi_query_methods'] = multi_query_config
            method_definitions = [
                (
                    1,
                    'synonym',
                    "你的任务是为问题中的财务专业名词补充详细解释。"
                    "上面已提供了财务术语词汇表(Financial glossary)，包含每个术语的别名、定义和计算方式。"
                    "任务要求：识别问题中包含的财务术语，参考 glossary 中的信息，在原问题之后单独列出每个术语的定义、近义词、计算方法。"
                    "格式：<原问题 名词解释：术语名称 定义...近义词...计算方法...>"
                    "示例：金盘科技2024年的毛利率是多少 -> "
                    "<金盘科技2024年的毛利率是多少 名词解释：毛利率 定义：毛利与营业收入之比，反映产品或业务的盈利空间 近义词：综合毛利率 计算方法：毛利率=(营业收入-营业成本)/营业收入>"
                    "优先使用 glossary 中提供的定义、近义词和计算方式。如果问题涉及财务术语但 glossary 中没有，可以用你自己的知识补充。"
                    "只有在问题完全不涉及任何财务术语时，才返回 <SKIP>。可生成1-2个带名词解释的查询，每个用尖括号包裹。"
                ),
                (
                    2,
                    'subquestion',
                    "根据财务指标将问题拆分为0-N个粒度更细的子问题。"
                    "每个子问题专注于单一指标/时间段/业务板块，并结合 glossary 里的术语或单位。"
                    "若没有合适的拆分则返回 <SKIP>；否则每个子问题单独用 <> 包裹。"
                ),
                (
                    3,
                    'variant',
                    "仅当原问题偏开放或信息不足时，生成情景化/变体提问，探索不同角度（如盈利质量、现金安全垫、海外扩张、补贴持续性等）。"
                    "若问题本就明确，则输出 <SKIP>；若需要改写，可生成1-2个查询，每个用 <> 包裹，并保持主体为金盘科技。"
                )
            ]
            import re
            for method_id, method_key, prompt in method_definitions:
                if not multi_query_config.get(method_key, False):
                    continue
                self._safe_print(f"[DEBUG] Multi-Query 方法 {method_id}...")
                try:
                    self._safe_print(f"[DEBUG] 调用 Qwen API 扩展查询...")
                    response = qwen.send_message(
                        model="qwen-turbo",
                        system_content=(
                            "You are assisting in an Enterprise RAG Challenge focused on annual reports. "
                            "Always maintain financial rigor and keep the company name unchanged."
                        ),
                        human_content=(
                            f"{prompt}\n\n"
                            f"{glossary_instruction}\n\n"
                            f"Original question: {query}"
                        ),
                        is_structured=False
                    )
                    extracted_queries = re.findall(r"<(.*?)>", response, flags=re.DOTALL)
                    self._safe_print(f"[DEBUG] 原始响应: {response[:200]}...")
                    self._safe_print(f"[DEBUG] 提取的查询: {extracted_queries}")
                    for q in extracted_queries:
                        q_stripped = q.strip()
                        self._safe_print(f"[DEBUG] 处理查询: '{q_stripped[:50]}...' (SKIP={q_stripped.upper() == 'SKIP'})")
                        if not q_stripped or q_stripped.upper() == "SKIP":
                            continue
                        queries.append(q_stripped)
                        expansion_texts['multi_query_texts'].append({
                            'method_id': method_id,
                            'query': q_stripped,
                            'concepts': concept_terms
                        })
                    self._safe_print(f"[DEBUG] Multi-Query 方法 {method_id} 完成，提取了 {len(extracted_queries)} 个查询，实际添加了 {len([q for q in extracted_queries if q.strip() and q.strip().upper() != 'SKIP'])} 个")
                except Exception as e:
                    self._safe_print(f"Expansion method {method_id} failed: {e}")
            timing_info['multi_query_expansion'] = time.time() - multi_query_start
        
        # 去重并清洗查询，避免重复 embedding 计算
        deduped_queries = []
        seen_queries = set()
        for q in queries:
            normalized_q = q.strip()
            if not normalized_q or normalized_q in seen_queries:
                continue
            deduped_queries.append(normalized_q)
            seen_queries.add(normalized_q)
        queries = deduped_queries

        inner_factor = 1.0
        self._safe_print("[DEBUG] queries is", queries)
        self._safe_print("[DEBUG] queries's length is", len(queries))

        # 预先生成 embeddings，避免在不同文档之间重复请求
        query_embeddings = {}
        embedding_start = time.time()
        for q in queries:
            try:
                emb_result = self.qwen.get_embeddings([q])
                if (
                    not emb_result
                    or not isinstance(emb_result, list)
                    or not emb_result[0]
                    or 'embedding' not in emb_result[0]
                ):
                    self._safe_print(f"[ERROR] embedding result is empty or invalid for query: {q[:80]}")
                    continue
                embedding = emb_result[0]['embedding']
                query_embeddings[q] = np.array(embedding, dtype=np.float32).reshape(1, -1)
            except Exception as e:
                self._safe_print(f"[ERROR] Failed to get embedding for query snippet '{q[:50]}': {e}")
        timing_info['embedding_generation'] = time.time() - embedding_start

        if not query_embeddings:
            raise ValueError("Failed to generate embeddings for all queries.")

        # 命中结果存储（用字典聚合）
        # key = (sha1, page_id or chunk_id), value = dict with distances, count, text
        aggregated_results = {}
        aggregation_lock = Lock()

        # 🎯 新检索策略：每个文档均召回 top_n 个chunks，然后统一按向量相似度排序
        # 收集所有文档的检索结果（总共 num_reports * top_n 个结果），
        # 统一按向量相似度（加权后的distance）排序，截断式选取前 top_n 个结果
        num_reports = len(matching_reports)
        
        self._safe_print(f"[INFO] 📊 检索策略: {num_reports}个文档, 每个文档检索 {top_n} 个chunks (总计最多 {num_reports * top_n} 个结果)")

        # 向量检索阶段
        if progress_callback:
            progress_callback("🔎 向量检索中...", 45)

        def process_query_for_document(report, query_text, embedding_array):
            local_hits = []
            try:
                document = report["document"]
                vector_db = report["vector_db"]
                chunks = document["content"]["chunks"]
                pages = document["content"]["pages"]
                sha1 = document["metainfo"]["sha1_name"]
                actual_top_n = min(top_n, len(chunks))
                if actual_top_n == 0:
                    return []
                distances, indices = vector_db.search(x=embedding_array, k=actual_top_n)
            
                for distance, index in zip(distances[0], indices[0]):
                    distance = round(float(distance)*inner_factor, 4)
                    chunk = chunks[index]
                    parent_page = next(page for page in pages if page["page"] == chunk["page"])
                    
                    if return_parent_pages:
                        # Include sha1 in key to differentiate same page numbers across different reports
                        key = (sha1, "page", parent_page["page"])
                        text = parent_page["text"]
                        page_id = parent_page["page"]
                    else:
                        key = (sha1, "chunk", index)
                        text = chunk["text"]
                        page_id = chunk["page"]
                    
                    local_hits.append((key, page_id, text, distance, sha1))
            except Exception as e:
                self._safe_print(f"[ERROR] Vector search failed for query '{query_text[:60]}' in report {report.get('name')}: {e}")
            return local_hits

        total_tasks = len(query_embeddings) * num_reports
        max_workers = min(self.parallel_workers, total_tasks) if total_tasks > 0 else 1
        vector_search_start = time.time()

        with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
            futures = []
            for report in matching_reports:
                for q_text, embedding_array in query_embeddings.items():
                    futures.append(executor.submit(process_query_for_document, report, q_text, embedding_array))

            for future in as_completed(futures):
                doc_hits = future.result()
                if not doc_hits:
                    continue
                with aggregation_lock:
                    for key, page_id, text, distance, sha1 in doc_hits:
                        if key not in aggregated_results:
                            aggregated_results[key] = {
                                "page": page_id,
                                "text": text,
                                "distances": [distance],
                                "count": 1,
                                "source_sha1": sha1  # Track source document
                            }
                        else:
                            aggregated_results[key]["distances"].append(distance)
                            aggregated_results[key]["count"] += 1

        timing_info['vector_search'] = time.time() - vector_search_start
    
        # 加权规则: 1次=×1.0, 2次=×1.2, 3次=×1.4。. 以此类推。 注意：当前 faiss 用的是 IndexFlatIP（内积），distance 越大表示相关性越高。因此，命中多次时，应该让 distance 增大。
        def weight_factor(count: int) -> float:
            return 1.0 + 0.2 * (count - 1)
    
        final_results = []
        for key, info in aggregated_results.items():
            base_distance = max(info["distances"])  # 取最大距离作为基准
            factor = weight_factor(info["count"])
            weighted_distance = round(base_distance * factor, 4)
        
            final_results.append({
                "distance": weighted_distance,
                "page": info["page"],
                "text": info["text"],
                "hit_count": info["count"],  # 方便调试看到被命中次数
                "source_sha1": info["source_sha1"]  # Include source document
            })
    
        # 聚合：按加权后的距离降序，取前 top_n（distance越大越相关）
        final_results = sorted(final_results, key=lambda x: x["distance"], reverse=True)
        
        # Debug: 显示聚合后的文档分布
        source_distribution = {}
        for res in final_results[:top_n]:
            source = res.get("source_sha1", "Unknown")
            source_distribution[source] = source_distribution.get(source, 0) + 1
        print(f"[DEBUG] Top {top_n} results distribution: {source_distribution}")
        
        final_results = final_results[:top_n]


        # 检索完成
        if progress_callback:
            progress_callback("✅ 检索完成，准备重排序...", 55)
        
        # Debug: 统计来源分布
        source_counts = {}
        for result in final_results[:top_n]:
            sha1 = result["source_sha1"]
            source_counts[sha1] = source_counts.get(sha1, 0) + 1
        print(f"[DEBUG] Top-{top_n} results distribution by source:")
        for sha1, count in sorted(source_counts.items(), key=lambda x: -x[1]):
            print(f"  {sha1}: {count} results")
        
        # 将时间信息、扩展文本与结果一起返回
        return {
            'results': final_results,
            'timing': timing_info,
            'expansion_texts': expansion_texts
        }

    def retrieve_all(self, company_name: str) -> List[Dict]:
        """Retrieve all pages from all reports matching the company name."""
        #print("\n retrieve_all be used")
        
        # Collect all matching reports
        matching_reports = []
        for report in self.all_dbs:
            document = report.get("document", {})
            metainfo = document.get("metainfo")
            if not metainfo:
                continue
            if metainfo.get("company_name") == company_name:
                matching_reports.append(report)
        
        if not matching_reports:
            _log.error(f"No report found with '{company_name}' company name.")
            raise ValueError(f"No report found with '{company_name}' company name.")
        
        if len(matching_reports) > 1:
            print(f"[INFO] retrieve_all: Found {len(matching_reports)} reports for '{company_name}', retrieving all pages from all reports")
        
        # Collect pages from all matching reports
        all_pages = []
        for report in matching_reports:
            document = report["document"]
            pages = document["content"]["pages"]
            sha1 = document["metainfo"]["sha1_name"]
            
            for page in sorted(pages, key=lambda p: p["page"]):
                result = {
                    "distance": 0.5,
                    "page": page["page"],
                    "text": page["text"],
                    "source_sha1": sha1  # Track which report this page comes from
                }
                all_pages.append(result)
            
        return all_pages