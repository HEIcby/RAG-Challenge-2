import json
import logging
from typing import List, Tuple, Dict, Union
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
    fallback_strategy: str = "all"  # "all" 或 "latest"
) -> List[Dict]:
    """
    基于时间信息和公司名路由到合适的文档
    
    Args:
        company_name: 公司名称
        question: 用户问题
        all_reports: 所有可用的报告
        fallback_strategy: 当没有时间信息时的回退策略
            - "all": 返回该公司所有文档
            - "latest": 只返回最新年份的文档
    
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
    
    # 2. 提取问题中的年份信息（带时间窗口扩展）
    years = extract_years_from_question(question, expand_window=True)
    
    # 3. 如果有明确年份，只返回对应年份的文档
    if years:
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
                    if report_year in years:
                        filtered_reports.append(report)
                except (ValueError, TypeError):
                    pass
        
        if filtered_reports:
            print(f"[INFO] 🎯 时间路由（含前后年窗口）: 年份 {years}，匹配到 {len(filtered_reports)} 个文档")
            return filtered_reports
        else:
            print(f"[WARNING] ⚠️ 识别到年份 {years}，但未找到对应文档，回退到全部文档")
    
    # 4. 没有时间信息时的回退策略
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
                    pass
        
        if latest_reports:
            print(f"[INFO] 📅 无明确时间信息，使用最新年份 {latest_year} 的文档")
            return latest_reports
    
    # 默认返回所有该公司的文档
    print(f"[INFO] 📚 无明确时间信息，使用该公司所有 {len(company_reports)} 个文档")
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
        
    def retrieve_by_company_name(self, company_name: str, query: str, top_n: int = 3, return_parent_pages: bool = False) -> List[Dict]:
        print("BM25Retriever retrieve_by_company_name is called")
        
        # 🎯 先收集所有文档，然后使用时间路由
        all_documents = []
        for path in self.documents_dir.glob("*.json"):
            with open(path, 'r', encoding='utf-8') as f:
                doc = json.load(f)
                all_documents.append({
                    "path": path,
                    "document": doc,
                    "sha1": doc["metainfo"]["sha1_name"]
                })
        
        # 使用时间路由过滤文档
        years = extract_years_from_question(query)
        matching_documents = []
        
        for doc_info in all_documents:
            doc = doc_info["document"]
            metainfo = doc.get("metainfo", {})
            sha1 = doc_info["sha1"]
            
            # 检查公司名
            if metainfo.get("company_name") != company_name:
                continue
            
            # 如果有年份信息，进一步过滤
            if years:
                # 优先从 metainfo 读取，如果没有则从 year_lookup 读取
                doc_year = metainfo.get("year")
                if doc_year is None and sha1 in self.year_lookup:
                    doc_year = self.year_lookup[sha1]
                
                if doc_year is not None:
                    try:
                        doc_year = int(doc_year)
                        if doc_year not in years:
                            continue
                    except (ValueError, TypeError):
                        pass
            
            matching_documents.append(doc_info)
        
        if not matching_documents:
            raise ValueError(f"No report found with '{company_name}' company name.")
        
        if years:
            print(f"[INFO] 🎯 BM25时间路由: 识别年份 {years}，匹配到 {len(matching_documents)} 个文档")
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
    def __init__(self, vector_db_dir: Path, documents_dir: Path, use_hyde: bool = True, use_multi_query: bool = True, subset_path: Path = None):
        self.vector_retriever = VectorRetriever(vector_db_dir, documents_dir, use_hyde=use_hyde, use_multi_query=use_multi_query, subset_path=subset_path)
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
        progress_callback=None
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
            
        Returns:
            List of reranked document dictionaries with scores
        """
        # Get initial results from vector retriever
        vector_results = self.vector_retriever.retrieve_by_company_name(
            company_name=company_name,
            query=query,
            top_n=llm_reranking_sample_size,
            return_parent_pages=return_parent_pages,
            use_hyde=use_hyde,
            use_multi_query=use_multi_query,
            progress_callback=progress_callback
        )
        
        print(f"[DEBUG] Initial vector results count: {len(vector_results)}")

        # 重排序阶段（这是最耗时的部分）
        if progress_callback:
            progress_callback("🎯 LLM 重排序中（这可能需要一些时间）...", 58)
        
        # Rerank results using LLM
        reranked_results = self.reranker.rerank_documents(
            query=query,
            documents=vector_results,
            documents_batch_size=documents_batch_size,
            llm_weight=llm_weight
        )

        print(f"[DEBUG] Reranked results count: {len(reranked_results)}")
        #print("[DEBUG] HybridRetriever retrieve_by_company_name is called")
        print(f"[DEBUG] Final top_n: {top_n}")
        return reranked_results[:top_n]


class VectorRetriever:
    def __init__(self, vector_db_dir: Path, documents_dir: Path, use_hyde: bool = True, use_multi_query: bool = True, subset_path: Path = None):
        self.vector_db_dir = vector_db_dir
        self.documents_dir = documents_dir
        self.subset_path = subset_path
        self.year_lookup = self._load_year_lookup() if subset_path else {}
        self.all_dbs = self._load_dbs()
        self.qwen = BaseQwenProcessor()
        self.use_hyde = use_hyde
        self.use_multi_query = use_multi_query
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

    
   
    def retrieve_by_company_name(self, company_name: str, query: str, llm_reranking_sample_size: int = None, top_n: int = 3, return_parent_pages: bool = False, use_hyde: bool = None, use_multi_query: bool = None, progress_callback=None) -> List[Tuple[str, float]]:
        import sys
        print("[DEBUG] VectorRetriever retrieve_by_company_name is called")
        sys.stdout.flush()

        # 🎯 使用时间智能路由替代原有的简单公司名过滤
        # 这样可以根据问题中的时间信息自动定位到对应年份的文档
        if progress_callback:
            progress_callback("📚 定位相关文档中...", 28)
        
        matching_reports = route_reports_by_time(
            company_name=company_name,
            question=query,
            all_reports=self.all_dbs,
            fallback_strategy="all"  # 无时间信息时使用所有文档
        )
        
        if not matching_reports:
            _log.error(f"No report found with '{company_name}' company name.")
            raise ValueError(f"No report found with '{company_name}' company name.")
        
        if len(matching_reports) > 1:
            print(f"[INFO] Found {len(matching_reports)} reports for '{company_name}', retrieving from all")
            sys.stdout.flush()
            for rep in matching_reports:
                doc = rep.get("document", {})
                metainfo = doc.get("metainfo", {})
                year = metainfo.get("year", "unknown")
                print(f"  - Report: {rep['name']} (Year: {year})")
                sys.stdout.flush()
        
        # Priority parameters
        use_hyde = self.use_hyde if use_hyde is None else use_hyde
        use_multi_query = self.use_multi_query if use_multi_query is None else use_multi_query
        print(f"[DEBUG][retrieve_by_company_name] use_hyde={use_hyde}, use_multi_query={use_multi_query}")
        sys.stdout.flush()
        
        qwen = BaseQwenProcessor()
        # 控制multi_query和hyde扩充
        queries = [query]

        if use_hyde:
            if progress_callback:
                progress_callback("🔮 HYDE 查询扩展中...", 32)
            print(f"[DEBUG] 开始 HYDE 扩展...")
            sys.stdout.flush()
            try:
                print(f"[DEBUG] 调用 Qwen API 生成假设答案...")
                sys.stdout.flush()
                fake_answer = qwen.send_message(
                    model="qwen-turbo",
                    system_content=(
                        "You are a creative report writer. "
                        "When asked a question, your task is NOT to retrieve real-time or factual financial data, "
                        "but instead to **invent, compile, or simulate** a helpful passage, article, or news-style report "
                        "that could plausibly assist in answering the query. "
                        "Even if the query asks about real numbers or unavailable information, "
                        "you should respond by **fabricating a coherent, contextually relevant narrative** "
                        "rather than disclaiming lack of data. "
                        "Your goal is to produce a well-written piece (report, analysis, or article) "
                        "that reads like it could come from a newspaper, magazine, or research commentary."
                    ),
                    human_content=f"Write a full passage to address this query in an informative and narrative way: {query}",
                    is_structured=False
                )
                if isinstance(fake_answer, list):
                    fake_answer_str = ''.join(fake_answer)
                else:
                    fake_answer_str = str(fake_answer)
                queries.append(fake_answer_str)
                print(f"[DEBUG] HYDE 扩展成功，生成假设答案长度: {len(fake_answer_str)}")
                sys.stdout.flush()
            except Exception as e:
                print(f"[ERROR] HYDE expansion failed: {e}")
                sys.stdout.flush()

        if use_multi_query:
            if progress_callback:
                progress_callback("🔄 Multi-Query 查询扩展中...", 38)
            print(f"[DEBUG] 开始 Multi-Query 扩展...")
            sys.stdout.flush()
            # expansion_methods = {
            #     1: "Expand the question by replacing key terms with synonyms or related terms while keeping the meaning in the context of annual reports and financial statements. Generate three queries, each wrapped in <>.",
            #     2: "Expand the question by including broader or narrower related terms (hypernyms or hyponyms) relevant to annual reports and financial statements. Generate three queries, each wrapped in <>.",
            #     3: "Rewrite the question into three paraphrased variations that keep the same intent in the context of annual reports and financial statements. Generate three queries, each wrapped in <>."
            # }
            expansion_methods = {
                1: "Expand the question by replacing key terms with synonyms or related terms while keeping the meaning in the context of annual reports and financial statements. Generate one query, wrapped in <>.",
                2: "Expand the question by including broader or narrower related terms (hypernyms or hyponyms) relevant to annual reports and financial statements. Generate one query, wrapped in <>.",
                3: "Rewrite the question into one paraphrased variation that keeps the same intent in the context of annual reports and financial statements. Generate one query, wrapped in <>."
            }
            import re
            for method_id, prompt in expansion_methods.items():
                print(f"[DEBUG] Multi-Query 方法 {method_id}...")
                sys.stdout.flush()
                try:
                    print(f"[DEBUG] 调用 Qwen API 扩展查询...")
                    sys.stdout.flush()
                    response = qwen.send_message(
                        model="qwen-turbo",
                        system_content="You are assisting in an Enterprise RAG Challenge focused on annual reports.",
                        human_content=f"{prompt}\nOriginal question: {query}",
                        is_structured=False
                    )
                    extracted_queries = re.findall(r"<(.*?)>", response, flags=re.DOTALL)
                    for q in extracted_queries:
                        queries.append(q.strip())
                    print(f"[DEBUG] Multi-Query 方法 {method_id} 完成，提取了 {len(extracted_queries)} 个查询")
                    sys.stdout.flush()
                except Exception as e:
                    print(f"Expansion method {method_id} failed: {e}")
        
        # 命中结果存储（用字典聚合）
        # key = (sha1, page_id or chunk_id), value = dict with distances, count, text
        aggregated_results = {}

        inner_factor = 1.0
        print("[DEBUG] queries is", queries)
        print("[DEBUG] queries's length is", len(queries))

        # 🎯 智能分配策略：将 top_n 平均分配到每个匹配的文档
        # 这样可以确保每个文档都有公平的机会被检索到
        # 避免单个文档dominate所有结果
        num_reports = len(matching_reports)
        top_n_per_report = max(1, top_n // num_reports)  # 确保至少为1
        remaining = top_n % num_reports  # 余数分配给前几个文档
        
        print(f"[INFO] 📊 检索策略: {num_reports}个文档, 每个分配约{top_n_per_report}个chunks (总预算{top_n})")
        if remaining > 0:
            print(f"[INFO] 💡 前{remaining}个文档额外获得1个chunk配额")

        # 向量检索阶段
        if progress_callback:
            progress_callback("🔎 向量检索中...", 45)

        # Process each matching report
        for idx, report in enumerate(matching_reports):
            document = report["document"]
            vector_db = report["vector_db"]
            chunks = document["content"]["chunks"]
            pages = document["content"]["pages"]
            sha1 = document["metainfo"]["sha1_name"]
            
            # 为每个文档分配合适的 top_n
            doc_top_n = top_n_per_report + (1 if idx < remaining else 0)
            actual_top_n = min(doc_top_n, len(chunks))
            
            print(f"[DEBUG] 从 {sha1} 检索 {actual_top_n} 个chunks (共{len(chunks)}个)")
            
            # Retrieve for each query
            for q in queries:
                if not q.strip():
                    print(f"[ERROR] query is empty, skip embedding: '{q}'")
                    continue
                emb_result = self.qwen.get_embeddings([q])
                if not emb_result or not isinstance(emb_result, list) or not emb_result[0] or 'embedding' not in emb_result[0]:
                    print(f"[ERROR] embedding result is empty or invalid for query: {q}, emb_result: {emb_result}")
                    continue
                print("[DEBUG] emb_result[0] =", emb_result[0])
                embedding = emb_result[0]['embedding']
                embedding_array = np.array(embedding, dtype=np.float32).reshape(1, -1)
                distances, indices = vector_db.search(x=embedding_array, k=actual_top_n)
            
                for distance, index in zip(distances[0], indices[0]):
                    distance = round(float(distance)*inner_factor, 4)
                    chunk = chunks[index]
                    parent_page = next(page for page in pages if page["page"] == chunk["page"])
                    
                    # Debug: 打印每个文档的检索结果
                    print(f"[DEBUG] Retrieved from {sha1}: page={chunk['page']}, distance={distance}, text_preview={chunk['text'][:50]}...")
                    
                    if return_parent_pages:
                        # Include sha1 in key to differentiate same page numbers across different reports
                        key = (sha1, "page", parent_page["page"])
                        text = parent_page["text"]
                        page_id = parent_page["page"]
                    else:
                        key = (sha1, "chunk", index)
                        text = chunk["text"]
                        page_id = chunk["page"]
                    
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
        
    
        return final_results

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