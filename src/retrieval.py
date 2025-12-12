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
                            "vector_similarity": score,
                            "page": parent_page["page"],
                            "text": parent_page["text"],
                            "source_sha1": sha1  # Add source document identifier
                        }
                        all_retrieval_results.append(result)
                else:
                    result = {
                        "vector_similarity": score,
                        "page": chunk["page"],
                        "text": chunk["text"],
                        "source_sha1": sha1  # Add source document identifier
                    }
                    all_retrieval_results.append(result)
        
        # Sort by score and return top_n results across all documents
        all_retrieval_results.sort(key=lambda x: x["vector_similarity"], reverse=True)
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
        retrieval_method: str = "basic",
        max_hops: int = 4,
        neighbor_k: int = 30,
    ):
        self.vector_retriever = VectorRetriever(
            vector_db_dir,
            documents_dir,
            use_hyde=use_hyde,
            use_multi_query=use_multi_query,
            subset_path=subset_path,
            parallel_workers=parallel_workers,
            multi_query_methods=multi_query_methods,
            retrieval_method=retrieval_method,
            max_hops=max_hops,
            neighbor_k=neighbor_k,
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
        multi_query_config: Optional[Dict[str, bool]] = None,
        retrieval_method: str = "basic",
        max_hops: int = 4,
        neighbor_k: int = 30
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
            multi_query_config=multi_query_config,
            retrieval_method=retrieval_method,
            max_hops=max_hops,
            neighbor_k=neighbor_k
        )
        
        # 处理返回结果（可能是dict或list）
        expansion_texts = {}
        algorithm_contribution = None
        if isinstance(vector_retrieval_result, dict) and 'timing' in vector_retrieval_result:
            timing_info.update(vector_retrieval_result['timing'])
            vector_results = vector_retrieval_result['results']
            # 提取扩展文本信息
            if 'expansion_texts' in vector_retrieval_result:
                expansion_texts = vector_retrieval_result['expansion_texts']
            # 提取算法贡献统计（仅hybrid_expansion）
            if 'algorithm_contribution' in vector_retrieval_result:
                algorithm_contribution = vector_retrieval_result['algorithm_contribution']
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
            'initial_retrieval_results': vector_results,  # 保存初始召回结果（reranking前）
            'timing': timing_info,
            'expansion_texts': expansion_texts,
            'reranker_stats': self.reranker.get_stats(),
            'algorithm_contribution': algorithm_contribution  # 传递算法贡献统计（如果存在）
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
        retrieval_method: str = "basic",
        max_hops: int = 4,
        neighbor_k: int = 30,
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
        self.retrieval_method = retrieval_method
        self.max_hops = max_hops
        self.neighbor_k = neighbor_k
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
    
    def _get_vector_by_id(self, vector_db, doc_id):
        """从 FAISS 索引中获取指定 ID 的向量"""
        try:
            return vector_db.reconstruct(int(doc_id))
        except Exception as e:
            self._safe_print(f"[WARNING] Failed to reconstruct vector for ID {doc_id}: {e}")
            return None
    
    def _normalize_vector(self, vec):
        """Normalize a vector to unit length."""
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec
        return vec / norm
    
    def _ssg_search(self, vector_db, anchor_id, anchor_vec, max_hops=4, neighbor_k=30):
        """
        SSG Traversal Algorithm implementation.
        Returns a dictionary with results and detailed traversal information.
        
        Args:
            vector_db: FAISS index
            anchor_id: Starting chunk index
            anchor_vec: Starting chunk embedding vector
            max_hops: Maximum number of hops to traverse
            neighbor_k: Number of neighbors to consider at each hop
        
        Returns:
            Dictionary with "results" (list of (score, idx) tuples) and "traversal_details"
        """
        visited = set([int(anchor_id)])
        results = []  # List of (score, index)
        
        # 详细追踪信息
        traversal_details = {
            "anchor": {"idx": int(anchor_id), "score": None},
            "hops": [],
            "path": [int(anchor_id)],
            "total_hops": 0,
            "total_discovered": 1
        }
        
        current_idx = int(anchor_id)
        current_vec = anchor_vec
        previous_similarity = 1.0  # Anchor similarity with itself
        
        # 添加anchor到结果
        results.append((1.0, current_idx))
        
        for hop_num in range(1, max_hops + 1):
            # 使用当前chunk的向量搜索邻居
            current_vec_reshaped = current_vec.reshape(1, -1)
            distances, indices = vector_db.search(x=current_vec_reshaped, k=neighbor_k + 1)  # +1 to exclude self
            
            candidates = []
            for d, idx in zip(distances[0], indices[0]):
                idx = int(idx)
                if idx == -1 or idx in visited:
                    continue
                
                # 获取候选chunk的向量
                candidate_vec = self._get_vector_by_id(vector_db, idx)
                if candidate_vec is None:
                    continue
                
                # 计算chunk-to-chunk相似度
                chunk_similarity = float(np.dot(current_vec.flatten(), candidate_vec.flatten()))
                
                candidates.append({
                    "idx": idx,
                    "score": chunk_similarity,
                    "selected": False
                })
            
            if not candidates:
                break
            
            # 按相似度排序，选择最佳候选
            candidates.sort(key=lambda x: x["score"], reverse=True)
            best_candidate = candidates[0]
            best_idx = best_candidate["idx"]
            best_similarity = best_candidate["score"]
            
            # 早停检查：如果相似度不再提升，停止遍历
            if best_similarity <= previous_similarity:
                break
            
            # 标记选中的候选
            best_candidate["selected"] = True
            
            # 记录这一跳的详细信息
            hop_info = {
                "hop_number": hop_num,
                "current_chunk": current_idx,
                "candidates": candidates[:10],  # 只记录前10个候选
                "selected_idx": best_idx,
                "selected_score": best_similarity
            }
            traversal_details["hops"].append(hop_info)
            
            # 跳转到新chunk
            visited.add(best_idx)
            traversal_details["path"].append(best_idx)
            current_idx = best_idx
            current_vec = self._get_vector_by_id(vector_db, best_idx)
            if current_vec is None:
                break
            
            # 添加新chunk到结果（使用chunk-to-chunk相似度作为初始分数）
            results.append((best_similarity, best_idx))
            previous_similarity = best_similarity
            traversal_details["total_discovered"] += 1
        
        traversal_details["total_hops"] = len(traversal_details["hops"])
        
        return {
            "results": results,
            "traversal_details": traversal_details
        }
    
    def _triangulation_search(self, vector_db, query_vec, anchor_id, anchor_vec, max_hops=4, neighbor_k=30):
        """
        Triangulation FullDim Algorithm implementation.
        Uses geometric triangulation in full embedding space to select next chunk.
        
        Args:
            vector_db: FAISS index
            query_vec: Query embedding vector
            anchor_id: Starting chunk index
            anchor_vec: Starting chunk embedding vector
            max_hops: Maximum number of hops to traverse
            neighbor_k: Number of neighbors to consider at each hop
        
        Returns:
            Dictionary with "results" (list of (score, idx) tuples) and "traversal_details"
        """
        visited = set([int(anchor_id)])
        results = []  # List of (centroid_score, index)
        
        query_vec_flat = query_vec.flatten()
        
        # 详细追踪信息
        traversal_details = {
            "anchor": {"idx": int(anchor_id), "score": None},
            "hops": [],
            "path": [int(anchor_id)],
            "total_hops": 0
        }
        
        # 计算anchor的query-to-chunk相似度
        anchor_query_sim = float(np.dot(query_vec_flat, anchor_vec.flatten()))
        traversal_details["anchor"]["score"] = anchor_query_sim
        results.append((anchor_query_sim, int(anchor_id)))
        
        current_idx = int(anchor_id)
        current_vec = anchor_vec
        
        for hop_num in range(1, max_hops + 1):
            # 使用当前chunk的向量搜索邻居
            current_vec_reshaped = current_vec.reshape(1, -1)
            distances, indices = vector_db.search(x=current_vec_reshaped, k=neighbor_k + 1)
            
            candidates = []
            for d, idx in zip(distances[0], indices[0]):
                idx = int(idx)
                if idx == -1 or idx in visited:
                    continue
                
                candidate_vec = self._get_vector_by_id(vector_db, idx)
                if candidate_vec is None:
                    continue
                
                candidate_vec_flat = candidate_vec.flatten()
                current_vec_flat = current_vec.flatten()
                
                # 计算query-to-candidate相似度
                query_to_candidate = float(np.dot(query_vec_flat, candidate_vec_flat))
                
                # 构建几何三角形：query, current_chunk, candidate
                # 计算三角形质心
                centroid = (query_vec_flat + current_vec_flat + candidate_vec_flat) / 3.0
                
                # 计算质心到查询的距离（使用欧氏距离）
                centroid_distance = float(np.linalg.norm(centroid - query_vec_flat))
                
                # 转换为相似度分数（距离越小，分数越高）
                # 使用负距离或转换为相似度分数
                centroid_score = 1.0 / (1.0 + centroid_distance)  # 简单的转换
                
                candidates.append({
                    "idx": idx,
                    "query_to_candidate": query_to_candidate,
                    "centroid_score": centroid_score,
                    "centroid_distance": centroid_distance,
                    "selected": False
                })
            
            if not candidates:
                break
            
            # 按质心分数排序（分数越高越好，即距离越小越好）
            candidates.sort(key=lambda x: x["centroid_score"], reverse=True)
            best_candidate = candidates[0]
            best_idx = best_candidate["idx"]
            best_centroid_score = best_candidate["centroid_score"]
            
            # 标记选中的候选
            best_candidate["selected"] = True
            
            # 记录这一跳的详细信息
            hop_info = {
                "hop_number": hop_num,
                "current_chunk": current_idx,
                "candidates": candidates[:10],  # 只记录前10个候选
                "selected_idx": best_idx,
                "centroid_score": best_centroid_score,
                "selection_reason": "质心距离最优"
            }
            traversal_details["hops"].append(hop_info)
            
            # 跳转到新chunk
            visited.add(best_idx)
            traversal_details["path"].append(best_idx)
            current_idx = best_idx
            current_vec = self._get_vector_by_id(vector_db, best_idx)
            if current_vec is None:
                break
            
            # 添加新chunk到结果（使用质心分数）
            results.append((best_centroid_score, best_idx))
        
        traversal_details["total_hops"] = len(traversal_details["hops"])
        
        return {
            "results": results,
            "traversal_details": traversal_details
        }
    
    def retrieve_by_company_name(self, company_name: str, query: str, llm_reranking_sample_size: int = None, top_n: int = 3, return_parent_pages: bool = False, use_hyde: bool = None, use_multi_query: bool = None, progress_callback=None, selected_years: List[int] = None, multi_query_config: Optional[Dict[str, bool]] = None, retrieval_method: str = "basic", max_hops: int = 2, neighbor_k: int = 10) -> List[Tuple[str, float]]:
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
        # 处理检索方法参数：优先使用传入参数（如果明确传入非默认值），否则使用实例变量
        # 关键逻辑：如果传入的参数是默认值"basic"，但实例变量中有非默认值，则使用实例变量
        # 这样可以支持动态更新（例如从UI更新processor.retrieval_method）
        self._safe_print(f"[DEBUG] 参数处理前: 传入retrieval_method={retrieval_method}, 实例self.retrieval_method={getattr(self, 'retrieval_method', 'N/A')}")
        
        if retrieval_method == "basic":
            # 如果传入的是默认值"basic"，检查是否有实例变量（可能是从UI更新的）
            if hasattr(self, 'retrieval_method') and self.retrieval_method != "basic":
                retrieval_method = self.retrieval_method
                self._safe_print(f"[DEBUG] ✅ 使用实例变量中的retrieval_method: {retrieval_method} (覆盖默认值'basic')")
            else:
                self._safe_print(f"[DEBUG] 使用默认值'basic'")
        else:
            # 如果传入的不是"basic"，直接使用传入的值（这是正确的行为）
            self._safe_print(f"[DEBUG] ✅ 使用传入的retrieval_method参数: {retrieval_method}")
        
        # 对于max_hops和neighbor_k，如果传入的是默认值，则使用实例变量
        if max_hops == 4 and hasattr(self, 'max_hops') and self.max_hops != 4:
            max_hops = self.max_hops
            self._safe_print(f"[DEBUG] 使用实例变量中的max_hops: {max_hops}")
        if neighbor_k == 30 and hasattr(self, 'neighbor_k') and self.neighbor_k != 30:
            neighbor_k = self.neighbor_k
            self._safe_print(f"[DEBUG] 使用实例变量中的neighbor_k: {neighbor_k}")
        self._safe_print(f"[DEBUG][retrieve_by_company_name] use_hyde={use_hyde}, use_multi_query={use_multi_query}, retrieval_method={retrieval_method}, max_hops={max_hops}, neighbor_k={neighbor_k}")
        
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
                    self._safe_print(f"[DEBUG] ========== Multi-Query 方法 {method_id} ({method_key}) ==========")
                    self._safe_print(f"[DEBUG] 原始响应 (前500字符): {response[:500]}...")
                    self._safe_print(f"[DEBUG] 提取的查询数量: {len(extracted_queries)}")
                    self._safe_print(f"[DEBUG] 提取的查询列表: {extracted_queries}")
                    
                    added_count = 0
                    skipped_count = 0
                    for q in extracted_queries:
                        q_stripped = q.strip()
                        is_skip = not q_stripped or q_stripped.upper() == "SKIP"
                        self._safe_print(f"[DEBUG] 处理查询: '{q_stripped[:80]}...' (长度={len(q_stripped)}, SKIP={is_skip})")
                        if is_skip:
                            skipped_count += 1
                            continue
                        queries.append(q_stripped)
                        expansion_texts['multi_query_texts'].append({
                            'method_id': method_id,
                            'query': q_stripped,
                            'concepts': concept_terms
                        })
                        added_count += 1
                    
                    self._safe_print(f"[DEBUG] Multi-Query 方法 {method_id} 统计:")
                    self._safe_print(f"[DEBUG]   - 提取的查询总数: {len(extracted_queries)}")
                    self._safe_print(f"[DEBUG]   - 跳过(SKIP): {skipped_count}")
                    self._safe_print(f"[DEBUG]   - 实际添加: {added_count}")
                    self._safe_print(f"[DEBUG] ========================================================")
                except Exception as e:
                    self._safe_print(f"Expansion method {method_id} failed: {e}")
            timing_info['multi_query_expansion'] = time.time() - multi_query_start
        
        # 去重并清洗查询，避免重复 embedding 计算
        self._safe_print(f"[DEBUG] ========== 查询去重处理 ==========")
        self._safe_print(f"[DEBUG] 去重前的查询总数: {len(queries)}")
        self._safe_print(f"[DEBUG] 去重前的查询列表:")
        for idx, q in enumerate(queries, 1):
            self._safe_print(f"[DEBUG]   {idx}. {q[:100]}...")
        
        deduped_queries = []
        seen_queries = set()
        duplicate_count = 0
        for q in queries:
            normalized_q = q.strip()
            if not normalized_q:
                continue
            if normalized_q in seen_queries:
                duplicate_count += 1
                self._safe_print(f"[DEBUG]   发现重复查询: '{normalized_q[:80]}...'")
                continue
            deduped_queries.append(normalized_q)
            seen_queries.add(normalized_q)
        queries = deduped_queries

        self._safe_print(f"[DEBUG] 去重后的查询总数: {len(queries)}")
        self._safe_print(f"[DEBUG] 重复查询数量: {duplicate_count}")
        self._safe_print(f"[DEBUG] 去重后的查询列表:")
        for idx, q in enumerate(queries, 1):
            self._safe_print(f"[DEBUG]   {idx}. {q[:100]}...")
        self._safe_print(f"[DEBUG] ==================================")

        inner_factor = 1.0

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
        # key = (sha1, page_id or chunk_id), value = dict with similarities, count, text, retrieval_sources
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
            """
            为单个查询-文档对执行检索
            返回格式：(key, page_id, text, vector_similarity, sha1, query_text, retrieval_source)
            增加query_text字段以追踪命中来源，增加retrieval_source字段以追踪检索方法来源
            """
            local_hits = []
            traversal_details_list = []
            try:
                document = report["document"]
                vector_db = report["vector_db"]
                chunks = document["content"]["chunks"]
                pages = document["content"]["pages"]
                sha1 = document["metainfo"]["sha1_name"]
                actual_top_n = min(top_n, len(chunks))
                if actual_top_n == 0:
                    return []
                
                # 根据检索方法选择不同的策略
                if retrieval_method == "basic":
                    # Basic Retrieval - 完全保持原有逻辑不变
                    distances, indices = vector_db.search(x=embedding_array, k=actual_top_n)
                
                    for distance, index in zip(distances[0], indices[0]):
                        vector_similarity = round(float(distance)*inner_factor, 4)
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
                        
                        local_hits.append((key, page_id, text, vector_similarity, sha1, "basic"))
                
                elif retrieval_method in ["ssg", "triangulation"]:
                    # SSG / Triangulation - 独立运行，不依赖 basic search
                    # 首先找到锚点（只找1个最相似的作为起始点）
                    anchor_idx = None
                    anchor_score = None
                    anchor_vec = None
                    distances, indices = vector_db.search(x=embedding_array, k=1)
                    if len(distances[0]) > 0 and indices[0][0] != -1:
                        anchor_idx = int(indices[0][0])
                        anchor_score = float(distances[0][0])
                        anchor_vec = self._get_vector_by_id(vector_db, anchor_idx)
                    
                    if anchor_vec is not None and anchor_idx is not None:
                        expansion_result = None
                        if retrieval_method == "ssg":
                            expansion_result = self._ssg_search(
                                vector_db, anchor_idx, anchor_vec, 
                                max_hops=max_hops, neighbor_k=neighbor_k
                            )
                        elif retrieval_method == "triangulation":
                            expansion_result = self._triangulation_search(
                                vector_db, embedding_array, anchor_idx, anchor_vec,
                                max_hops=max_hops, neighbor_k=neighbor_k
                            )
                        
                        if expansion_result and isinstance(expansion_result, dict):
                            expanded_results = expansion_result.get("results", [])
                            traversal_details = expansion_result.get("traversal_details", None)
                            
                            # 保存遍历详情
                            if traversal_details:
                                traversal_details_list.append(traversal_details)
                            
                            # 设置锚点分数
                            if traversal_details:
                                traversal_details["anchor"]["score"] = anchor_score
                            
                            # 处理扩展结果
                            for score, idx in expanded_results:
                                if idx == -1:
                                    continue
                                
                                chunk = chunks[idx]
                                parent_page = next(page for page in pages if page["page"] == chunk["page"])
                                
                                if return_parent_pages:
                                    key = (sha1, "page", parent_page["page"])
                                    text = parent_page["text"]
                                    page_id = parent_page["page"]
                                else:
                                    key = (sha1, "chunk", idx)
                                    text = chunk["text"]
                                    page_id = chunk["page"]
                                
                                # 对于 SSG，重新计算 query-to-chunk 相似度
                                if retrieval_method == "ssg":
                                    candidate_vec = self._get_vector_by_id(vector_db, idx)
                                    if candidate_vec is not None:
                                        query_vec_flat = embedding_array.flatten()
                                        candidate_vec_flat = candidate_vec.flatten()
                                        query_sim = np.dot(query_vec_flat, candidate_vec_flat)
                                        vector_similarity = round(float(query_sim) * inner_factor, 4)
                                    else:
                                        vector_similarity = round(float(score) * inner_factor, 4)
                                    retrieval_source = "ssg"
                                else:  # triangulation
                                    # Triangulation: 使用质心分数，或者重新计算 query-to-chunk 相似度
                                    candidate_vec = self._get_vector_by_id(vector_db, idx)
                                    if candidate_vec is not None:
                                        query_vec_flat = embedding_array.flatten()
                                        candidate_vec_flat = candidate_vec.flatten()
                                        query_sim = np.dot(query_vec_flat, candidate_vec_flat)
                                        vector_similarity = round(float(query_sim) * inner_factor, 4)
                                    else:
                                        vector_similarity = round(float(score) * inner_factor, 4)
                                    retrieval_source = "triangulation"
                                
                                local_hits.append((key, page_id, text, vector_similarity, sha1, retrieval_source))
                        else:
                            # 如果扩展失败，至少返回锚点
                            if anchor_idx is not None:
                                chunk = chunks[anchor_idx]
                                parent_page = next(page for page in pages if page["page"] == chunk["page"])
                                
                                if return_parent_pages:
                                    key = (sha1, "page", parent_page["page"])
                                    text = parent_page["text"]
                                    page_id = parent_page["page"]
                                else:
                                    key = (sha1, "chunk", anchor_idx)
                                    text = chunk["text"]
                                    page_id = chunk["page"]
                                
                                vector_similarity = round(float(anchor_score) * inner_factor, 4)
                                retrieval_source = retrieval_method  # "ssg" or "triangulation"
                                local_hits.append((key, page_id, text, vector_similarity, sha1, retrieval_source))
                    else:
                        # 锚点查找失败，返回空
                        return []
                elif retrieval_method == "hybrid_expansion":
                    # Hybrid Expansion: Basic Retrieval -> Top-K -> SSG扩展(Top-10) + Triangulation扩展(Top-20)
                    basic_top_k = 50  # 可配置参数
                    distances, indices = vector_db.search(x=embedding_array, k=min(basic_top_k, len(chunks)))
                    
                    basic_results = []
                    basic_keys_set = set()  # 用于快速检查chunk是否在basic Top-50中
                    # 用于追踪算法特定的召回结果（仅新发现的chunk）
                    ssg_new_chunks = []  # SSG新发现的chunk（不在basic Top-50中）
                    tri_new_chunks = []  # Triangulation新发现的chunk（不在basic Top-50中）
                    
                    for distance, index in zip(distances[0], indices[0]):
                        vector_similarity = round(float(distance)*inner_factor, 4)
                        chunk = chunks[index]
                        parent_page = next(page for page in pages if page["page"] == chunk["page"])
                        
                        if return_parent_pages:
                            key = (sha1, "page", parent_page["page"])
                            text = parent_page["text"]
                            page_id = parent_page["page"]
                        else:
                            key = (sha1, "chunk", index)
                            text = chunk["text"]
                            page_id = chunk["page"]
                        
                        basic_results.append((key, page_id, text, vector_similarity, sha1, index))
                        basic_keys_set.add(key)  # 记录basic Top-50的keys
                        local_hits.append((key, page_id, text, vector_similarity, sha1, "basic"))
                    
                    # 对Top-10进行SSG扩展
                    ssg_top_k = 10
                    ssg_total_expanded = 0  # 统计SSG扩展的总数
                    ssg_new_only = 0  # 统计仅由SSG召回的chunk数
                    for key, page_id, text, vector_similarity, sha1, idx in basic_results[:ssg_top_k]:
                        chunk_vec = self._get_vector_by_id(vector_db, idx)
                        if chunk_vec is not None:
                            ssg_result = self._ssg_search(
                                vector_db, idx, chunk_vec, 
                                max_hops=max_hops, neighbor_k=neighbor_k
                            )
                            if ssg_result and isinstance(ssg_result, dict):
                                expanded_results = ssg_result.get("results", [])
                                self._safe_print(f"[DEBUG] SSG扩展: anchor page={page_id}, 扩展结果数={len(expanded_results)}")
                                for score, expanded_idx in expanded_results:
                                    if expanded_idx == -1:
                                        continue
                                    
                                    expanded_chunk = chunks[expanded_idx]
                                    expanded_parent_page = next(page for page in pages if page["page"] == expanded_chunk["page"])
                                    
                                    if return_parent_pages:
                                        expanded_key = (sha1, "page", expanded_parent_page["page"])
                                        expanded_text = expanded_parent_page["text"]
                                        expanded_page_id = expanded_parent_page["page"]
                                    else:
                                        expanded_key = (sha1, "chunk", expanded_idx)
                                        expanded_text = expanded_chunk["text"]
                                        expanded_page_id = expanded_chunk["page"]
                                    
                                    # 检查这个chunk是否已经在basic_results中（避免重复标记）
                                    # 如果已经在basic_results中，我们仍然添加它，但标记为"ssg"，这样在聚合时会正确显示方法多样性
                                    # 重新计算 query-to-chunk 相似度
                                    candidate_vec = self._get_vector_by_id(vector_db, expanded_idx)
                                    if candidate_vec is not None:
                                        query_vec_flat = embedding_array.flatten()
                                        candidate_vec_flat = candidate_vec.flatten()
                                        query_sim = np.dot(query_vec_flat, candidate_vec_flat)
                                        expanded_similarity = round(float(query_sim) * inner_factor, 4)
                                    else:
                                        expanded_similarity = round(float(score) * inner_factor, 4)
                                    
                                    # 无论chunk是否在basic Top-50中，都添加为"ssg"，这样聚合时才能正确显示方法多样性
                                    # 如果已经在basic中，聚合逻辑会合并（basic + ssg = 2种方法）
                                    local_hits.append((expanded_key, expanded_page_id, expanded_text, expanded_similarity, sha1, "ssg"))
                                    ssg_total_expanded += 1
                                    
                                    # 只有当这个chunk不在basic的Top-50中时，才记录为"新发现的chunk"（用于算法贡献分析）
                                    if expanded_key not in basic_keys_set:
                                        # 新发现的chunk，记录到ssg_new_chunks（用于算法贡献分析）
                                        ssg_new_chunks.append({
                                            "key": expanded_key,
                                            "page": expanded_page_id,
                                            "text": expanded_text,
                                            "vector_similarity": expanded_similarity,
                                            "source_sha1": sha1,
                                            "anchor_page": page_id,  # 从哪个anchor扩展而来
                                            "score": score  # SSG内部得分
                                        })
                                        ssg_new_only += 1
                                        self._safe_print(f"[DEBUG] SSG新发现chunk: page={expanded_page_id}, similarity={expanded_similarity:.4f}, anchor={page_id}")
                                    else:
                                        self._safe_print(f"[DEBUG] SSG扩展chunk（已在Basic Top-50中）: page={expanded_page_id}, similarity={expanded_similarity:.4f}, anchor={page_id}")
                                    # 如果expanded_key在basic_keys_set中，说明它也在basic Top-50中，聚合时会显示方法多样性（basic + ssg）
                    
                    self._safe_print(f"[DEBUG] SSG扩展统计: 总扩展数={ssg_total_expanded}, 仅SSG召回的chunk数={ssg_new_only}, 已在Basic Top-50中的chunk数={ssg_total_expanded - ssg_new_only}")
                    
                    # 对Top-20进行Triangulation扩展
                    tri_top_k = 20
                    tri_total_expanded = 0  # 统计Triangulation扩展的总数
                    tri_new_only = 0  # 统计仅由Triangulation召回的chunk数
                    for key, page_id, text, vector_similarity, sha1, idx in basic_results[:tri_top_k]:
                        chunk_vec = self._get_vector_by_id(vector_db, idx)
                        if chunk_vec is not None:
                            tri_result = self._triangulation_search(
                                vector_db, embedding_array, idx, chunk_vec,
                                max_hops=max_hops, neighbor_k=neighbor_k
                            )
                            if tri_result and isinstance(tri_result, dict):
                                expanded_results = tri_result.get("results", [])
                                for score, expanded_idx in expanded_results:
                                    if expanded_idx == -1:
                                        continue
                                    
                                    expanded_chunk = chunks[expanded_idx]
                                    expanded_parent_page = next(page for page in pages if page["page"] == expanded_chunk["page"])
                                    
                                    if return_parent_pages:
                                        expanded_key = (sha1, "page", expanded_parent_page["page"])
                                        expanded_text = expanded_parent_page["text"]
                                        expanded_page_id = expanded_parent_page["page"]
                                    else:
                                        expanded_key = (sha1, "chunk", expanded_idx)
                                        expanded_text = expanded_chunk["text"]
                                        expanded_page_id = expanded_chunk["page"]
                                    
                                    # 检查这个chunk是否已经在basic_results中（避免重复标记）
                                    # 如果已经在basic_results中，我们仍然添加它，但标记为"triangulation"，这样在聚合时会正确显示方法多样性
                                    # 重新计算 query-to-chunk 相似度
                                    candidate_vec = self._get_vector_by_id(vector_db, expanded_idx)
                                    if candidate_vec is not None:
                                        query_vec_flat = embedding_array.flatten()
                                        candidate_vec_flat = candidate_vec.flatten()
                                        query_sim = np.dot(query_vec_flat, candidate_vec_flat)
                                        expanded_similarity = round(float(query_sim) * inner_factor, 4)
                                    else:
                                        expanded_similarity = round(float(score) * inner_factor, 4)
                                    
                                    # 无论chunk是否在basic Top-50中，都添加为"triangulation"，这样聚合时才能正确显示方法多样性
                                    # 如果已经在basic中，聚合逻辑会合并（basic + triangulation = 2种方法）
                                    local_hits.append((expanded_key, expanded_page_id, expanded_text, expanded_similarity, sha1, "triangulation"))
                                    tri_total_expanded += 1
                                    
                                    # 只有当这个chunk不在basic的Top-50中时，才记录为"新发现的chunk"（用于算法贡献分析）
                                    if expanded_key not in basic_keys_set:
                                        # 新发现的chunk，记录到tri_new_chunks（用于算法贡献分析）
                                        tri_new_chunks.append({
                                            "key": expanded_key,
                                            "page": expanded_page_id,
                                            "text": expanded_text,
                                            "vector_similarity": expanded_similarity,
                                            "source_sha1": sha1,
                                            "anchor_page": page_id,  # 从哪个anchor扩展而来
                                            "score": score  # Triangulation内部得分（质心得分）
                                        })
                                        tri_new_only += 1
                                        self._safe_print(f"[DEBUG] Triangulation新发现chunk: page={expanded_page_id}, similarity={expanded_similarity:.4f}, anchor={page_id}")
                                    else:
                                        self._safe_print(f"[DEBUG] Triangulation扩展chunk（已在Basic Top-50中）: page={expanded_page_id}, similarity={expanded_similarity:.4f}, anchor={page_id}")
                                    # 如果expanded_key在basic_keys_set中，说明它也在basic Top-50中，聚合时会显示方法多样性（basic + triangulation）
                    
                    self._safe_print(f"[DEBUG] Triangulation扩展统计: 总扩展数={tri_total_expanded}, 仅Triangulation召回的chunk数={tri_new_only}, 已在Basic Top-50中的chunk数={tri_total_expanded - tri_new_only}")
                
                else:
                    # 未知的检索方法，回退到 basic
                    self._safe_print(f"[WARNING] Unknown retrieval_method '{retrieval_method}', falling back to basic")
                    distances, indices = vector_db.search(x=embedding_array, k=actual_top_n)
                
                    for distance, index in zip(distances[0], indices[0]):
                        vector_similarity = round(float(distance)*inner_factor, 4)
                        chunk = chunks[index]
                        parent_page = next(page for page in pages if page["page"] == chunk["page"])
                        
                        if return_parent_pages:
                            key = (sha1, "page", parent_page["page"])
                            text = parent_page["text"]
                            page_id = parent_page["page"]
                        else:
                            key = (sha1, "chunk", index)
                            text = chunk["text"]
                            page_id = chunk["page"]
                        
                        local_hits.append((key, page_id, text, vector_similarity, sha1, "basic"))
                
            except Exception as e:
                self._safe_print(f"[ERROR] Vector search failed for query '{query_text[:60]}' in report {report.get('name')}: {e}")
            
            # 如果使用了新算法且有遍历详情，返回字典；否则返回列表（保持兼容）
            # 如果是hybrid_expansion，即使没有traversal_details，也要返回字典以包含algorithm_specific_results
            if traversal_details_list or retrieval_method == "hybrid_expansion":
                result_dict = {
                    "hits": local_hits
                }
                # 如果有遍历详情，添加到字典中
                if traversal_details_list:
                    result_dict["traversal_details"] = traversal_details_list[0] if len(traversal_details_list) == 1 else traversal_details_list
                
                # 如果是hybrid_expansion，添加算法特定的召回信息
                if retrieval_method == "hybrid_expansion":
                    result_dict["algorithm_specific_results"] = {
                        "ssg_new_chunks": ssg_new_chunks,  # SSG新发现的chunk（不在basic Top-50中）
                        "triangulation_new_chunks": tri_new_chunks,  # Triangulation新发现的chunk（不在basic Top-50中）
                        "basic_count": len(basic_results),  # Basic检索的数量
                        "ssg_stats": {  # SSG扩展统计
                            "total_expanded": ssg_total_expanded,  # 总扩展数
                            "new_only": ssg_new_only,  # 仅SSG召回的chunk数
                            "in_basic_top50": ssg_total_expanded - ssg_new_only  # 已在Basic Top-50中的chunk数
                        },
                        "triangulation_stats": {  # Triangulation扩展统计
                            "total_expanded": tri_total_expanded,  # 总扩展数
                            "new_only": tri_new_only,  # 仅Triangulation召回的chunk数
                            "in_basic_top50": tri_total_expanded - tri_new_only  # 已在Basic Top-50中的chunk数
                        }
                    }
                    self._safe_print(f"[DEBUG] 返回algorithm_specific_results: ssg_stats={result_dict['algorithm_specific_results']['ssg_stats']}, tri_stats={result_dict['algorithm_specific_results']['triangulation_stats']}")
                return result_dict
            return local_hits

        total_tasks = len(query_embeddings) * num_reports
        max_workers = min(self.parallel_workers, total_tasks) if total_tasks > 0 else 1
        vector_search_start = time.time()

        # 收集所有 traversal_details（每个 query-document 对可能有一个）
        all_traversal_details = []
        # 收集算法特定的召回信息（仅hybrid_expansion）
        all_algorithm_specific_results = {
            "ssg_new_chunks": [],
            "triangulation_new_chunks": [],
            "basic_count": 0,
            "ssg_stats": {
                "total_expanded": 0,
                "new_only": 0,
                "in_basic_top50": 0
            },
            "triangulation_stats": {
                "total_expanded": 0,
                "new_only": 0,
                "in_basic_top50": 0
            }
        }
        
        with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
            # 创建一个映射来追踪每个future对应的查询文本
            future_to_query = {}
            for report in matching_reports:
                for q_text, embedding_array in query_embeddings.items():
                    future = executor.submit(process_query_for_document, report, q_text, embedding_array)
                    future_to_query[future] = q_text
            
            for future in as_completed(future_to_query.keys()):
                query_text = future_to_query[future]  # 获取对应的查询文本
                result = future.result()
                if not result:
                    continue
                
                # 提取 hits 和 traversal_details
                if isinstance(result, dict):
                    doc_hits = result.get("hits", [])
                    trav_details = result.get("traversal_details", None)
                    if trav_details:
                        all_traversal_details.append(trav_details)
                    # 提取算法特定的召回信息（仅hybrid_expansion）
                    algo_results = result.get("algorithm_specific_results", None)
                    if algo_results:
                        all_algorithm_specific_results["ssg_new_chunks"].extend(algo_results.get("ssg_new_chunks", []))
                        all_algorithm_specific_results["triangulation_new_chunks"].extend(algo_results.get("triangulation_new_chunks", []))
                        all_algorithm_specific_results["basic_count"] += algo_results.get("basic_count", 0)
                        
                        # 累加SSG统计
                        ssg_stats = algo_results.get("ssg_stats", {})
                        if ssg_stats:
                            all_algorithm_specific_results["ssg_stats"]["total_expanded"] += ssg_stats.get("total_expanded", 0)
                            all_algorithm_specific_results["ssg_stats"]["new_only"] += ssg_stats.get("new_only", 0)
                            all_algorithm_specific_results["ssg_stats"]["in_basic_top50"] += ssg_stats.get("in_basic_top50", 0)
                            self._safe_print(f"[DEBUG] 累加SSG统计: 当前文档={ssg_stats}, 累加后总计={all_algorithm_specific_results['ssg_stats']}")
                        
                        # 累加Triangulation统计
                        tri_stats = algo_results.get("triangulation_stats", {})
                        if tri_stats:
                            all_algorithm_specific_results["triangulation_stats"]["total_expanded"] += tri_stats.get("total_expanded", 0)
                            all_algorithm_specific_results["triangulation_stats"]["new_only"] += tri_stats.get("new_only", 0)
                            all_algorithm_specific_results["triangulation_stats"]["in_basic_top50"] += tri_stats.get("in_basic_top50", 0)
                            self._safe_print(f"[DEBUG] 累加Triangulation统计: 当前文档={tri_stats}, 累加后总计={all_algorithm_specific_results['triangulation_stats']}")
                        
                        self._safe_print(f"[DEBUG] 收集到算法特定结果: basic_count={algo_results.get('basic_count', 0)}, ssg_new={len(algo_results.get('ssg_new_chunks', []))}, tri_new={len(algo_results.get('triangulation_new_chunks', []))}")
                else:
                    # 兼容旧格式（元组列表）
                    doc_hits = result
                
                if not doc_hits:
                    continue
                
                # 记录这个查询检索到的chunk数量
                self._safe_print(f"[DEBUG] 查询 '{query_text[:80]}...' 检索到 {len(doc_hits)} 个chunks")
                
                with aggregation_lock:
                    # 处理每个hit（支持5元组和6元组格式，向后兼容）
                    for hit in doc_hits:
                        if len(hit) == 5:
                            # 旧格式（向后兼容）：(key, page_id, text, vector_similarity, sha1)
                            key, page_id, text, vector_similarity, sha1 = hit
                            retrieval_source = "basic"  # 默认值
                        elif len(hit) == 6:
                            # 新格式：(key, page_id, text, vector_similarity, sha1, retrieval_source)
                            key, page_id, text, vector_similarity, sha1, retrieval_source = hit
                        else:
                            self._safe_print(f"[ERROR] Unexpected hit format with {len(hit)} elements: {hit}")
                            continue
                        
                        if key not in aggregated_results:
                            aggregated_results[key] = {
                                "page": page_id,
                                "text": text,
                                "similarities": [vector_similarity],  # 改名：distances -> similarities
                                "query_sources": [query_text],  # 记录命中来源
                                "retrieval_sources": [retrieval_source],  # 新增：记录检索方法来源
                                "count": 1,
                                "source_sha1": sha1
                            }
                            # 调试：记录新chunk的初始方法（仅对SSG和Triangulation）
                            if retrieval_source in ["ssg", "triangulation"]:
                                self._safe_print(f"[DEBUG] 聚合: 新chunk page={page_id}, 初始方法={retrieval_source} (仅由{retrieval_source}召回)")
                        else:
                            # 追加相似度分数（用于计算最大值）
                            aggregated_results[key]["similarities"].append(vector_similarity)
                            
                            # 只有当这个检索方法还没有被记录时，才添加到retrieval_sources
                            # 这是为了避免同一个chunk被同一个方法多次命中时重复添加
                            if retrieval_source not in aggregated_results[key]["retrieval_sources"]:
                                old_methods = aggregated_results[key]["retrieval_sources"].copy()
                                aggregated_results[key]["retrieval_sources"].append(retrieval_source)
                                new_methods = aggregated_results[key]["retrieval_sources"]
                                
                                # 调试：记录方法添加（仅对SSG和Triangulation，或方法数量变化时）
                                if retrieval_source in ["ssg", "triangulation"] or len(new_methods) >= 2:
                                    method_change = f"{old_methods} -> {new_methods}"
                                    self._safe_print(f"[DEBUG] 聚合: chunk page={page_id}, 添加方法={retrieval_source}, 方法变化={method_change}")
                            
                            # 只有当这个查询还没有被计数时，才增加count
                            # 这是为了避免 return_parent_pages=True 时，同一个查询命中同一page的多个chunks导致重复计数
                            if query_text not in aggregated_results[key]["query_sources"]:
                                aggregated_results[key]["count"] += 1
                                aggregated_results[key]["query_sources"].append(query_text)

        timing_info['vector_search'] = time.time() - vector_search_start
        
        # 添加聚合结果的详细调试信息
        self._safe_print(f"[DEBUG] ========== 聚合结果统计 ==========")
        self._safe_print(f"[DEBUG] 总共聚合了 {len(aggregated_results)} 个唯一的chunk")
        
        # 统计命中次数分布
        hit_count_distribution = {}
        high_hit_chunks = []  # 命中次数>=3的chunks
        
        for key, info in aggregated_results.items():
            count = info["count"]
            hit_count_distribution[count] = hit_count_distribution.get(count, 0) + 1
            if count >= 3:
                high_hit_chunks.append({
                    "key": key,
                    "page": info["page"],
                    "count": count,
                    "similarities": info["similarities"],
                    "query_sources": info.get("query_sources", [])
                })
        
        self._safe_print(f"[DEBUG] 命中次数分布:")
        for count in sorted(hit_count_distribution.keys(), reverse=True):
            self._safe_print(f"[DEBUG]   - 命中{count}次: {hit_count_distribution[count]}个chunks")
        
        if high_hit_chunks:
            self._safe_print(f"[DEBUG] 高命中chunks (>=3次):")
            for chunk in high_hit_chunks[:10]:  # 只显示前10个
                self._safe_print(f"[DEBUG]   - Page {chunk['page']}: 命中{chunk['count']}次")
                self._safe_print(f"[DEBUG]      得分: {chunk['similarities']}")
                if chunk['query_sources']:
                    self._safe_print(f"[DEBUG]      查询来源数: {len(chunk['query_sources'])}")
                    for i, q in enumerate(chunk['query_sources'][:5], 1):  # 只显示前5个查询
                        self._safe_print(f"[DEBUG]        {i}. {q[:80]}...")
        self._safe_print(f"[DEBUG] ==================================")
    
        # 新的聚合策略：保留最大相似度 + 查询命中数奖励 + 方法多样性奖励
        def calculate_final_similarity(info):
            # 1. 保留最大相似度（优先信任直接相关的方法）
            base_similarity = max(info["similarities"])
            
            # 2. 查询命中数奖励（保持当前逻辑）
            query_bonus = 1.0 + 0.2 * (info["count"] - 1)
            
            # 3. 方法多样性奖励
            # 注意：如果retrieval_sources为空，说明这个chunk可能有问题，但我们不应该默认假设它是"basic"
            raw_sources = info.get("retrieval_sources", [])
            if not raw_sources:
                self._safe_print(f"[WARNING] Chunk page={info['page']} 没有retrieval_sources，这可能是个bug")
                raw_sources = ["basic"]  # 降级处理
            unique_methods = set(raw_sources)
            method_diversity_bonus = 1.0
            if len(unique_methods) >= 2:
                # 被2种以上方法命中，给予奖励
                method_diversity_bonus = 1.0 + 0.1 * (len(unique_methods) - 1)
                # 2种方法：1.1，3种方法：1.2
            
            # 4. 最终得分
            final_similarity = base_similarity * query_bonus * method_diversity_bonus
            return final_similarity, base_similarity, unique_methods
    
        final_results = []
        # 记录所有新发现的page（仅由SSG或Triangulation召回）
        new_pages_only_ssg = []
        new_pages_only_tri = []
        for idx, (key, info) in enumerate(aggregated_results.items()):
            final_similarity, base_similarity, unique_methods = calculate_final_similarity(info)
            
            # 记录新发现的page（仅由SSG或Triangulation召回，不在Basic Top-50中）
            if len(unique_methods) == 1:
                if "ssg" in unique_methods:
                    new_pages_only_ssg.append((info["page"], info["source_sha1"], final_similarity))
                elif "triangulation" in unique_methods:
                    new_pages_only_tri.append((info["page"], info["source_sha1"], final_similarity))
            
            # 调试：检查retrieval_sources（仅在前20个chunk中输出，避免日志过多）
            if idx < 20:
                raw_sources = info.get("retrieval_sources", ["basic"])
                methods_str = ", ".join(sorted(unique_methods))
                method_count = len(unique_methods)
                
                # 根据方法数量分类显示
                if method_count == 1:
                    method_type = "单独召回"
                    if "ssg" in unique_methods:
                        method_type += " (仅SSG)"
                    elif "triangulation" in unique_methods:
                        method_type += " (仅Triangulation)"
                    elif "basic" in unique_methods:
                        method_type += " (仅Basic)"
                elif method_count == 2:
                    method_type = "两种方法组合"
                    if "basic" in unique_methods and "ssg" in unique_methods:
                        method_type += " (Basic+SSG)"
                    elif "basic" in unique_methods and "triangulation" in unique_methods:
                        method_type += " (Basic+Triangulation)"
                    elif "ssg" in unique_methods and "triangulation" in unique_methods:
                        method_type += " (SSG+Triangulation)"
                elif method_count == 3:
                    method_type = "三种方法联合 (Basic+SSG+Triangulation)"
                else:
                    method_type = f"{method_count}种方法"
                
                self._safe_print(f"[DEBUG] Chunk #{idx+1} page={info['page']}, 方法类型={method_type}, 方法列表=[{methods_str}], raw_sources={raw_sources}")
            
            final_results.append({
                "vector_similarity": round(final_similarity, 4),  # 最终向量相似度得分（用于排序）
                "max_original_similarity": round(base_similarity, 4),  # 原始向量相似度最高分
                "page": info["page"],
                "text": info["text"],
                "hit_count": info["count"],  # 命中次数
                "retrieval_sources": list(unique_methods),  # 新增：方法来源列表（已去重）
                "source_sha1": info["source_sha1"],  # Include source document
                "query_sources": info.get("query_sources", [])  # 查询来源（用于调试）
            })
    
        # 聚合：按加权后的相似度降序，取前 top_n（vector_similarity越大越相关）
        final_results = sorted(final_results, key=lambda x: x["vector_similarity"], reverse=True)
        
        # 调试：显示新发现的page统计
        if new_pages_only_ssg or new_pages_only_tri:
            self._safe_print(f"[DEBUG] 新发现的Page统计: SSG={len(new_pages_only_ssg)}, Triangulation={len(new_pages_only_tri)}")
            if new_pages_only_tri:
                self._safe_print(f"[DEBUG] Triangulation新发现的Page (前10个): {new_pages_only_tri[:10]}")
                # 检查所有新发现的page是否在final_results中
                found_count = 0
                not_found_count = 0
                for page, sha1, sim in new_pages_only_tri:
                    rank_in_results = next((idx for idx, r in enumerate(final_results) if r.get("page") == page and r.get("source_sha1") == sha1), None)
                    if rank_in_results is not None:
                        found_count += 1
                        # 只打印前20个的详细信息，避免日志过多
                        if found_count <= 20:
                            self._safe_print(f"[DEBUG] Page {page} (SHA1={sha1}, sim={sim:.4f}) 在final_results中: ✅ 排名 #{rank_in_results+1}")
                    else:
                        not_found_count += 1
                        # 只打印前10个未找到的详细信息
                        if not_found_count <= 10:
                            self._safe_print(f"[DEBUG] Page {page} (SHA1={sha1}, sim={sim:.4f}) 在final_results中: ❌ 未找到")
                self._safe_print(f"[DEBUG] Triangulation新发现的Page统计: 在final_results中={found_count}, 未找到={not_found_count}, 总计={len(new_pages_only_tri)}")
        
        # 保存截断前的完整结果（用于显示"初始召回结果"）
        all_initial_results = final_results.copy()  # 扩展后的全部结果（截断前）
        
        # Debug: 显示聚合后的文档分布
        source_distribution = {}
        for res in final_results[:top_n]:
            source = res.get("source_sha1", "Unknown")
            source_distribution[source] = source_distribution.get(source, 0) + 1
        print(f"[DEBUG] Top {top_n} results distribution: {source_distribution}")
        print(f"[DEBUG] 扩展后的全部结果数量: {len(all_initial_results)}, 截断后数量: {top_n}")
        
        final_results = final_results[:top_n]  # 截断：只取前top_n进入reranker


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
        # 收集遍历详情（如果有）
        retrieval_details = None
        if all_traversal_details:
            retrieval_details = {
                "method": retrieval_method,
                "traversal_info": all_traversal_details[0] if len(all_traversal_details) == 1 else all_traversal_details,
                "max_hops": max_hops,
                "neighbor_k": neighbor_k
            }
        
        # 如果是hybrid_expansion，添加算法特定的召回信息
        algorithm_contribution = None
        if retrieval_method == "hybrid_expansion":
            # 即使 basic_count 为 0，也返回统计信息（可能是多文档查询导致）
            basic_count = all_algorithm_specific_results.get("basic_count", 0)
            ssg_new_chunks = all_algorithm_specific_results.get("ssg_new_chunks", [])
            tri_new_chunks = all_algorithm_specific_results.get("triangulation_new_chunks", [])
            
            algorithm_contribution = {
                "basic_retrieval_count": basic_count,
                "ssg_new_chunks_count": len(ssg_new_chunks),
                "triangulation_new_chunks_count": len(tri_new_chunks),
                "ssg_new_chunks": ssg_new_chunks,
                "triangulation_new_chunks": tri_new_chunks,
                "ssg_stats": all_algorithm_specific_results.get("ssg_stats", {
                    "total_expanded": 0,
                    "new_only": 0,
                    "in_basic_top50": 0
                }),
                "triangulation_stats": all_algorithm_specific_results.get("triangulation_stats", {
                    "total_expanded": 0,
                    "new_only": 0,
                    "in_basic_top50": 0
                })
            }
            self._safe_print(f"[DEBUG] Hybrid Expansion统计: Basic={basic_count}, SSG新发现={len(ssg_new_chunks)}, Triangulation新发现={len(tri_new_chunks)}")
            final_ssg_stats = all_algorithm_specific_results.get("ssg_stats", {})
            final_tri_stats = all_algorithm_specific_results.get("triangulation_stats", {})
            self._safe_print(f"[DEBUG] 最终统计信息: ssg_stats={final_ssg_stats}, tri_stats={final_tri_stats}")
            self._safe_print(f"[DEBUG] algorithm_contribution 已生成: {algorithm_contribution is not None}, keys={list(algorithm_contribution.keys()) if algorithm_contribution else []}")
        
        return {
            'results': final_results,  # 截断后的结果（进入reranker）
            'timing': timing_info,
            'expansion_texts': expansion_texts,
            'retrieval_details': retrieval_details,
            'initial_retrieval_results': all_initial_results,  # 初始召回结果：扩展后的全部结果（截断前）
            'algorithm_contribution': algorithm_contribution  # 算法贡献统计（仅hybrid_expansion）
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
                    "vector_similarity": 0.5,
                    "page": page["page"],
                    "text": page["text"],
                    "source_sha1": sha1  # Track which report this page comes from
                }
                all_pages.append(result)
            
        return all_pages