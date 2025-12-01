import json
from typing import Union, Dict, List, Optional
import re
from pathlib import Path
from src.retrieval import VectorRetriever, HybridRetriever
from src.api_requests import APIProcessor
from tqdm import tqdm
import pandas as pd
import threading
import concurrent.futures
import os


class QuestionsProcessor:
    def __init__(
        self,
        vector_db_dir: Union[str, Path] = './vector_dbs',
        documents_dir: Union[str, Path] = './documents',
        questions_file_path: Optional[Union[str, Path]] = None,
        new_challenge_pipeline: bool = False,
        subset_path: Optional[Union[str, Path]] = None,
        parent_document_retrieval: bool = True,
        llm_reranking: bool = False,
        llm_reranking_sample_size: int = 20,
        top_n_retrieval: int = 10,
        parallel_requests: int = 10,
        api_provider: str = "qwen",
        answering_model: str = "gpt-4o-2024-08-06",
        full_context: bool = False,
        use_hyde: bool = True,
        use_multi_query: bool = True,
        expand_upstream: bool = False,
        expand_top_k: int = 5,
        expand_context_size: int = 2,
        multi_query_methods: Optional[Dict[str, bool]] = None
    ):
        print(f"[QuestionsProcessor] LLM provider: {api_provider}")
        print(f"[QuestionsProcessor] Answering model: {answering_model}")
        #os.environ["LLM_RERANK_MODEL"] = answering_model
        self.questions = self._load_questions(questions_file_path)
        self.documents_dir = Path(documents_dir)
        self.vector_db_dir = Path(vector_db_dir)
        self.subset_path = Path(subset_path) if subset_path else None
        
        self.new_challenge_pipeline = new_challenge_pipeline
        self.return_parent_pages = parent_document_retrieval
        self.llm_reranking = llm_reranking
        self.llm_reranking_sample_size = llm_reranking_sample_size
        self.top_n_retrieval = top_n_retrieval
        self.answering_model = answering_model
        self.parallel_requests = parallel_requests
        self.api_provider = api_provider
        self.openai_processor = APIProcessor(provider=api_provider)
        self.full_context = full_context
        self.use_hyde = use_hyde
        self.use_multi_query = use_multi_query
        self.expand_upstream = expand_upstream
        self.expand_top_k = expand_top_k
        self.expand_context_size = expand_context_size
        self.multi_query_methods = multi_query_methods or {
            'synonym': True,
            'subquestion': True,
            'variant': True
        }
        #print(f"[DEBUG][QuestionsProcessor.__init__] use_hyde={self.use_hyde}, use_multi_query={self.use_multi_query}")

        self.answer_details = []
        self.detail_counter = 0
        self._lock = threading.Lock()

    def _load_questions(self, questions_file_path: Optional[Union[str, Path]]) -> List[Dict[str, str]]:
        if questions_file_path is None:
            return []
        with open(questions_file_path, 'r', encoding='utf-8') as file:
            return json.load(file)

    def _format_retrieval_results(self, retrieval_results) -> str:
        """Format vector retrieval results into RAG context string"""
        if not retrieval_results:
            return ""
        
        context_parts = []
        for result in retrieval_results:
            page_number = result['page']
            text = result['text']
            context_parts.append(f'Text retrieved from page {page_number}: \n"""\n{text}\n"""')
            
        return "\n\n---\n\n".join(context_parts)

    def _expand_adjacent_pages(self, core_pages: list, context_size: int = 2) -> dict:
        """
        扩充核心页面的相邻页面。
        
        Args:
            core_pages: LLM选用的核心页面列表
            context_size: 上下扩充的页数（默认2页）
        
        Returns:
            {
                'core_pages': [10, 25],  # 原始核心页面
                'expanded_pages': [8,9,10,11,12,23,24,25,26,27],  # 扩充后的所有页面
                'adjacent_pages': [8,9,11,12,23,24,26,27]  # 纯扩充的页面（不含核心页）
            }
        """
        core_set = set(core_pages)
        expanded_set = set()
        
        # 对每个核心页面，扩充上下N页
        for page in core_pages:
            for offset in range(-context_size, context_size + 1):
                adjacent_page = page + offset
                if adjacent_page > 0:  # 页码必须 > 0
                    expanded_set.add(adjacent_page)
        
        expanded_list = sorted(expanded_set)
        adjacent_only = sorted(expanded_set - core_set)
        
        return {
            'core_pages': sorted(core_set),
            'expanded_pages': expanded_list,
            'adjacent_pages': adjacent_only
        }
    
    def _build_page_groups(self, reranked_results: list, top_k: int, context_size: int) -> list:
        """
        对重排序后的top K页面构造页面组合。
        
        Args:
            reranked_results: 重排序后的结果列表
            top_k: 取前K个页面作为核心页
            context_size: 上下扩充的页数
        
        Returns:
            List of page groups, each group contains:
            {
                'core_page': 10,
                'core_score': 0.92,
                'source_sha1': 'J2020',
                'pages': [8, 9, 10, 11, 12],  # 组合中的所有页面
                'core_index': 2  # 核心页在组合中的索引
            }
        """
        if not reranked_results:
            return []
        
        page_groups = []
        top_results = reranked_results[:top_k]
        
        for group_id, result in enumerate(top_results):
            core_page = result.get('page')
            core_score = result.get('combined_score') or result.get('distance', 0.0)
            source_sha1 = result.get('source_sha1', '')
            
            # 构造组合页面列表（上下各context_size页）
            group_pages = []
            for offset in range(-context_size, context_size + 1):
                page = core_page + offset
                if page > 0:
                    group_pages.append(page)
            
            # 找到核心页在组合中的索引
            core_index = group_pages.index(core_page) if core_page in group_pages else 0
            
            page_groups.append({
                'group_id': group_id,  # 添加组合ID
                'core_page': core_page,
                'core_score': core_score,
                'source_sha1': source_sha1,
                'pages': group_pages,
                'core_index': core_index
            })
        
        print(f"[INFO] 🔄 构造了 {len(page_groups)} 个页面组合（top {top_k}）")
        return page_groups
    
    def _load_page_text_from_document(self, source_sha1: str, page_number: int) -> str:
        """
        从原始文档中加载指定页面的文本内容。
        
        Args:
            source_sha1: 文档的SHA1标识
            page_number: 页码（1-based）
        
        Returns:
            页面文本内容，如果找不到则返回空字符串
        """
        import json
        
        # 尝试从 documents_dir 加载文档
        doc_path = self.documents_dir / f"{source_sha1}.json"
        
        if not doc_path.exists():
            print(f"[WARNING] 文档不存在: {doc_path}")
            return ""
        
        try:
            with open(doc_path, 'r', encoding='utf-8') as f:
                document = json.load(f)
            
            # 查找指定页码的内容
            pages = document.get('content', {}).get('pages', [])
            for page in pages:
                if page.get('page') == page_number:
                    return page.get('text', '')
            
            print(f"[WARNING] 页码 {page_number} 在文档 {source_sha1} 中未找到")
            return ""
        except Exception as e:
            print(f"[ERROR] 加载文档 {source_sha1} 失败: {e}")
            return ""
    
    def _identify_selected_groups(self, llm_selected_pages: list, page_groups: list) -> list:
        """
        识别LLM选用的页面属于哪些组合。
        
        Args:
            llm_selected_pages: LLM选用的页面列表
            page_groups: 所有页面组合列表
        
        Returns:
            LLM选用的组合列表（至少包含一个LLM选用页面的组合）
        """
        selected_set = set(llm_selected_pages)
        selected_groups = []
        
        for group in page_groups:
            # 检查组合中是否有LLM选用的页面
            group_pages = set(group['pages'])
            if group_pages & selected_set:  # 交集非空
                selected_groups.append(group)
        
        print(f"[INFO] 🎯 从 {len(page_groups)} 个组合中识别出 {len(selected_groups)} 个被选用的组合")
        return selected_groups
    
    def _extract_references_from_groups(self, llm_selected_pages: list, company_name: str, 
                                       retrieval_results: list, selected_groups: list) -> list:
        """
        从选用的组合中提取引用，标记核心页和扩充页。
        
        Args:
            llm_selected_pages: LLM直接选用的页面
            company_name: 公司名称
            retrieval_results: 检索结果
            selected_groups: 选用的组合列表
        
        Returns:
            引用列表，包含核心页和扩充页
        """
        # 构建页面到文本的映射
        page_to_result = {}
        for result in retrieval_results:
            page = result.get('page')
            if page:
                page_to_result[page] = result
        
        # 提取所有选用组合的页面
        all_pages = set()
        page_to_group = {}  # 页面到组合的映射
        for group in selected_groups:
            for page in group['pages']:
                all_pages.add(page)
                if page not in page_to_group:
                    page_to_group[page] = group
        
        llm_selected_set = set(llm_selected_pages)
        refs = []
        
        for page in sorted(all_pages):
            is_core = page in llm_selected_set
            group = page_to_group.get(page, {})
            
            # 获取页面内容
            if page in page_to_result:
                result = page_to_result[page]
                sha1 = result.get('source_sha1', '')
                text = result.get('text', '')
            else:
                sha1 = group.get('source_sha1', '')
                text = self._load_page_text_from_document(sha1, page)
            
            refs.append({
                "pdf_sha1": sha1,
                "page_index": page,
                "chunk_text": text,
                "is_expanded": not is_core,  # 非LLM直接选用的就是扩充页
                "group_id": group.get('group_id'),
                "core_page": group.get('core_page'),
                "group_score": group.get('core_score')
            })
        
        print(f"[INFO] 📚 提取了 {len(refs)} 个引用（核心页: {len(llm_selected_set)}, 扩充页: {len(refs) - len(llm_selected_set)}）")
        return refs
    
    def _load_group_chunks(self, page_groups: list, retrieval_results: list) -> list:
        """
        加载页面组合中的所有chunks，从原始文档加载文本内容。
        
        Args:
            page_groups: 页面组合列表
            retrieval_results: 原始检索结果（用于获取已有的文本）
        
        Returns:
            List of chunks with group metadata:
            {
                'page': 10,
                'text': '...',
                'source_sha1': 'J2020',
                'is_core': True,
                'group_score': 0.92,
                'group_id': 0,
                'core_page': 10,
                'distance': 0.85  # 继承核心页的向量得分
            }
        """
        # 构建page -> result的映射，用于获取已有文本
        page_to_result = {}
        for result in retrieval_results:
            page = result.get('page')
            if page:
                page_to_result[page] = result
        
        all_chunks = []
        for group in page_groups:
            group_id = group.get('group_id', 0)  # 使用组合自带的ID
            core_page = group['core_page']
            core_score = group['core_score']
            source_sha1 = group['source_sha1']
            group_pages = group['pages']
            
            for page in group_pages:
                is_core = (page == core_page)
                
                # 尝试从检索结果获取文本
                if page in page_to_result:
                    text = page_to_result[page].get('text', '')
                    vector_score = page_to_result[page].get('distance', 0.0)
                else:
                    # 从原始文档加载
                    text = self._load_page_text_from_document(source_sha1, page)
                    vector_score = core_score  # 继承核心页得分
                
                chunk = {
                    'page': page,
                    'text': text,
                    'source_sha1': source_sha1,
                    'is_core': is_core,
                    'group_score': core_score,  # 组合得分
                    'group_id': group_id,
                    'core_page': core_page,
                    'distance': vector_score,  # 向量得分
                    'relevance_score': None,  # 扩充页面没有LLM相关性得分
                    'combined_score': core_score if is_core else None  # 只有核心页有组合得分
                }
                all_chunks.append(chunk)
        
        # 去重：如果多个组合包含相同页面，保留得分最高的
        unique_chunks = {}
        for chunk in all_chunks:
            page = chunk['page']
            if page not in unique_chunks or chunk['group_score'] > unique_chunks[page]['group_score']:
                unique_chunks[page] = chunk
        
        result_chunks = list(unique_chunks.values())
        print(f"[INFO] 📦 加载了 {len(result_chunks)} 个唯一页面（去重后）")
        return result_chunks

    def _extract_references(self, pages_list: list, company_name: str, retrieval_results: list, expand_adjacent: bool = True, context_size: int = 2) -> list:
        """
        Extract references with correct source SHA1 from retrieval results.
        Uses actual source_sha1 from retrieval results instead of blindly taking first match from CSV.
        
        Args:
            pages_list: LLM选用的核心页面列表
            company_name: 公司名称
            retrieval_results: 检索结果列表
            expand_adjacent: 是否扩充相邻页面（默认True）
            context_size: 扩充的上下文大小（默认上下各2页）
        
        Returns:
            包含核心页面和扩充页面的引用列表，每个引用标记是否为扩充页面
        """
        # Build a mapping from page number to source_sha1
        page_to_sha1 = {}
        page_to_text = {}
        for result in retrieval_results:
            page = result.get('page')
            source_sha1 = result.get('source_sha1', '')
            text = result.get('text', '')
            if page is not None and source_sha1:
                page_to_sha1[page] = source_sha1
                page_to_text[page] = text
        
        # 如果不扩充，使用原有逻辑
        if not expand_adjacent or not pages_list:
            refs = []
            for page in pages_list:
                sha1 = page_to_sha1.get(page, '')
                chunk_text = page_to_text.get(page, '')
                
                refs.append({
                    "pdf_sha1": sha1,
                    "page_index": page,
                    "chunk_text": chunk_text,
                    "is_expanded": False  # 标记为非扩充页面
                })
            return refs
        
        # 扩充相邻页面
        expansion_info = self._expand_adjacent_pages(pages_list, context_size)
        core_pages = set(expansion_info['core_pages'])
        all_pages = expansion_info['expanded_pages']
        
        # 获取主要的source_sha1（假设所有核心页面来自同一文档）
        primary_sha1 = ""
        if pages_list:
            primary_sha1 = page_to_sha1.get(pages_list[0], '')
        
        print(f"[INFO] 📄 扩充参考页面: 核心页面 {expansion_info['core_pages']} -> 扩充后 {all_pages}")
        
        refs = []
        for page in all_pages:
            is_core = page in core_pages
            sha1 = page_to_sha1.get(page, primary_sha1)  # 使用已知SHA1或主SHA1
            
            # 核心页面：从检索结果获取文本
            if is_core:
                chunk_text = page_to_text.get(page, '')
            else:
                # 扩充页面：从原始文档加载文本
                chunk_text = self._load_page_text_from_document(sha1, page)
            
            refs.append({
                "pdf_sha1": sha1,
                "page_index": page,
                "chunk_text": chunk_text,
                "is_expanded": not is_core  # 标记是否为扩充页面
            })
        
        return refs

    def _format_all_retrieved_chunks(self, retrieval_results: list, llm_selected_pages: list, expanded_pages: list = None) -> list:
        """
        Format all retrieved chunks with scores and LLM selection status.
        
        Args:
            retrieval_results: List of retrieval results with scores
            llm_selected_pages: List of pages that LLM selected as relevant (核心页面)
            expanded_pages: List of all expanded pages including adjacent ones (扩充后的所有页面)
        
        Returns:
            List of formatted chunks with metadata
        """
        formatted_chunks = []
        llm_selected_set = set(llm_selected_pages)
        expanded_set = set(expanded_pages) if expanded_pages else set()
        
        for idx, result in enumerate(retrieval_results, 1):
            page = result.get('page')
            text = result.get('text', '')
            source_sha1 = result.get('source_sha1', '')
            
            # 得分构成详细信息
            vector_score = result.get('distance', 0.0)  # 向量相似度得分
            relevance_score = result.get('relevance_score', None)  # LLM相关性得分（重排序时才有）
            combined_score = result.get('combined_score', None)  # 组合得分（重排序时才有）
            reasoning = result.get('reasoning', '')  # LLM推理过程（重排序时才有）
            
            # 判断页面状态
            is_core_selected = page in llm_selected_set  # LLM直接选用的核心页面
            is_expanded = page in expanded_set and not is_core_selected  # 扩充的相邻页面
            
            formatted_chunks.append({
                "rank": idx,
                "page": page,
                "source_sha1": source_sha1,
                "text": text,
                "vector_score": vector_score,
                "relevance_score": relevance_score,
                "combined_score": combined_score,
                "reasoning": reasoning,
                "selected_by_llm": is_core_selected,  # LLM直接选用
                "is_expanded": is_expanded  # 相邻扩充页面
            })
        
        return formatted_chunks

    def _validate_page_references(self, claimed_pages: list, retrieval_results: list, min_pages: int = 2, max_pages: int = 8) -> list:
        """
        Validate that all page numbers mentioned in the LLM's answer are actually from the retrieval results.
        If fewer than min_pages valid references remain, add top pages from retrieval results.
        """
        if claimed_pages is None:
            claimed_pages = []
        
        retrieved_pages = [result['page'] for result in retrieval_results]
        
        validated_pages = [page for page in claimed_pages if page in retrieved_pages]
        
        if len(validated_pages) < len(claimed_pages):
            removed_pages = set(claimed_pages) - set(validated_pages)
            print(f"[DEBUG] [Warning] Removed {len(removed_pages)} hallucinated page references: {removed_pages}")
        
        if len(validated_pages) < min_pages and retrieval_results:
            existing_pages = set(validated_pages)
            
            for result in retrieval_results:
                page = result['page']
                if page not in existing_pages:
                    validated_pages.append(page)
                    existing_pages.add(page)
                    
                    if len(validated_pages) >= min_pages:
                        break
        
        if len(validated_pages) > max_pages:
            print(f"[DEBUG] [Warning] Trimming references from {len(validated_pages)} to {max_pages} pages")
            validated_pages = validated_pages[:max_pages]
        
        return validated_pages

    def get_answer_for_company(self, company_name: str, question: str, schema: str, conversation_history: Optional[List[Dict]] = None, progress_callback=None, selected_years: Optional[List[int]] = None) -> dict:
        """
        Get answer for a company's question with optional conversation history.
        
        Args:
            company_name: Company name
            question: Current question (original, without context)
            schema: Answer schema type
            conversation_history: Optional list of previous Q&A pairs
                Format: [{"question": "...", "answer": "..."}, ...]
            progress_callback: Optional callback function(stage: str, progress: int)
            selected_years: Optional list of years to filter documents (None = all years)
        
        Returns:
            dict: 包含答案和各阶段用时信息
        """
        import time
        
        timing_info = {
            'init_retriever': 0.0,
            'retrieval': 0.0,
            'hyde_expansion': 0.0,
            'multi_query_expansion': 0.0,
            'llm_reranking': 0.0,
            'upstream_expansion': 0.0,
            'format_results': 0.0,
            'generate_answer': 0.0,
            'total_time': 0.0
        }
        
        total_start_time = time.time()
        
        #print(f"[DEBUG][get_answer_for_company] self.use_hyde={self.use_hyde}, self.use_multi_query={self.use_multi_query}")
        
        # 阶段 1: 初始化检索器
        if progress_callback:
            progress_callback("🔍 分析问题中...", 10)
        
        init_start = time.time()
        
        if self.llm_reranking:
            retriever = HybridRetriever(
                vector_db_dir=self.vector_db_dir,
                documents_dir=self.documents_dir,
                use_hyde=self.use_hyde,
                use_multi_query=self.use_multi_query,
                subset_path=self.subset_path,
                parallel_workers=self.parallel_requests,
                multi_query_methods=self.multi_query_methods
            )
        else:
            retriever = VectorRetriever(
                vector_db_dir=self.vector_db_dir,
                documents_dir=self.documents_dir,
                use_hyde=self.use_hyde,
                use_multi_query=self.use_multi_query,
                subset_path=self.subset_path,
                parallel_workers=self.parallel_requests,
                multi_query_methods=self.multi_query_methods
            )
        timing_info['init_retriever'] = time.time() - init_start

        # 阶段 2: 召回相关文档
        if progress_callback:
            progress_callback("📚 召回相关文档中...", 25)
        
        retrieval_start = time.time()
        if self.full_context:
            retrieval_results = retriever.retrieve_all(company_name)
            timing_info['retrieval'] = time.time() - retrieval_start
        else:
            # 只在VectorRetriever时传递use_hyde和use_multi_query
            # 使用原始问题进行检索（不包含对话历史）
            if isinstance(retriever, VectorRetriever):
                # 需要在retrieval.py中跟踪HYDE和Multi-Query的时间
                # 这里先记录总检索时间，具体细分在retrieval.py中处理
                retrieval_results = retriever.retrieve_by_company_name(
                    company_name=company_name,
                    query=question,
                    llm_reranking_sample_size=self.llm_reranking_sample_size,
                    top_n=self.top_n_retrieval,
                    return_parent_pages=self.return_parent_pages,
                    use_hyde=self.use_hyde,
                    use_multi_query=self.use_multi_query,
                    multi_query_config=self.multi_query_methods,
                    progress_callback=progress_callback,
                    selected_years=selected_years
                )
            else:
                # HybridRetriever 也需要传递 progress_callback
                retrieval_results = retriever.retrieve_by_company_name(
                    company_name=company_name,
                    query=question,
                    llm_reranking_sample_size=self.llm_reranking_sample_size,
                    top_n=self.top_n_retrieval,
                    return_parent_pages=self.return_parent_pages,
                    use_hyde=self.use_hyde,
                    use_multi_query=self.use_multi_query,
                    multi_query_config=self.multi_query_methods,
                    progress_callback=progress_callback,
                    selected_years=selected_years
                )
            timing_info['retrieval'] = time.time() - retrieval_start
        
        # 处理检索结果（可能是dict或list）
        expansion_texts = {}
        reranker_stats = {}
        if isinstance(retrieval_results, dict):
            # 提取扩展文本信息
            if 'expansion_texts' in retrieval_results:
                expansion_texts = retrieval_results['expansion_texts']
            # 提取时间信息
            if 'timing' in retrieval_results:
                timing_info.update(retrieval_results['timing'])
            if 'reranker_stats' in retrieval_results:
                reranker_stats = retrieval_results['reranker_stats']
            # 提取结果
            if 'results' in retrieval_results:
                retrieval_results = retrieval_results['results']
        
        if not retrieval_results:
            raise ValueError("No relevant context found")
        
        # 🔄 上游扩充：在重排序后、LLM生成答案前扩充页面组合
        page_groups = []
        if self.expand_upstream and self.llm_reranking:
            if progress_callback:
                progress_callback("🔄 扩充页面组合中...", 60)
            
            upstream_start = time.time()
            # 构造页面组合
            page_groups = self._build_page_groups(
                reranked_results=retrieval_results,
                top_k=self.expand_top_k,
                context_size=self.expand_context_size
            )
            
            # 加载组合中的所有页面内容
            expanded_chunks = self._load_group_chunks(page_groups, retrieval_results)
            
            # 合并到retrieval_results（去重后）
            existing_pages = {r['page'] for r in retrieval_results}
            for chunk in expanded_chunks:
                if chunk['page'] not in existing_pages:
                    retrieval_results.append(chunk)
            
            timing_info['upstream_expansion'] = time.time() - upstream_start
            print(f"[INFO] ✅ 上游扩充完成：{len(retrieval_results)} 个页面（含扩充）")
        
        # 阶段 3: 格式化检索结果（用于页面选择）
        if progress_callback:
            progress_callback("📝 整理检索结果中...", 70)
        
        format_start = time.time()
        all_retrieval_context = self._format_retrieval_results(retrieval_results)
        timing_info['format_results'] = time.time() - format_start
        
        # 构造带对话历史的问题（用于LLM生成答案）
        question_with_context = self._build_contextual_question(question, conversation_history)
        
        # 阶段 3.5: 页面选择（两阶段流程的第一步）
        if progress_callback:
            progress_callback("🎯 LLM选择相关页面中...", 75)
        
        page_selection_start = time.time()
        import src.prompts as prompts
        
        # 使用轻量级提示词进行页面选择
        page_selection_prompt = prompts.PageSelectionPrompt
        page_selection_user_prompt = page_selection_prompt.user_prompt.format(
            question=question,
            context=all_retrieval_context
        )
        
        # 使用轻量级模型进行页面选择（可以更快更便宜）
        selection_model = "qwen-turbo" if self.api_provider == "qwen" else self.answering_model
        
        page_selection_result = self.openai_processor.processor.send_message(
            model=selection_model,
            system_content=page_selection_prompt.system_prompt,
            human_content=page_selection_user_prompt,
            is_structured=True,
            response_format=page_selection_prompt.PageSelectionSchema
        )
        
        # 提取选定的页面
        selected_pages = page_selection_result.get("selected_pages", [])
        selection_reasoning = page_selection_result.get("reasoning", "")
        
        timing_info['page_selection'] = time.time() - page_selection_start
        
        # 验证选定的页面是否在检索结果中
        retrieved_pages = {result.get('page') for result in retrieval_results if result.get('page') is not None}
        validated_selected_pages = [p for p in selected_pages if p in retrieved_pages]
        
        # 如果验证后没有页面，使用前几个检索结果作为后备
        if not validated_selected_pages and retrieval_results:
            print(f"[WARNING] 页面选择结果无效，使用前{min(5, len(retrieval_results))}个检索结果作为后备")
            validated_selected_pages = [r.get('page') for r in retrieval_results[:5] if r.get('page') is not None]
        
        print(f"[INFO] 🎯 页面选择完成：从 {len(retrieval_results)} 个检索结果中选择了 {len(validated_selected_pages)} 个页面")
        if selection_reasoning:
            print(f"[INFO] 📝 选择理由：{selection_reasoning}")
        
        # 过滤检索结果，只保留选定的页面
        filtered_retrieval_results = [
            result for result in retrieval_results 
            if result.get('page') in validated_selected_pages
        ]
        
        # 阶段 4: 格式化选定的检索结果（用于生成答案）
        if progress_callback:
            progress_callback("✍️ 生成最终答案中...", 80)
        
        format_selected_start = time.time()
        rag_context = self._format_retrieval_results(filtered_retrieval_results)
        timing_info['format_selected_results'] = time.time() - format_selected_start
        
        # 获取提示词信息（用于展示）
        system_prompt, response_format, user_prompt_template = self.openai_processor._build_rag_context_prompts(schema)
        formatted_user_prompt = user_prompt_template.format(context=rag_context, question=question_with_context)
        
        answer_start = time.time()
        answer_dict = self.openai_processor.get_answer_from_rag_context(
            question=question_with_context,
            rag_context=rag_context,
            schema=schema,
            model=self.answering_model
        )
        timing_info['generate_answer'] = time.time() - answer_start
        
        timing_info['total_time'] = time.time() - total_start_time
        
        # 将时间信息、提示词信息和扩展文本添加到返回结果中
        if isinstance(answer_dict, dict):
            answer_dict['timing'] = timing_info
            # 保存提示词信息用于展示
            answer_dict['prompt_info'] = {
                'system_prompt': system_prompt,
                'user_prompt': formatted_user_prompt,
                'rag_context': rag_context,
                'question': question_with_context,
                'schema': schema,
                'model': self.answering_model,
                # 保存页面选择信息
                'page_selection': {
                    'selected_pages': validated_selected_pages,
                    'selection_reasoning': selection_reasoning,
                    'all_retrieval_context': all_retrieval_context  # 保存所有检索结果的上下文（用于展示）
                }
            }
            # 保存扩展文本信息用于展示
            answer_dict['expansion_texts'] = expansion_texts
            if reranker_stats:
                answer_dict['reranker_stats'] = reranker_stats
        
        # 防御性检查：确保返回的是字典而不是列表
        if isinstance(answer_dict, list):
            print(f"[WARNING] get_answer_from_rag_context 返回了列表而不是字典: {answer_dict}")
            # 尝试从列表中提取第一个元素，如果是字典的话
            if answer_dict and isinstance(answer_dict[0], dict):
                answer_dict = answer_dict[0]
            else:
                # 创建一个默认的错误响应
                answer_dict = {
                    "final_answer": "解析错误：API返回了意外的数据格式",
                    "detailed_analysis": [],
                    "reasoning_summary": "数据格式错误",
                    "relevant_pages": []
                }
            
            # 补充必要的信息（即使出错也要保留这些信息）
            if isinstance(answer_dict, dict):
                answer_dict['timing'] = timing_info
                answer_dict['prompt_info'] = {
                    'system_prompt': system_prompt,
                    'user_prompt': formatted_user_prompt,
                    'rag_context': rag_context,
                    'question': question_with_context,
                    'schema': schema,
                    'model': self.answering_model,
                    'page_selection': {
                        'selected_pages': validated_selected_pages,
                        'selection_reasoning': selection_reasoning,
                        'all_retrieval_context': all_retrieval_context
                    }
                }
                answer_dict['expansion_texts'] = expansion_texts
                if reranker_stats:
                    answer_dict['reranker_stats'] = reranker_stats
        
        self.response_data = self.openai_processor.response_data
        if self.new_challenge_pipeline:
            # 使用页面选择阶段选定的页面（两阶段流程）
            pages = validated_selected_pages if validated_selected_pages else answer_dict.get("relevant_pages", [])
            # 验证页面是否在原始检索结果中
            validated_pages = self._validate_page_references(pages, retrieval_results)
            
            # 根据扩充模式处理引用
            if self.expand_upstream and page_groups:
                # 上游扩充模式：识别LLM选用的页面属于哪些组合
                selected_groups = self._identify_selected_groups(validated_pages, page_groups)
                
                # 提取所有选用组合的页面作为引用
                all_group_pages = []
                for group in selected_groups:
                    all_group_pages.extend(group['pages'])
                
                # 去重并排序
                expanded_pages = sorted(set(all_group_pages))
                
                print(f"[INFO] 🎯 LLM选用了 {len(selected_groups)} 个组合，共 {len(expanded_pages)} 个页面")
                
                answer_dict["relevant_pages"] = validated_pages
                answer_dict["references"] = self._extract_references_from_groups(
                    validated_pages,
                    company_name,
                    retrieval_results,
                    selected_groups
                )
                answer_dict["selected_groups"] = [
                    {
                        "group_id": g['group_id'],
                        "core_page": g['core_page'],
                        "core_score": g['core_score'],
                        "pages": g['pages']
                    }
                    for g in selected_groups
                ]
            else:
                # 下游扩充模式：原有逻辑
                expansion_info = self._expand_adjacent_pages(validated_pages, context_size=2)
                expanded_pages = expansion_info['expanded_pages']
                
                answer_dict["relevant_pages"] = validated_pages
                answer_dict["references"] = self._extract_references(
                    validated_pages, 
                    company_name, 
                    retrieval_results, 
                    expand_adjacent=True,
                    context_size=2
                )
            
            # 添加所有检索到的chunks信息（包含得分），并标记扩充页面
            answer_dict["all_retrieved_chunks"] = self._format_all_retrieved_chunks(
                retrieval_results, 
                validated_pages,
                expanded_pages
            )
        return answer_dict
    
    def _build_contextual_question(self, current_question: str, conversation_history: Optional[List[Dict]] = None) -> str:
        """
        Build a question with conversation context.
        
        Args:
            current_question: The current question
            conversation_history: List of previous Q&A pairs
        
        Returns:
            Enhanced question with context
        """
        if not conversation_history:
            return current_question
        
        # 构造历史对话上下文
        context_parts = []
        for h in conversation_history:
            q = h.get('question', '')
            a = h.get('answer', 'N/A')
            context_parts.append(f"Q: {q}\nA: {a}")
        
        context_str = "\n\n".join(context_parts)
        
        # 构造增强问题
        enhanced_question = f"""历史对话上下文：
{context_str}

当前问题：{current_question}

请结合上述历史对话的背景信息回答当前问题。如果当前问题使用了指代词（如"它"、"这个"、"该公司"等）或需要对比分析，请参考历史对话内容进行理解和回答。"""
        
        return enhanced_question

    def _extract_companies_from_subset(self, question_text: str) -> list[str]:
        """Extract company names from a question by matching against companies in the subset file."""
        if not hasattr(self, 'companies_df'):
            if self.subset_path is None:
                raise ValueError("subset_path must be provided to use subset extraction")
            self.companies_df = pd.read_csv(self.subset_path)
        
        found_companies = []
        company_names = sorted(self.companies_df['company_name'].unique(), key=len, reverse=True)
        
        #print(f"[DEBUG] Trying to match company in question:{question_text}")
        for company in company_names:
            escaped_company = re.escape(company)
            pattern = rf'{escaped_company}(?:\W|$)'
            if re.search(pattern, question_text, re.IGNORECASE):
                #print(f"[DEBUG] Matched company: '{company}' in question: {question_text}")
                found_companies.append(company)
                question_text = re.sub(pattern, '', question_text, flags=re.IGNORECASE)
        
        return found_companies

    def process_question(self, question: str, schema: str, conversation_history: Optional[List[Dict]] = None):
        """
        Process a question with optional conversation history.
        
        Args:
            question: The question to process
            schema: Answer schema type
            conversation_history: Optional list of previous Q&A pairs
        """
        if self.new_challenge_pipeline:
            extracted_companies = self._extract_companies_from_subset(question)
        else:
            extracted_companies = re.findall(r'"([^"]*)"', question)
        
        if len(extracted_companies) == 0:
            raise ValueError("No company name found in the question.")
        
        if len(extracted_companies) == 1:
            company_name = extracted_companies[0]
            answer_dict = self.get_answer_for_company(
                company_name=company_name, 
                question=question, 
                schema=schema,
                conversation_history=conversation_history
            )
            return answer_dict
        else:
            return self.process_comparative_question(question, extracted_companies, schema)
    
    def _create_answer_detail_ref(self, answer_dict: dict, question_index: int) -> str:
        """Create a reference ID for answer details and store the details"""
        ref_id = f"#/answer_details/{question_index}"
        with self._lock:
            self.answer_details[question_index] = {
                "step_by_step_analysis": answer_dict['step_by_step_analysis'],
                "reasoning_summary": answer_dict['reasoning_summary'],
                "relevant_pages": answer_dict['relevant_pages'],
                "response_data": self.response_data,
                "self": ref_id
            }
        return ref_id

    def _calculate_statistics(self, processed_questions: List[dict], print_stats: bool = False) -> dict:
        """Calculate statistics about processed questions."""
        total_questions = len(processed_questions)
        error_count = sum(1 for q in processed_questions if "error" in q)
        na_count = sum(1 for q in processed_questions if (q.get("value") if "value" in q else q.get("answer")) == "N/A")
        success_count = total_questions - error_count - na_count
        if print_stats:
            print(f"\nFinal Processing Statistics:")
            print(f"Total questions: {total_questions}")
            print(f"Errors: {error_count} ({(error_count/total_questions)*100:.1f}%)")
            print(f"N/A answers: {na_count} ({(na_count/total_questions)*100:.1f}%)")
            print(f"Successfully answered: {success_count} ({(success_count/total_questions)*100:.1f}%)\n")
        
        return {
            "total_questions": total_questions,
            "error_count": error_count,
            "na_count": na_count,
            "success_count": success_count
        }

    def process_questions_list(self, questions_list: List[dict], output_path: str = None, submission_file: bool = False, team_email: str = "", submission_name: str = "", pipeline_details: str = "") -> dict:
        total_questions = len(questions_list)
        # Add index to each question so we know where to write the answer details
        questions_with_index = [{**q, "_question_index": i} for i, q in enumerate(questions_list)]
        self.answer_details = [None] * total_questions  # Preallocate list for answer details
        processed_questions = []
        parallel_threads = self.parallel_requests

        if parallel_threads <= 1:
            for question_data in tqdm(questions_with_index, desc="Processing questions"):
                processed_question = self._process_single_question(question_data)
                processed_questions.append(processed_question)
                if output_path:
                    self._save_progress(processed_questions, output_path, submission_file=submission_file, team_email=team_email, submission_name=submission_name, pipeline_details=pipeline_details)
                print("\n")
        else:
            with tqdm(total=total_questions, desc="Processing questions") as pbar:
                for i in range(0, total_questions, parallel_threads):
                    batch = questions_with_index[i : i + parallel_threads]
                    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_threads) as executor:
                        # executor.map will return results in the same order as the input list.
                        batch_results = list(executor.map(self._process_single_question, batch))
                    processed_questions.extend(batch_results)
                    
                    if output_path:
                        self._save_progress(processed_questions, output_path, submission_file=submission_file, team_email=team_email, submission_name=submission_name, pipeline_details=pipeline_details)
                    pbar.update(len(batch_results))
        
        statistics = self._calculate_statistics(processed_questions, print_stats = True)
        
        return {
            "questions": processed_questions,
            "answer_details": self.answer_details,
            "statistics": statistics
        }

    def _process_single_question(self, question_data: dict) -> dict:
        question_index = question_data.get("_question_index", 0)

        # 兼容新旧格式，优先取非空字段
        question_text = question_data.get("question") or question_data.get("text")
        schema = question_data.get("schema") or question_data.get("kind")

        # 跳过无效问题
        if not isinstance(question_text, str) or not question_text.strip():
            print(f"[WARNING] Skipping invalid question: {question_text}")
            return {"error": "Invalid question text", "question": question_text, "schema": schema}

        try:
            answer_dict = self.process_question(question_text, schema)

            if "error" in answer_dict:
                detail_ref = self._create_answer_detail_ref({
                    "step_by_step_analysis": None,
                    "reasoning_summary": None,
                    "relevant_pages": None
                }, question_index)
                # 保持原有分支逻辑
                if self.new_challenge_pipeline:
                    return {
                        "question_text": question_text,
                        "kind": schema,
                        "value": None,
                        "references": [],
                        "error": answer_dict["error"],
                        "answer_details": {"$ref": detail_ref}
                    }
                else:
                    return {
                        "question": question_text,
                        "schema": schema,
                        "answer": None,
                        "error": answer_dict["error"],
                        "answer_details": {"$ref": detail_ref},
                    }
            detail_ref = self._create_answer_detail_ref(answer_dict, question_index)
            if self.new_challenge_pipeline:
                return {
                    "question_text": question_text,
                    "kind": schema,
                    "value": answer_dict.get("final_answer"),
                    "references": answer_dict.get("references", []),
                    "answer_details": {"$ref": detail_ref}
                }
            else:
                return {
                    "question": question_text,
                    "schema": schema,
                    "answer": answer_dict.get("final_answer"),
                    "answer_details": {"$ref": detail_ref},
                }
        except Exception as err:
            return self._handle_processing_error(question_text, schema, err, question_index)

    def _handle_processing_error(self, question_text: str, schema: str, err: Exception, question_index: int) -> dict:
        """
        Handle errors during question processing.
        Log error details and return a dictionary containing error information.
        """
        import traceback
        error_message = str(err)
        tb = traceback.format_exc()
        error_ref = f"#/answer_details/{question_index}"
        error_detail = {
            "error_traceback": tb,
            "self": error_ref
        }
        
        with self._lock:
            self.answer_details[question_index] = error_detail
        
        print(f"Error encountered processing question: {question_text}")
        print(f"Error type: {type(err).__name__}")
        print(f"Error message: {error_message}")
        print(f"Full traceback:\n{tb}\n")
        
        if self.new_challenge_pipeline:
            return {
                "question_text": question_text,
                "kind": schema,
                "value": None,
                "references": [],
                "error": f"{type(err).__name__}: {error_message}",
                "answer_details": {"$ref": error_ref}
            }
        else:
            return {
                "question": question_text,
                "schema": schema,
                "answer": None,
                "error": f"{type(err).__name__}: {error_message}",
                "answer_details": {"$ref": error_ref},
            }

    def _post_process_submission_answers(self, processed_questions: List[dict]) -> List[dict]:
        """
        Post-process answers for submission format:
        1. Convert page indices from one-based to zero-based
        2. Clear references for N/A answers
        3. Format answers according to submission schema
        4. Include step_by_step_analysis from answer details
        """
        #print(f"[DEBUG] Post-process references")
        submission_answers = []
        
        for q in processed_questions:
            question_text = q.get("question_text") or q.get("question")
            kind = q.get("kind") or q.get("schema")
            value = "N/A" if "error" in q else (q.get("value") if "value" in q else q.get("answer"))
            references = q.get("references", [])
            
            answer_details_ref = q.get("answer_details", {}).get("$ref", "")
            step_by_step_analysis = None
            if answer_details_ref and answer_details_ref.startswith("#/answer_details/"):
                try:
                    index = int(answer_details_ref.split("/")[-1])
                    if 0 <= index < len(self.answer_details) and self.answer_details[index]:
                        step_by_step_analysis = self.answer_details[index].get("step_by_step_analysis")
                except (ValueError, IndexError):
                    pass


            # DON'T Clear references if value is N/A
            if value == "N/A" and False:
                references = []
            else:
                # Convert page indices from one-based to zero-based,并保留chunk_text字段
                
                references = [
                    {
                        "pdf_sha1": ref["pdf_sha1"],
                        "page_index": ref["page_index"] - 1,
                        "chunk_text": ref.get("chunk_text", "")
                    }
                    for ref in references
                ]
            
            submission_answer = {
                "question_text": question_text,
                "kind": kind,
                "value": value,
                "references": references,
            }
            
            if step_by_step_analysis:
                submission_answer["reasoning_process"] = step_by_step_analysis
            
            submission_answers.append(submission_answer)
        
        return submission_answers

    def _save_progress(self, processed_questions: List[dict], output_path: Optional[str], submission_file: bool = False, team_email: str = "", submission_name: str = "", pipeline_details: str = ""):
        #print(f"[DEBUG] Save progress to {output_path}")
        if output_path:
            statistics = self._calculate_statistics(processed_questions)
            
            # Prepare debug content
            result = {
                "questions": processed_questions,
                "answer_details": self.answer_details,
                "statistics": statistics
            }
            output_file = Path(output_path)
            debug_file = output_file.with_name(output_file.stem + "_debug" + output_file.suffix)
            with open(debug_file, 'w', encoding='utf-8') as file:
                json.dump(result, file, ensure_ascii=False, indent=2)
            
            if submission_file:
                # Post-process answers for submission
                submission_answers = self._post_process_submission_answers(processed_questions)
                submission = {
                    "answers": submission_answers,
                    "team_email": team_email,
                    "submission_name": submission_name,
                    "details": pipeline_details
                }
                with open(output_file, 'w', encoding='utf-8') as file:
                    json.dump(submission, file, ensure_ascii=False, indent=2)

    def process_all_questions(self, output_path: str = 'questions_with_answers.json', team_email: str = "79250515615@yandex.com", submission_name: str = "Ilia_Ris SO CoT + Parent Document Retrieval", submission_file: bool = False, pipeline_details: str = ""):
        result = self.process_questions_list(
            self.questions,
            output_path,
            submission_file=submission_file,
            team_email=team_email,
            submission_name=submission_name,
            pipeline_details=pipeline_details
        )
        return result

    def process_comparative_question(self, question: str, companies: List[str], schema: str) -> dict:
        """
        Process a question involving multiple companies in parallel:
        1. Rephrase the comparative question into individual questions
        2. Process each individual question using parallel threads
        3. Combine results into final comparative answer
        """
        # Step 1: Rephrase the comparative question
        rephrased_questions = self.openai_processor.get_rephrased_questions(
            original_question=question,
            companies=companies
        )
        
        individual_answers = {}
        aggregated_references = []
        
        # Step 2: Process each individual question in parallel
        def process_company_question(company: str) -> tuple[str, dict]:
            """Helper function to process one company's question and return (company, answer)"""
            sub_question = rephrased_questions.get(company)
            if not sub_question:
                raise ValueError(f"Could not generate sub-question for company: {company}")
            
            answer_dict = self.get_answer_for_company(
                company_name=company, 
                question=sub_question, 
                schema="number"
            )
            return company, answer_dict

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_company = {
                executor.submit(process_company_question, company): company 
                for company in companies
            }
            
            for future in concurrent.futures.as_completed(future_to_company):
                try:
                    company, answer_dict = future.result()
                    individual_answers[company] = answer_dict
                    
                    company_references = answer_dict.get("references", [])
                    aggregated_references.extend(company_references)
                except Exception as e:
                    company = future_to_company[future]
                    print(f"Error processing company {company}: {str(e)}")
                    raise
        
        # Remove duplicate references
        unique_refs = {}
        for ref in aggregated_references:
            key = (ref.get("pdf_sha1"), ref.get("page_index"))
            unique_refs[key] = ref
        aggregated_references = list(unique_refs.values())
        
        # Step 3: Get the comparative answer using all individual answers
        comparative_answer = self.openai_processor.get_answer_from_rag_context(
            question=question,
            rag_context=individual_answers,
            schema="comparative",
            model=self.answering_model
        )
        self.response_data = self.openai_processor.response_data
        
        comparative_answer["references"] = aggregated_references
        return comparative_answer
    