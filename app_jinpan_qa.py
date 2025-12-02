#!/usr/bin/env python3
"""
金盘科技 RAG 问答系统 - Streamlit 前端
基于 val_jinpan_colab.ipynb 的交互式本地前端
"""

import streamlit as st
import sys
import os
from pathlib import Path
import json
from datetime import datetime
import traceback
import pandas as pd
from typing import List
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None
from PIL import Image
import io

# 预配置 API Keys
os.environ["DASHSCOPE_API_KEY"] = "sk-6a44d15e56dd4007945ccc41b97b499c"
os.environ["GOOGLE_API_KEY"] = "AIzaSyA4pIV3SB-OWYfGZoZjDM_8dbU6Zycpaz8"

# 添加项目根目录到Python路径
root_dir = Path(__file__).parent
sys.path.insert(0, str(root_dir))

from src.questions_processing import QuestionsProcessor
from src.api_requests import APIProcessor

# 加载 benchmark 标准答案映射
@st.cache_data
def load_benchmark_answers(benchmark_path: str) -> dict:
    """
    加载 benchmark CSV，建立 问题 -> 标准答案 的映射
    """
    import csv
    import re
    mapping = {}
    try:
        with open(benchmark_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                question = row.get('问题', '').strip()
                answer = row.get('标准回答', '').strip()
                if question and answer:
                    # 清理问题文本用于匹配
                    question_clean = re.sub(r'\s+', ' ', question)
                    mapping[question_clean] = answer
    except Exception as e:
        st.warning(f"加载 benchmark 失败: {e}")
    return mapping

# 从 questions.csv 或 benchmark 中获取标准答案
@st.cache_data
def get_standard_answer(question: str, questions_df: pd.DataFrame = None, benchmark_map: dict = None) -> str:
    """
    获取问题的标准答案
    优先从 questions.csv 的"标准回答"列获取，如果没有则从 benchmark 中匹配
    """
    import re
    
    # 清理问题文本
    question_clean = re.sub(r'\s+', ' ', question.strip())
    
    # 1. 先从 questions.csv 中查找
    if questions_df is not None:
        for idx, row in questions_df.iterrows():
            if question_clean == re.sub(r'\s+', ' ', str(row.get('提问内容', '')).strip()):
                standard_answer = row.get('标准回答', '')
                if standard_answer and str(standard_answer).strip():
                    return str(standard_answer).strip()
    
    # 2. 从 benchmark 中匹配
    if benchmark_map:
        # 精确匹配
        if question_clean in benchmark_map:
            return benchmark_map[question_clean]
        
        # 模糊匹配（去除标点符号）
        question_normalized = re.sub(r'[^\w]', '', question_clean)
        for bq, answer in benchmark_map.items():
            bq_normalized = re.sub(r'[^\w]', '', bq)
            if question_normalized == bq_normalized:
                return answer
    
    return ""

# 加载 subset.csv 映射（SHA1 -> 文档信息）
@st.cache_data
def load_document_mapping(subset_path: str) -> dict:
    """
    加载 subset.csv，建立 SHA1 -> {company_name, year} 的映射
    """
    import csv
    mapping = {}
    try:
        with open(subset_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                sha1 = row.get('sha1', '')
                company_name = row.get('company_name', '')
                year = row.get('year', '')
                if sha1:
                    mapping[sha1] = {
                        'company_name': company_name,
                        'year': year,
                        'display_name': f"{company_name} {year}年报" if year else company_name
                    }
    except Exception as e:
        st.error(f"加载 subset.csv 失败: {e}")
    return mapping

# 获取可用年份列表
@st.cache_data
def get_available_years(subset_path: str, company_name: str) -> List[int]:
    """
    从 subset.csv 获取指定公司的所有可用年份
    
    Args:
        subset_path: subset.csv 文件路径
        company_name: 公司名称
    
    Returns:
        排序后的年份列表
    """
    import csv
    years = set()
    try:
        with open(subset_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('company_name', '') == company_name:
                    year_str = row.get('year', '').strip()
                    if year_str:
                        try:
                            years.add(int(year_str))
                        except ValueError:
                            pass
    except Exception as e:
        st.warning(f"获取可用年份失败: {e}")
    return sorted(list(years))

# 页面配置
st.set_page_config(
    page_title="金盘科技 RAG 问答系统",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS
st.markdown("""
    <style>
    /* 增加侧边栏宽度 */
    [data-testid="stSidebar"] {
        min-width: 400px;
        max-width: 450px;
    }
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .answer-box {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border: 1px solid #dee2e6;
        color: #212529;
    }
    .question-box {
        background-color: #cfe2ff;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #0d6efd;
        color: #052c65;
    }
    .reference-box {
        background-color: #fff3cd;
        padding: 0.8rem;
        border-radius: 0.3rem;
        margin: 0.3rem 0;
        font-size: 0.9rem;
        border: 1px solid #ffecb5;
        color: #664d03;
    }
    /* 改善按钮对比度 */
    .stButton > button {
        border: 1px solid #dee2e6;
    }
    /* 改善文本框对比度 */
    .stTextInput > div > div > input {
        background-color: #ffffff;
        border: 2px solid #ced4da;
        color: #212529;
    }
    /* Tab 标签样式优化 */
    .stTabs [data-baseweb="tab-list"] button {
        color: #495057;
        font-weight: 500;
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        color: #0d6efd;
    }
    /* 成功/警告/信息框对比度增强 */
    .stSuccess {
        background-color: #d1e7dd;
        color: #0a3622;
    }
    .stWarning {
        background-color: #fff3cd;
        color: #664d03;
    }
    .stInfo {
        background-color: #cfe2ff;
        color: #052c65;
    }
    </style>
""", unsafe_allow_html=True)

# 初始化session state
if 'processor' not in st.session_state:
    st.session_state.processor = None
if 'history' not in st.session_state:
    st.session_state.history = []
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
if 'current_question' not in st.session_state:
    st.session_state.current_question = ""
if 'current_schema' not in st.session_state:
    st.session_state.current_schema = "jingpan"
if 'example_clicked' not in st.session_state:
    st.session_state.example_clicked = False
if 'widget_key_counter' not in st.session_state:
    st.session_state.widget_key_counter = 0
if 'enable_multi_turn' not in st.session_state:
    st.session_state.enable_multi_turn = False  # 默认关闭多轮对话
if 'context_turns' not in st.session_state:
    st.session_state.context_turns = 3  # 默认保留3轮历史
if 'flow_step_selector' not in st.session_state:
    st.session_state.flow_step_selector = 'overview'

def initialize_system():
    """初始化RAG问答系统"""
    try:
        root_path = Path("data/val_set")
        company_name = "金盘科技"
        
        # 检查数据库是否存在
        vector_db_dir = root_path / "databases" / "vector_dbs"
        documents_dir = root_path / "databases" / "chunked_reports"
        subset_path = root_path / "subset.csv"
        
        if not documents_dir.exists() or not vector_db_dir.exists():
            st.error("❌ 数据库不存在！请先运行 main.py 处理 PDF 文件")
            return False
        
        # 获取配置
        config = st.session_state.config
        
        with st.spinner("🔧 正在初始化问答系统..."):
            processor = QuestionsProcessor(
                vector_db_dir=vector_db_dir,
                documents_dir=documents_dir,
                questions_file_path=None,
                new_challenge_pipeline=True,
                subset_path=subset_path,
                parent_document_retrieval=True,
                llm_reranking=config['llm_reranking'],
                llm_reranking_sample_size=config.get('rerank_sample_size', 50),
                top_n_retrieval=config['top_n_retrieval'],
                parallel_requests=config.get('parallel_requests', 4),
                api_provider=config['api_provider'],
                answering_model=config['answering_model'],
                full_context=False,
                use_hyde=config['use_hyde'],
                use_multi_query=config['use_multi_query'],
                multi_query_methods=config.get('multi_query_methods'),
                expand_upstream=config.get('expand_upstream', False),
                expand_top_k=config.get('expand_top_k', 5),
                expand_context_size=config.get('expand_context_size', 1)
            )
            
            st.session_state.processor = processor
            st.session_state.company_name = company_name
            st.session_state.initialized = True
            
        return True
    except Exception as e:
        st.error(f"❌ 初始化失败: {str(e)}")
        with st.expander("查看详细错误"):
            st.code(traceback.format_exc())
        return False

def get_pdf_page_image(pdf_path: str, page_num: int, dpi: int = 150):
    """
    从PDF提取指定页码的图片（使用PyMuPDF）
    
    Args:
        pdf_path: PDF文件路径
        page_num: 页码索引（**0-based**，第1页=0，第2页=1，以此类推）
        dpi: 图片分辨率（实际使用zoom参数）
    
    Returns:
        PIL Image对象或None
    """
    if fitz is None:
        st.warning("⚠️ PyMuPDF未安装，无法显示PDF页面图片")
        return None
    
    try:
        # 打开PDF文档
        doc = fitz.open(pdf_path)
        
        # 检查页码是否有效
        if page_num < 0 or page_num >= len(doc):
            st.warning(f"⚠️ 页码 {page_num} 超出范围 (总页数: {len(doc)})")
            return None
        
        # 获取指定页面
        page = doc[page_num]
        
        # 设置缩放比例 (dpi/72，因为PDF默认是72dpi)
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        
        # 渲染页面为图片
        pix = page.get_pixmap(matrix=mat)
        
        # 转换为PIL Image
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        
        doc.close()
        return img
        
    except Exception as e:
        st.warning(f"⚠️ 无法提取PDF页面图片: {str(e)}")
        return None

def format_answer_display(answer_dict: dict, question: str = ""):
    """格式化并显示答案"""
    # 获取答案
    answer = answer_dict.get("final_answer", answer_dict.get("answer", "N/A"))
    
    # 获取标准答案
    standard_answer = ""
    if question:
        try:
            questions_df = pd.read_csv("data/val_set/questions_selected_100.csv")
            benchmark_map = load_benchmark_answers("金盘财报查询场景问题benchmark-原先的表格.csv")
            standard_answer = get_standard_answer(question, questions_df, benchmark_map)
        except Exception as e:
            st.warning(f"获取标准答案失败: {e}")
    
    # 获取计时信息
    timing = answer_dict.get("timing", {})
    
    # 主答案 - 使用更明显的对比色
    st.markdown("### 📊 答案")
    
    # 并排显示RAG答案和标准答案
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🤖 RAG生成的答案**")
        st.markdown(f'<div class="answer-box"><h3 style="color: #0d6efd; margin-top: 0; margin-bottom: 0;">💡 {answer}</h3></div>', 
                unsafe_allow_html=True)
    
    with col2:
        st.markdown("**✅ 标准答案**")
        if standard_answer:
            st.markdown(f'<div class="answer-box" style="background-color: #d1e7dd;"><h3 style="color: #0a3622; margin-top: 0; margin-bottom: 0;">📋 {standard_answer}</h3></div>', 
                        unsafe_allow_html=True)
        else:
            st.info("暂无标准答案")
    
    # 显示计时信息（简洁的指标卡片）
    if timing:
        st.markdown("---")
        st.markdown("### ⏱️ 性能指标")
        
        # 计算关键阶段的用时
        total_time = timing.get("total_time", 0.0)
        retrieval_time = timing.get("retrieval", 0.0)  # 总检索时间（包含HYDE、Multi-Query、向量搜索）
        hyde_time = timing.get("hyde_expansion", 0.0)
        multi_query_time = timing.get("multi_query_expansion", 0.0)
        vector_search_time = timing.get("vector_search", 0.0)
        llm_reranking_time = timing.get("llm_reranking", 0.0)
        generate_answer_time = timing.get("generate_answer", 0.0)
        
        # 向量检索总时间：如果vector_search单独统计，则相加；否则使用retrieval_time
        if vector_search_time > 0:
            vector_retrieval_total = hyde_time + multi_query_time + vector_search_time
        else:
            # vector_search未单独统计，使用总检索时间
            vector_retrieval_total = retrieval_time
        
        # 使用4列布局展示关键指标
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.metric("总用时", f"{total_time:.2f}s" if total_time > 0 else "N/A")
        
        with metric_col2:
            st.metric("向量检索", f"{vector_retrieval_total:.2f}s" if vector_retrieval_total > 0 else "N/A")
        
        with metric_col3:
            if llm_reranking_time > 0:
                st.metric("LLM重排序", f"{llm_reranking_time:.2f}s")
            else:
                st.metric("LLM重排序", "未使用")
        
        with metric_col4:
            st.metric("生成答案", f"{generate_answer_time:.2f}s" if generate_answer_time > 0 else "N/A")
        
        reranker_stats = answer_dict.get("reranker_stats") or timing.get("reranker_stats")
        if reranker_stats:
            st.markdown("#### 🤖 LLM重排序统计")
            stat_cols = st.columns(3)
            success_rate = reranker_stats.get("success_rate", 0.0) * 100
            stat_cols[0].metric("成功率", f"{success_rate:.1f}%")
            stat_cols[1].metric("请求总数", reranker_stats.get("total_requests", 0))
            stat_cols[2].metric(
                "平均LLM耗时",
                f"{reranker_stats.get('avg_llm_latency', 0.0):.2f}s"
            )
            st.caption(
                f"并发上限: {reranker_stats.get('max_concurrent_requests', 'N/A')} | "
                f"QPS限制: {reranker_stats.get('request_rate_limit', 'N/A')} | "
                f"批次回退: {reranker_stats.get('batch_fallbacks', 0)} | "
                f"缺失排名补偿: {reranker_stats.get('missing_rankings', 0)}"
            )
            if reranker_stats.get("last_error"):
                st.info(f"最近错误：{reranker_stats['last_error']}")

        # 可选：使用expander展示更详细的各阶段用时（用于调试）
        with st.expander("📊 查看详细计时信息"):
            timing_df = pd.DataFrame([
                {'阶段': '初始化检索器', '用时(秒)': timing.get('init_retriever', 0.0)},
                {'阶段': 'HYDE扩展', '用时(秒)': timing.get('hyde_expansion', 0.0)},
                {'阶段': 'Multi-Query扩展', '用时(秒)': timing.get('multi_query_expansion', 0.0)},
                {'阶段': '向量搜索', '用时(秒)': timing.get('vector_search', 0.0)},
                {'阶段': '向量检索总时间', '用时(秒)': timing.get('retrieval', 0.0)},
                {'阶段': 'LLM重排序', '用时(秒)': timing.get('llm_reranking', 0.0)},
                {'阶段': '上游扩充', '用时(秒)': timing.get('upstream_expansion', 0.0)},
                {'阶段': '格式化结果', '用时(秒)': timing.get('format_results', 0.0)},
                {'阶段': '生成答案', '用时(秒)': timing.get('generate_answer', 0.0)},
                {'阶段': '总用时', '用时(秒)': timing.get('total_time', 0.0)},
            ])
            st.dataframe(timing_df, use_container_width=True, hide_index=True)
    
    # 创建标签页
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔍 分析过程", "📝 推理总结", "📚 LLM选用的参考", "🗂️ 所有检索结果", "💬 生成提示词"])
    
    with tab1:
        if "step_by_step_analysis" in answer_dict:
            analysis = answer_dict["step_by_step_analysis"]
            if isinstance(analysis, list):
                for i, step in enumerate(analysis, 1):
                    st.markdown(f"**{i}.** {step}")
            else:
                st.write(analysis)
        else:
            st.info("无详细分析")
    
    with tab2:
        if "reasoning_summary" in answer_dict:
            st.write(answer_dict["reasoning_summary"])
        else:
            st.info("无推理总结")
    
    with tab3:
        if "references" in answer_dict and answer_dict["references"]:
            refs = answer_dict["references"]
            
            # 加载文档映射
            doc_mapping = load_document_mapping("data/val_set/subset.csv")
            
            # 统计核心页面和扩充页面
            core_count = sum(1 for ref in refs if not ref.get('is_expanded', False))
            expanded_count = sum(1 for ref in refs if ref.get('is_expanded', False))
            
            st.markdown(f"### 📚 LLM选用的参考资料")
            
            # 检查是否使用上游扩充
            if "selected_groups" in answer_dict:
                # 上游扩充模式：显示组合信息
                selected_groups = answer_dict["selected_groups"]
                st.caption(f"🔄 使用上游扩充模式 | 选用 {len(selected_groups)} 个页面组合 | 共 {len(refs)} 页（核心页: {core_count}，扩充页: {expanded_count}）")
                
                # 显示每个组合
                for group_idx, group in enumerate(selected_groups, 1):
                    core_page = group['core_page']
                    core_score = group['core_score']
                    pages = group['pages']
                    
                    with st.expander(f"📦 组合 {group_idx}: 核心页 {core_page} (得分: {core_score:.4f}) - 包含 {len(pages)} 页", expanded=(group_idx == 1)):
                        st.info(f"📄 页面范围: {pages[0]} - {pages[-1]} | 核心页: {core_page} | 组合得分: {core_score:.4f}")
                        
                        # 显示组合中的页面
                        group_refs = [r for r in refs if r['page_index'] in pages]
                        for ref in group_refs:
                            page_num = ref['page_index']
                            is_core = not ref.get('is_expanded', False)
                            doc_sha1 = ref.get('pdf_sha1', '')
                            
                            if is_core:
                                badge = '⭐ 核心页'
                                color = '#28a745'
                            else:
                                badge = '📍 扩充页'
                                color = '#007bff'
                            
                            st.markdown(f'<span style="background-color: {color}; color: white; padding: 2px 6px; border-radius: 3px; font-size: 0.8em;">{badge}</span> 第 {page_num} 页', unsafe_allow_html=True)
            else:
                # 下游扩充模式：原有显示
                st.caption(f"✅ 核心引用: {core_count}个 | 📍 扩充页面: {expanded_count}个（自动添加相邻页面）")
            
            # 按文档分组并按页码排序
            from collections import defaultdict
            doc_groups = defaultdict(list)
            for ref in refs:
                sha1 = ref.get("pdf_sha1", "")
                page = ref.get("page_index", "N/A")
                chunk_text = ref.get("chunk_text", "")
                is_expanded = ref.get("is_expanded", False)
                group_id = ref.get("group_id")
                core_page = ref.get("core_page")
                group_score = ref.get("group_score")
                if sha1 and page != "N/A":
                    doc_groups[sha1].append({
                        'page': page,
                        'text': chunk_text,
                        'is_expanded': is_expanded,
                        'group_id': group_id,
                        'core_page': core_page,
                        'group_score': group_score
                    })
            
            # 按文档显示，每个文档内部按页码排序
            for doc_sha1, pages_data in doc_groups.items():
                # 获取文档显示名称
                doc_info = doc_mapping.get(doc_sha1, {})
                doc_display_name = doc_info.get('display_name', doc_sha1)
                
                # 按页码排序
                pages_data.sort(key=lambda x: x['page'])
                
                # 统计该文档的核心和扩充页面数
                doc_core = sum(1 for p in pages_data if not p['is_expanded'])
                doc_expanded = sum(1 for p in pages_data if p['is_expanded'])
                
                # 显示文档标题
                st.markdown(f"### 📄 {doc_display_name}")
                st.caption(f"核心引用: {doc_core}个 | 扩充页面: {doc_expanded}个 | 共 {len(pages_data)} 页")
                
                # 为每个页码显示图片和文本
                for idx, page_data in enumerate(pages_data, 1):
                    page_num = page_data['page']
                    chunk_text = page_data['text']
                    is_expanded = page_data['is_expanded']
                    group_id = page_data.get('group_id')
                    core_page = page_data.get('core_page')
                    group_score = page_data.get('group_score')
                    
                    # 根据是否扩充页面使用不同的图标和标签
                    if is_expanded:
                        icon = "📍"
                        badge = '<span style="background-color: #007bff; color: white; padding: 2px 8px; border-radius: 3px; font-size: 0.85em;">📍 相邻扩充</span>'
                        if group_id is not None and core_page is not None:
                            group_info = f" | 组合 {group_id + 1}（核心页: {core_page}）"
                        else:
                            group_info = ""
                    else:
                        icon = "✅"
                        badge = '<span style="background-color: #28a745; color: white; padding: 2px 8px; border-radius: 3px; font-size: 0.85em; font-weight: bold;">✅ LLM核心引用</span>'
                        if group_score is not None:
                            group_info = f" | 组合得分: {group_score:.4f}"
                        else:
                            group_info = ""
                    
                    with st.expander(f"{icon} 引用 {idx}: 第 {page_num} 页{group_info}", expanded=(idx == 1 and not is_expanded)):
                        # 显示页面类型标签
                        st.markdown(badge, unsafe_allow_html=True)
                        st.markdown("")  # 空行
                        
                        # 构建PDF路径
                        pdf_path = Path("data/val_set/pdf_reports") / f"{doc_sha1}.pdf"
                        
                        if pdf_path.exists():
                            # 显示PDF页面图片
                            st.markdown(f"**📖 文档第 {page_num} 页:**")
                            
                            # 提取并显示PDF页面图片
                            # 注意：page_num 是 1-based（第1页、第2页...），但 PyMuPDF 使用 0-based 索引
                            page_image = get_pdf_page_image(str(pdf_path), page_num - 1)
                            if page_image:
                                st.image(page_image, use_container_width=True, caption=f"{doc_sha1} - 页码 {page_num}")
                            else:
                                st.warning("无法加载页面图片")
                        else:
                            st.warning(f"未找到PDF文件: {doc_sha1}.pdf")
                        
                        # 显示文本摘录
                        if chunk_text:
                            st.markdown("**📝 相关文本摘录:**")
                            st.caption(chunk_text[:300] + "..." if len(chunk_text) > 300 else chunk_text)
        else:
            st.info("无引用信息")
        
        # 显示源文档SHA1
        if "source_sha1" in answer_dict:
            st.markdown(f"**📄 主要来源:** `{answer_dict['source_sha1']}`")
    
    with tab4:
        # 显示所有检索到的chunks
        if "all_retrieved_chunks" in answer_dict and answer_dict["all_retrieved_chunks"]:
            all_chunks = answer_dict["all_retrieved_chunks"]
            
            # 加载文档映射
            doc_mapping = load_document_mapping("data/val_set/subset.csv")
            
            st.markdown(f"### 🔎 检索到 {len(all_chunks)} 个相关文本块")
            st.caption("✨ 标记为 **LLM选用** 的是模型最终引用的文本块")
            
            # 统计信息
            llm_selected_count = sum(1 for chunk in all_chunks if chunk.get('selected_by_llm', False))
            
            # 判断是否使用了重排序（如果有combined_score则使用了重排序）
            has_reranking = any(chunk.get('combined_score') is not None for chunk in all_chunks)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("总检索数", len(all_chunks))
            with col2:
                st.metric("LLM选用", llm_selected_count, delta=f"{llm_selected_count}/{len(all_chunks)}")
            with col3:
                if has_reranking:
                    # 过滤掉 None 值，只计算有效的 combined_score
                    valid_scores = [chunk.get('combined_score', 0) for chunk in all_chunks if chunk.get('combined_score') is not None]
                    avg_combined = sum(valid_scores) / len(valid_scores) if valid_scores else 0
                    st.metric("平均组合得分", f"{avg_combined:.4f}")
                else:
                    # 过滤掉 None 值，只计算有效的 vector_score
                    valid_scores = [chunk.get('vector_score', 0) for chunk in all_chunks if chunk.get('vector_score') is not None]
                    avg_vector = sum(valid_scores) / len(valid_scores) if valid_scores else 0
                    st.metric("平均向量得分", f"{avg_vector:.4f}")
            with col4:
                if has_reranking:
                    st.info("✅ 使用了LLM重排序")
                else:
                    st.info("📊 纯向量检索")
            
            st.markdown("---")
            
            # 按得分排序显示
            for chunk in all_chunks:
                rank = chunk.get('rank', 0)
                page = chunk.get('page', 'N/A')
                source_sha1 = chunk.get('source_sha1', '')
                text = chunk.get('text', '')
                vector_score = chunk.get('vector_score', 0.0)
                relevance_score = chunk.get('relevance_score', None)
                combined_score = chunk.get('combined_score', None)
                reasoning = chunk.get('reasoning', '')
                selected = chunk.get('selected_by_llm', False)
                is_expanded = chunk.get('is_expanded', False)  # 是否为扩充的相邻页面
                
                # 获取文档显示名称
                doc_info = doc_mapping.get(source_sha1, {})
                doc_display_name = doc_info.get('display_name', source_sha1)
                
                # 根据页面状态，使用不同的样式
                if selected:
                    icon = "⭐"
                    badge = '<span style="background-color: #28a745; color: white; padding: 2px 8px; border-radius: 3px; font-size: 0.85em; font-weight: bold;">✅ LLM核心引用</span>'
                    border_color = "#28a745"
                elif is_expanded:
                    icon = "📍"
                    badge = '<span style="background-color: #007bff; color: white; padding: 2px 8px; border-radius: 3px; font-size: 0.85em;">📍 相邻扩充</span>'
                    border_color = "#007bff"
                else:
                    icon = "📄"
                    badge = '<span style="background-color: #6c757d; color: white; padding: 2px 8px; border-radius: 3px; font-size: 0.85em;">未选用</span>'
                    border_color = "#dee2e6"
                
                # 构建显示的得分信息
                if combined_score is not None:
                    score_display = f"组合得分: {combined_score:.4f}"
                else:
                    score_display = f"向量得分: {vector_score:.4f}"
                
                # 标记文本
                status_text = ""
                if selected:
                    status_text = "⭐"
                elif is_expanded:
                    status_text = "📍"
                
                # 显示每个chunk
                with st.expander(
                    f"{icon} 排名 #{rank} - {doc_display_name} 第{page}页 - {score_display} {status_text}",
                    expanded=(rank == 1 and selected)
                ):
                    # 顶部信息栏
                    st.markdown(f"""
                    <div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin-bottom: 10px; border-left: 4px solid {border_color};">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <strong>📁 文档:</strong> {doc_display_name} | 
                                <strong>📄 页码:</strong> {page} |
                                <strong>🏆 排名:</strong> #{rank}
                            </div>
                            <div>
                                {badge}
                            </div>
                        </div>
                        <div style="margin-top: 8px; font-size: 0.9em;">
                            <strong>🔗 SHA1:</strong> <code>{source_sha1}</code>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # 详细得分构成
                    st.markdown("**📊 得分详情:**")
                    score_cols = st.columns(3)
                    with score_cols[0]:
                        st.metric("向量相似度", f"{vector_score:.6f}", help="基于嵌入向量的语义相似度得分（越高越相似）")
                    with score_cols[1]:
                        if relevance_score is not None:
                            st.metric("LLM相关性", f"{relevance_score:.6f}", help="LLM判断的相关性得分（0-1之间）")
                        else:
                            st.metric("LLM相关性", "未使用", help="未启用LLM重排序")
                    with score_cols[2]:
                        if combined_score is not None:
                            st.metric("组合得分", f"{combined_score:.6f}", help="向量得分与LLM得分的加权组合")
                        else:
                            st.metric("组合得分", "未使用", help="未启用LLM重排序")
                    
                    # LLM推理过程（如果有）
                    if reasoning and selected:
                        st.markdown("**🤔 LLM推理过程:**")
                        st.info(reasoning)
                    
                    # PDF预览（如果是LLM选用的）
                    if selected:
                        pdf_path = Path("data/val_set/pdf_reports") / f"{source_sha1}.pdf"
                        if pdf_path.exists():
                            st.markdown("**📖 PDF页面预览:**")
                            page_image = get_pdf_page_image(str(pdf_path), page - 1)
                            if page_image:
                                st.image(page_image, use_container_width=True, caption=f"{doc_display_name} - 第{page}页")
                    
                    # 文本内容
                    st.markdown("**📝 文本内容:**")
                    st.text_area(
                        "文本",
                        text,
                        height=150,
                        key=f"chunk_{rank}_{page}_{source_sha1}",
                        label_visibility="collapsed"
                    )
        else:
            st.info("无检索结果信息")
    
    with tab5:
        # 显示生成阶段的提示词信息
        if "prompt_info" in answer_dict:
            prompt_info = answer_dict["prompt_info"]
            
            st.markdown("### 💬 LLM生成阶段的提示词")
            st.caption(f"📋 Schema: {prompt_info.get('schema', 'N/A')} | 🤖 Model: {prompt_info.get('model', 'N/A')}")
            
            # 页面选择信息（两阶段流程）
            if "page_selection" in prompt_info:
                page_selection = prompt_info["page_selection"]
                st.markdown("---")
                st.markdown("#### 🎯 页面选择阶段（两阶段流程的第一步）")
                selected_pages = page_selection.get('selected_pages', [])
                selection_reasoning = page_selection.get('selection_reasoning', '')
                all_retrieval_context = page_selection.get('all_retrieval_context', '')
                
                col1, col2 = st.columns(2)
                with col1:
                    # 计算所有检索结果的数量（通过分割 "---" 来估算）
                    total_retrieval_count = len(all_retrieval_context.split('---')) if all_retrieval_context else 0
                    st.metric("📊 总检索数量", total_retrieval_count if total_retrieval_count > 0 else "N/A")
                with col2:
                    st.metric("✅ 选定页面数", len(selected_pages))
                
                if selected_pages:
                    st.markdown(f"**选定的页面：** {', '.join(map(str, selected_pages))}")
                
                if selection_reasoning:
                    st.markdown("**选择理由：**")
                    st.info(selection_reasoning)
                
                # 显示所有检索结果的上下文（用于对比）
                if all_retrieval_context:
                    with st.expander("📋 查看所有检索结果的上下文（页面选择阶段使用）", expanded=False):
                        st.caption(f"这是页面选择阶段看到的所有检索结果（共 {len(all_retrieval_context.split('---'))} 个结果）")
                        st.text_area(
                            "All Retrieval Context",
                            all_retrieval_context,
                            height=400,
                            key="all_retrieval_context_display",
                            label_visibility="collapsed"
                        )
            
            st.markdown("---")
            
            # System Prompt
            st.markdown("#### 📘 System Prompt（系统提示词）")
            st.text_area(
                "System Prompt",
                prompt_info.get('system_prompt', ''),
                height=300,
                key="system_prompt_display",
                label_visibility="collapsed"
            )
            
            st.markdown("---")
            
            # User Prompt
            st.markdown("#### 📝 User Prompt（用户提示词）")
            st.caption("包含完整的上下文信息和问题")
            st.text_area(
                "User Prompt",
                prompt_info.get('user_prompt', ''),
                height=400,
                key="user_prompt_display",
                label_visibility="collapsed"
            )
            
            st.markdown("---")
            
            # RAG Context（仅上下文部分）
            st.markdown("#### 📚 RAG Context（检索到的上下文）")
            if "page_selection" in prompt_info:
                st.caption("这是传递给LLM的上下文信息，仅包含页面选择阶段选定的页面文本（两阶段流程）")
            else:
                st.caption("这是传递给LLM的上下文信息，包含所有检索到的页面文本")
            rag_context = prompt_info.get('rag_context', '')
            if rag_context:
                # 计算上下文长度
                context_length = len(rag_context)
                st.caption(f"上下文长度: {context_length:,} 字符")
                st.text_area(
                    "RAG Context",
                    rag_context,
                    height=500,
                    key="rag_context_display",
                    label_visibility="collapsed"
                )
            else:
                st.info("无上下文信息")
            
            st.markdown("---")
            
            # Question
            st.markdown("#### ❓ Question（问题）")
            st.caption("发送给LLM的完整问题（可能包含对话历史）")
            question = prompt_info.get('question', '')
            st.text_area(
                "Question",
                question,
                height=150,
                key="question_display",
                label_visibility="collapsed"
            )
            
            # 展示扩展文本（HYDE和Multi-Query）
            if "expansion_texts" in answer_dict:
                expansion_texts = answer_dict.get("expansion_texts", {})
                
                st.markdown("---")
                st.markdown("### 🔄 查询扩展生成的文本")
                
                # HYDE扩展文本
                if expansion_texts.get('hyde_text'):
                    st.markdown("#### 🔮 HYDE 扩展（假设答案）")
                    st.caption("HYDE方法生成的假设答案文本，用于增强检索")
                    st.text_area(
                        "HYDE Text",
                        expansion_texts['hyde_text'],
                        height=200,
                        key="hyde_text_display",
                        label_visibility="collapsed"
                    )
                else:
                    st.markdown("#### 🔮 HYDE 扩展")
                    st.info("未启用HYDE扩展")
                
                st.markdown("---")
                
                # Multi-Query扩展文本
                multi_query_texts = expansion_texts.get('multi_query_texts', [])
                mq_methods_used = expansion_texts.get('multi_query_methods', {})
                
                st.markdown("#### 🔄 Multi-Query 扩展（扩展查询）")
                
                # 检查是否启用了 Multi-Query
                if not any(mq_methods_used.values()):
                    st.info("⚪ 未启用 Multi-Query 扩展")
                elif multi_query_texts:
                    st.caption(f"Multi-Query方法生成的扩展查询，共 {len(multi_query_texts)} 个")
                    
                    for idx, mq_item in enumerate(multi_query_texts, 1):
                        method_id = mq_item.get('method_id', idx)
                        query_text = mq_item.get('query', '')
                        
                        method_names = {
                            1: "名词解释",
                            2: "指标拆分",
                            3: "情景变体"
                        }
                        method_name = method_names.get(method_id, f"方法{method_id}")
                        
                        with st.expander(f"📝 {method_name} (方法 {method_id})", expanded=(idx == 1)):
                            st.text_area(
                                f"扩展查询 {idx}",
                                query_text,
                                height=100,
                                key=f"multi_query_{method_id}_{idx}",
                                label_visibility="collapsed"
                            )
                else:
                    # Multi-Query 已启用但没有生成扩展查询（LLM 判断问题已足够清晰）
                    st.info("✅ Multi-Query 已启用，但 LLM 判断当前问题已足够清晰，无需扩展查询")

                # 显示启用的 Multi-Query 方法
                if any(mq_methods_used.values()):
                    label_map = {
                        'synonym': "名词解释",
                        'subquestion': "指标拆分",
                        'variant': "情景变体"
                    }
                    enabled_labels = [label_map[k] for k, v in mq_methods_used.items() if v]
                    if enabled_labels:
                        st.caption("本次启用的扩展方式：" + "、".join(enabled_labels))
                
                # 显示名词解释（Glossary）
                glossary_context = expansion_texts.get('glossary_context')
                if glossary_context:
                    st.markdown("---")
                    st.markdown("##### 📖 Multi-Query 使用的财务名词解释")
                    st.text_area(
                        "Glossary Context",
                        glossary_context,
                        height=220,
                        key="multi_query_glossary_display",
                        label_visibility="collapsed"
                    )
        else:
            st.info("⚠️ 提示词信息不可用（可能是旧版本的答案）")

def save_history():
    """保存问答历史"""
    if st.session_state.history:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"qa_history_{timestamp}.json"
        
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(st.session_state.history, f, ensure_ascii=False, indent=2)
        
        return filename
    return None

def prepare_conversation_history(max_turns: int) -> list:
    """
    准备对话历史用于多轮对话
    
    Args:
        max_turns: 最多保留的历史轮数
    
    Returns:
        格式化的历史对话列表，包含问题和简化的答案
    """
    if not st.session_state.history or not st.session_state.enable_multi_turn:
        return None
    
    # 获取最近的N轮对话
    recent_history = st.session_state.history[-max_turns:] if len(st.session_state.history) > max_turns else st.session_state.history
    
    # 格式化历史记录（提取关键信息）
    formatted_history = []
    for record in recent_history:
        question = record.get('question', '')
        answer_dict = record.get('answer', {})
        
        # 提取关键答案信息（优先使用推理摘要，其次使用最终答案）
        if 'reasoning_summary' in answer_dict and answer_dict['reasoning_summary']:
            answer = answer_dict['reasoning_summary']
        elif 'final_answer' in answer_dict:
            answer = str(answer_dict['final_answer'])
        else:
            answer = 'N/A'
        
        formatted_history.append({
            'question': question,
            'answer': answer
        })
    
    return formatted_history

# ==================== 侧边栏配置 ====================
with st.sidebar:
    st.title("⚙️ 系统配置")
    
    if 'config' not in st.session_state:
        st.session_state.config = {
            'api_provider': 'qwen',
            'answering_model': 'qwen-max',
            'top_n_retrieval': 10,
            'use_hyde': True,
            'use_multi_query': True,
            'llm_reranking': True,
            'rerank_sample_size': 50,
            'expand_upstream': True,
            'expand_top_k': 5,
            'expand_context_size': 1,
            'parallel_requests': 4,
            'multi_query_methods': {
                'synonym': True,
                'subquestion': False,
                'variant': False
            }
        }
    
    flow_steps = [
        {"id": "overview", "label": "流程概览", "icon": "🏁"},
        {"id": "model", "label": "模型配置", "icon": "🤖"},
        {"id": "retrieval", "label": "基础检索", "icon": "⚙️"},
        {"id": "enhancement", "label": "检索增强", "icon": "🚀"},
        {"id": "rerank", "label": "LLM重排序", "icon": "🎯"},
        {"id": "expansion", "label": "上游扩充", "icon": "🔄"},
        {"id": "data", "label": "数据与多轮", "icon": "📅"},
    ]
    flow_options = [step["id"] for step in flow_steps]
    current_step = st.session_state.get("flow_step_selector", flow_options[0])
    selected_step = st.radio(
        "流程节点",
        options=flow_options,
        index=flow_options.index(current_step),
        format_func=lambda x: next(step["label"] for step in flow_steps if step["id"] == x),
        key="flow_step_selector",
        label_visibility="collapsed"
    )
    
    st.markdown("""
    <style>
    .flow-container {display:flex;flex-direction:column;gap:6px;margin-bottom:12px;}
    .flow-step {border:1px solid #e1e6ef;border-radius:10px;padding:6px 12px;background:#f8f9fc;color:#495057;font-weight:500;display:flex;align-items:center;gap:8px;}
    .flow-step.active {background:linear-gradient(90deg,#0d6efd,#6ea8fe);color:#fff;border-color:#0d6efd;box-shadow:0 4px 10px rgba(13,110,253,0.2);}
    .flow-arrow {text-align:center;color:#adb5bd;}
    </style>
    """, unsafe_allow_html=True)
    
    flow_html = "<div class='flow-container'>"
    for idx, step in enumerate(flow_steps):
        active_class = "active" if step["id"] == selected_step else ""
        flow_html += f"<div class='flow-step {active_class}'>{step['icon']} {step['label']}</div>"
        if idx < len(flow_steps) - 1:
            flow_html += "<div class='flow-arrow'>↓</div>"
    flow_html += "</div>"
    st.markdown(flow_html, unsafe_allow_html=True)
    
    config_defaults = st.session_state.config
    if 'multi_query_methods' not in config_defaults:
        config_defaults['multi_query_methods'] = {
            'synonym': True,
            'subquestion': False,
            'variant': False
        }
    multi_query_methods_defaults = config_defaults['multi_query_methods']
    selected_multi_query_methods = multi_query_methods_defaults.copy()
    api_provider = config_defaults.get('api_provider', 'qwen')
    answering_model = config_defaults.get('answering_model', 'qwen-max')
    top_n_retrieval = config_defaults.get('top_n_retrieval', 10)
    use_hyde = config_defaults.get('use_hyde', True)
    use_multi_query = config_defaults.get('use_multi_query', True)
    llm_reranking = config_defaults.get('llm_reranking', True)
    rerank_sample_size = config_defaults.get('rerank_sample_size', 50)
    expand_upstream = config_defaults.get('expand_upstream', True)
    expand_top_k = config_defaults.get('expand_top_k', 5)
    expand_context_size = config_defaults.get('expand_context_size', 1)
    selected_years = st.session_state.get("selected_years", []) or []
    parallel_requests = config_defaults.get('parallel_requests', 4)
    
    with st.expander("✨ 流程概览 · 推荐配置", expanded=(selected_step == "overview")):
        st.markdown("""
        **🎯 推荐设置（已应用）**
        
        ✅ 检索数量：10  
        ✅ HYDE：开启  
        ✅ Multi-Query：开启  
        ✅ LLM重排序：开启  
        ✅ 初始召回：50  
        ✅ 上游扩充：开启  
        ✅ 核心页面：5  
        ✅ 扩充页数：上下各1页  
        ✅ 多轮对话：关闭  
        
        💡 使用上方流程图可快速跳转至对应步骤进行配置
        """)
    
    with st.expander("🤖 模型配置", expanded=(selected_step == "model")):
        api_provider = st.selectbox(
            "API 提供商",
            options=['qwen', 'openai', 'gemini'],
            index=['qwen', 'openai', 'gemini'].index(api_provider) if api_provider in ['qwen', 'openai', 'gemini'] else 0,
            help="选择大语言模型API提供商"
        )
        
        if api_provider == 'qwen':
            model_options = ['qwen-max', 'qwen-plus', 'qwen-turbo']
        elif api_provider == 'openai':
            model_options = ['gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo']
        else:
            model_options = ['gemini-1.5-pro', 'gemini-1.5-flash']
    
    answering_model = st.selectbox(
        "回答模型",
        options=model_options,
            index=model_options.index(answering_model) if answering_model in model_options else 0,
        help="用于生成答案的模型"
    )
    
    with st.expander("⚙️ 基础检索", expanded=(selected_step == "retrieval")):
        top_n_retrieval = st.slider(
            "📊 最终检索数量",
            min_value=5,
            max_value=30,
            value=top_n_retrieval,
            step=5,
            help="经过重排序后最终传递给LLM的文档块数量"
        )
    
    with st.expander("🚀 检索增强", expanded=(selected_step == "enhancement")):
        use_hyde = st.checkbox(
            "✨ HYDE（假设性文档扩展）",
            value=use_hyde,
            help="生成假设性答案辅助检索，提高语义匹配度",
            key="hyde_checkbox"
        )
        use_multi_query = st.checkbox(
            "🔄 Multi-Query（多查询扩展）",
            value=use_multi_query,
            help="生成多个相关查询并行检索，提高召回率",
            key="multiquery_checkbox"
        )
        if use_multi_query:
            st.markdown("#### 🧩 Multi-Query 扩展方式")
            col_syn, col_sub, col_var = st.columns(3)
            synonym_enabled = col_syn.checkbox(
                "名词解释",
                value=multi_query_methods_defaults.get('synonym', True),
                help="为财务名词补充同义词、定义、计算方式",
                key=f"multiquery_synonym_checkbox_{selected_step}"
            )
            subquestion_enabled = col_sub.checkbox(
                "指标拆分",
                value=multi_query_methods_defaults.get('subquestion', False),
                help="按指标/时间拆分多条子问题",
                key=f"multiquery_sub_checkbox_{selected_step}"
            )
            variant_enabled = col_var.checkbox(
                "情景变体",
                value=multi_query_methods_defaults.get('variant', False),
                help="在问题开放或模糊时生成不同视角的提问",
                key=f"multiquery_variant_checkbox_{selected_step}"
            )
            selected_multi_query_methods = {
                'synonym': synonym_enabled,
                'subquestion': subquestion_enabled,
                'variant': variant_enabled
            }
            if not any(selected_multi_query_methods.values()):
                st.warning("⚠️ 所有扩展方式均已关闭，将仅使用原问题进行检索")
        else:
            selected_multi_query_methods = {
                'synonym': False,
                'subquestion': False,
                'variant': False
            }
    
    with st.expander("🎯 LLM 重排序", expanded=(selected_step == "rerank")):
        llm_reranking = st.checkbox(
            "🧠 启用 LLM 重排序",
            value=llm_reranking,
            help="使用LLM评估相关性并重新排序，显著提高精确度",
            key="llm_rerank_checkbox"
        )
        
        if llm_reranking:
            rerank_sample_size = st.slider(
                "🔍 初始召回数量",
                min_value=20,
                max_value=100,
                value=rerank_sample_size,
                step=10,
                help="LLM重排序前先召回的候选chunks数量（越大越全面但越慢）"
            )
            st.success(f"✅ 重排序流程：召回 **{rerank_sample_size}** → LLM重排 → 返回 **{top_n_retrieval}**")
        else:
            rerank_sample_size = 10
    
    with st.expander("🔄 上游扩充", expanded=(selected_step == "expansion")):
        if llm_reranking:
            expand_upstream = st.checkbox(
                "📈 启用上游扩充",
                value=expand_upstream,
                help="在答案生成前扩充页面组合，让LLM基于更完整的上下文生成高质量答案",
                key="upstream_checkbox"
            )
            if expand_upstream:
                col1, col2 = st.columns(2)
                with col1:
                    expand_top_k = st.slider(
                        "核心页面数",
                        min_value=3,
                        max_value=10,
                        value=expand_top_k,
                        help="选取重排序后的前K个页面作为核心"
                    )
                with col2:
                    expand_context_size = st.slider(
                        "上下扩充页数",
                        min_value=1,
                        max_value=3,
                        value=expand_context_size,
                        help="每个核心页面上下各扩充N页"
                    )
                estimated_pages = expand_top_k * (2 * expand_context_size + 1)
                st.info(f"📊 预计【{estimated_pages}】页上下文，去重后约 20-40 页")
                estimated_tokens = estimated_pages * 800
                if estimated_tokens > 25000:
                    st.error(f"🚨 Token 预估 {estimated_tokens:,}，可能超限，建议降低扩充范围")
                elif estimated_tokens > 15000:
                    st.warning(f"⚠️ Token 预估 {estimated_tokens:,}，响应时间可能较长")
                else:
                    st.success(f"✅ Token 预估 {estimated_tokens:,}，处于安全范围")
            else:
                expand_top_k = 5
                expand_context_size = 1
                st.info("💡 当前使用下游扩充，仅在答案生成后补充引用")
        else:
            expand_upstream = False
            expand_top_k = 5
            expand_context_size = 1
            st.info("⚠️ 请先启用 LLM 重排序以使用上游扩充")
    
    with st.expander("📅 数据与多轮对话", expanded=(selected_step == "data")):
        if st.session_state.initialized:
            subset_path = Path("data/val_set/subset.csv")
            company_name = st.session_state.get("company_name", "金盘科技")
            available_years = get_available_years(str(subset_path), company_name)
            if available_years:
                st.info(f"💡 可用年份: {', '.join(map(str, available_years))}")
                selected_years = st.multiselect(
                    "选择特定年份数据（留空=所有年份）",
                    options=available_years,
                    default=selected_years,
                    help="选择特定年份进行检索；不选则默认全量",
                    key="year_selector"
                )
                st.session_state.selected_years = selected_years if selected_years else None
            else:
                st.warning("⚠️ 无可用年份，默认在所有年份中检索")
                st.session_state.selected_years = None
        else:
            st.info("ℹ️ 系统尚未初始化，暂无法读取年份信息")
        st.session_state.selected_years = None
    
    enable_multi_turn = st.checkbox(
        "启用多轮对话",
            value=st.session_state.enable_multi_turn,
            help="启用后记住上下文，可能增加token消耗",
        key="multi_turn_checkbox"
    )
    st.session_state.enable_multi_turn = enable_multi_turn
    if enable_multi_turn:
        context_turns = st.slider(
            "保留对话轮数",
            min_value=1,
            max_value=10,
            value=st.session_state.context_turns,
            step=1,
            help="设置保留多少轮历史对话作为上下文",
            key="context_turns_slider"
        )
        st.session_state.context_turns = context_turns
        st.info(f"💡 当前保留最近 **{context_turns}** 轮对话作为上下文")
    else:
        st.warning("⚠️ 多轮对话已关闭，每次问答相互独立")
    
    # 检测配置变化
    new_config = {
        'api_provider': api_provider,
        'answering_model': answering_model,
        'top_n_retrieval': top_n_retrieval,
        'use_hyde': use_hyde,
        'use_multi_query': use_multi_query,
        'llm_reranking': llm_reranking,
        'rerank_sample_size': rerank_sample_size,
        'expand_upstream': expand_upstream,
        'expand_top_k': expand_top_k,
        'expand_context_size': expand_context_size,
        'parallel_requests': parallel_requests,
        'multi_query_methods': selected_multi_query_methods
    }
    
    # 如果配置改变且系统已初始化，需要重新初始化
    if st.session_state.initialized and st.session_state.config != new_config:
        st.session_state.initialized = False
        st.session_state.processor = None
        st.warning("⚠️ 检测到配置变化，系统将在下次查询时重新初始化")
    
    # 更新配置
    st.session_state.config = new_config
    
    st.markdown("---")
    st.subheader("📊 系统状态")
    
    if st.session_state.initialized:
        st.success("✅ 系统已初始化")
        st.info(f"🏢 公司: {st.session_state.company_name}")
        st.info(f"💬 历史问答: {len(st.session_state.history)} 条")
    else:
        st.warning("⚠️ 系统未初始化")
    
    # 初始化按钮
    if st.button("🔄 重新初始化系统", use_container_width=True):
        st.session_state.initialized = False
        st.session_state.processor = None
        st.rerun()
    
    # 保存历史按钮
    if st.button("💾 保存问答历史", use_container_width=True):
        filename = save_history()
        if filename:
            st.success(f"✅ 历史已保存到 {filename}")
        else:
            st.warning("⚠️ 无历史记录可保存")
    
    # 清空历史按钮
    if st.button("🗑️ 清空对话历史", use_container_width=True, type="secondary"):
        if st.session_state.history:
            st.session_state.history = []
            st.success("✅ 对话历史已清空")
            st.rerun()
        else:
            st.info("ℹ️ 当前无历史记录")
    
    # 清空历史按钮
    if st.button("🗑️ 清空历史", use_container_width=True):
        st.session_state.history = []
        st.success("✅ 历史已清空")
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 📊 批量评估")
    
    # 批量评估配置
    with st.expander("⚙️ 评估配置（可选）", expanded=False):
        st.markdown("#### 🔧 评估时使用的配置")
        st.info("💡 如果不修改，将使用上方流程配置中的当前设置")
        
        eval_col1, eval_col2 = st.columns(2)
        
        with eval_col1:
            st.markdown("##### 🚀 检索增强")
            eval_use_hyde = st.checkbox(
                "启用 HYDE",
                value=config_defaults.get('use_hyde', True),
                help="生成假设性答案辅助检索",
                key="eval_use_hyde"
            )
            eval_use_multi_query = st.checkbox(
                "启用 Multi-Query",
                value=config_defaults.get('use_multi_query', True),
                help="生成多个相关查询并行检索",
                key="eval_use_multi_query"
            )
            
            if eval_use_multi_query:
                st.markdown("**Multi-Query 方法：**")
                eval_mq_synonym = st.checkbox(
                    "名词解释",
                    value=config_defaults.get('multi_query_methods', {}).get('synonym', True),
                    help="为财务名词补充定义、近义词、计算方法",
                    key="eval_mq_synonym"
                )
                eval_mq_subquestion = st.checkbox(
                    "指标拆分",
                    value=config_defaults.get('multi_query_methods', {}).get('subquestion', False),
                    help="按指标/时间拆分子问题",
                    key="eval_mq_subquestion"
                )
                eval_mq_variant = st.checkbox(
                    "情景变体",
                    value=config_defaults.get('multi_query_methods', {}).get('variant', False),
                    help="生成不同角度的提问",
                    key="eval_mq_variant"
                )
        
        with eval_col2:
            st.markdown("##### 🎯 重排序与扩充")
            eval_llm_reranking = st.checkbox(
                "启用 LLM 重排序",
                value=config_defaults.get('llm_reranking', True),
                help="使用LLM对检索结果进行智能重排序",
                key="eval_llm_reranking"
            )
            
            if eval_llm_reranking:
                eval_rerank_sample_size = st.number_input(
                    "重排序样本数",
                    min_value=10,
                    max_value=100,
                    value=config_defaults.get('rerank_sample_size', 20),
                    step=10,
                    help="LLM重排序时处理的样本数量",
                    key="eval_rerank_sample_size"
                )
            else:
                eval_rerank_sample_size = 20
            
            eval_expand_upstream = st.checkbox(
                "启用上下游扩充",
                value=config_defaults.get('expand_upstream', True),
                help="扩充检索结果的上下文页面",
                key="eval_expand_upstream"
            )
            
            if eval_expand_upstream:
                eval_expand_top_k = st.number_input(
                    "扩充 top-k",
                    min_value=1,
                    max_value=20,
                    value=config_defaults.get('expand_top_k', 5),
                    step=1,
                    help="对前k个检索结果进行上下游扩充",
                    key="eval_expand_top_k"
                )
                eval_expand_context_size = st.number_input(
                    "扩充大小",
                    min_value=1,
                    max_value=5,
                    value=config_defaults.get('expand_context_size', 1),
                    step=1,
                    help="向上和向下各扩充的页面数",
                    key="eval_expand_context_size"
                )
            else:
                eval_expand_top_k = 5
                eval_expand_context_size = 1
            
            st.markdown("##### 📊 检索参数")
            eval_top_n = st.number_input(
                "最终检索数量",
                min_value=5,
                max_value=50,
                value=config_defaults.get('top_n_retrieval', 10),
                step=5,
                help="最终返回的检索结果数量",
                key="eval_top_n"
            )
        
        # 应用评估配置按钮
        if st.button("✅ 应用此配置到评估", use_container_width=True):
            st.session_state.eval_config = {
                'use_hyde': eval_use_hyde,
                'use_multi_query': eval_use_multi_query,
                'multi_query_methods': {
                    'synonym': eval_mq_synonym if eval_use_multi_query else False,
                    'subquestion': eval_mq_subquestion if eval_use_multi_query else False,
                    'variant': eval_mq_variant if eval_use_multi_query else False
                },
                'llm_reranking': eval_llm_reranking,
                'rerank_sample_size': eval_rerank_sample_size,
                'expand_upstream': eval_expand_upstream,
                'expand_top_k': eval_expand_top_k,
                'expand_context_size': eval_expand_context_size,
                'top_n_retrieval': eval_top_n
            }
            st.success("✅ 评估配置已应用！点击下方按钮开始评估")
    
    if st.button("🚀 一键评估所有问题", use_container_width=True, type="primary"):
        st.session_state.evaluating = True
        st.rerun()
    
    parallel_requests = st.slider(
        "🧵 批量评估并发数",
        min_value=1,
        max_value=16,
        value=config_defaults.get('parallel_requests', 4),
        step=1,
        help="设置一键评估运行时使用的并行线程数（数值越大速度越快，但占用资源更多）",
        key="parallel_requests_slider"
    )
    st.session_state.config['parallel_requests'] = parallel_requests
    
    st.markdown("---")
    st.markdown("### 📖 使用说明")
    st.markdown("""
    1. **配置模型**: 选择API提供商和模型
    2. **调整参数**: 设置检索数量和增强策略
    3. **输入问题**: 在主界面输入问题
    4. **选择类型**: 选择期望的答案类型
    5. **查看结果**: 系统返回答案和分析过程
    
    **答案类型说明**:
    - `jingpan`: 金盘科技专用（中文财务）
    - `number`: 数字类答案
    - `boolean`: 是/否类答案
    - `name`: 单个名称
    - `names`: 多个名称列表
    """)

# ==================== 评估结果可视化辅助函数 ====================
@st.cache_data
def load_evaluation_results(val_result_dir: str = "data/val_set/val_result"):
    """加载所有评估结果文件"""
    result_dir = Path(val_result_dir)
    if not result_dir.exists():
        return []
    
    results = []
    for json_file in sorted(result_dir.glob("evaluation_*.json"), reverse=True):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                data['file_name'] = json_file.name
                data['file_path'] = str(json_file)
                results.append(data)
        except Exception as e:
            st.warning(f"加载评估文件失败 {json_file.name}: {e}")
    
    return results

def format_config_summary(config: dict) -> str:
    """格式化配置摘要"""
    parts = []
    if config.get('use_hyde'):
        parts.append("HYDE")
    if config.get('use_multi_query'):
        mq_methods = []
        if config.get('multi_query_methods', {}).get('synonym'):
            mq_methods.append("名词解释")
        if config.get('multi_query_methods', {}).get('subquestion'):
            mq_methods.append("指标拆分")
        if config.get('multi_query_methods', {}).get('variant'):
            mq_methods.append("情景变体")
        if mq_methods:
            parts.append(f"Multi-Query({','.join(mq_methods)})")
    if config.get('llm_reranking'):
        parts.append(f"LLM重排序(样本{config.get('rerank_sample_size', 20)})")
    if config.get('expand_upstream'):
        parts.append(f"上游扩充(k={config.get('expand_top_k', 5)},±{config.get('expand_context_size', 1)})")
    return " | ".join(parts) if parts else "基础配置"

def find_question_across_results(question: str, evaluation_results: List[dict]) -> List[dict]:
    """在所有评估结果中查找某个问题的答案"""
    matches = []
    for eval_data in evaluation_results:
        for result in eval_data.get('results', []):
            if result.get('question', '').strip() == question.strip():
                matches.append({
                    'config': eval_data.get('config', {}),
                    'config_summary': format_config_summary(eval_data.get('config', {})),
                    'timestamp': eval_data.get('timestamp', ''),
                    'file_name': eval_data.get('file_name', ''),
                    'rag_answer': result.get('rag_answer', ''),
                    'standard_answer': result.get('standard_answer', ''),
                    'score': result.get('score', 0.0),
                    'reasoning': result.get('reasoning', ''),
                    'is_correct': result.get('is_correct', False)
                })
    return matches

# ==================== 主界面 ====================
# 主功能选择
main_tab1, main_tab2 = st.tabs(["💬 问答系统", "📊 评估结果分析"])

with main_tab1:
    st.title("🏢 金盘科技 RAG 问答系统")
    st.markdown("基于 **FAISS + Qwen + 时间路由** 的智能财务问答系统")
    
    # 初始化系统
    if not st.session_state.initialized:
        if initialize_system():
            st.success("✅ 系统初始化成功！")
            st.rerun()
        else:
            st.stop()
    
    # 批量评估功能
    if st.session_state.get('evaluating', False):
        st.session_state.evaluating = False
        
        st.markdown("---")
        st.markdown("## 📊 批量评估进行中...")
        
        try:
            # 加载问题
            questions_df = pd.read_csv("data/val_set/questions_selected_100.csv")
            benchmark_map = load_benchmark_answers("金盘财报查询场景问题benchmark-原先的表格.csv")
            
            # 创建评估结果目录
            val_result_dir = Path("data/val_set/val_result")
            val_result_dir.mkdir(parents=True, exist_ok=True)
            
            # 初始化评估结果
            evaluation_results = []
            total_questions = len(questions_df)
            correct_count = 0
            total_score = 0.0
            
            # 收集各阶段时间
            timing_accumulator = {
                'init_retriever': [],
                'retrieval': [],
                'hyde_expansion': [],
                'multi_query_expansion': [],
                'llm_reranking': [],
                'upstream_expansion': [],
                'format_results': [],
                'generate_answer': [],
                'vector_search': [],
                'total_time': []
            }
            
            # 创建进度条
            progress_bar = st.progress(0)
            status_text = st.empty()
            results_container = st.container()
            
            # 初始化API处理器
            api_processor = APIProcessor(provider="qwen")
            company_name = st.session_state.company_name
            config = st.session_state.config
            
            # 检查是否有专门的评估配置
            eval_config = st.session_state.get('eval_config', None)
            if eval_config:
                st.info(f"📋 使用自定义评估配置: HYDE={eval_config['use_hyde']}, Multi-Query={eval_config['use_multi_query']}, LLM重排序={eval_config['llm_reranking']}")
                # 使用评估配置
                config_info = {
                    'top_n_retrieval': eval_config.get('top_n_retrieval', 10),
                    'use_hyde': eval_config.get('use_hyde', True),
                    'use_multi_query': eval_config.get('use_multi_query', True),
                    'multi_query_methods': eval_config.get('multi_query_methods', {'synonym': True, 'subquestion': False, 'variant': False}),
                    'llm_reranking': eval_config.get('llm_reranking', True),
                    'rerank_sample_size': eval_config.get('rerank_sample_size', 20),
                    'expand_upstream': eval_config.get('expand_upstream', False),
                    'expand_top_k': eval_config.get('expand_top_k', 5),
                    'expand_context_size': eval_config.get('expand_context_size', 1),
                    'parent_document_retrieval': True,
                    'parallel_requests': config.get('parallel_requests', 4),
                    'answering_model': config.get('answering_model', 'qwen-max'),
                    'api_provider': config.get('api_provider', 'qwen')
                }
                # 临时更新processor的配置
                st.session_state.processor.use_hyde = eval_config['use_hyde']
                st.session_state.processor.use_multi_query = eval_config['use_multi_query']
                st.session_state.processor.multi_query_methods = eval_config['multi_query_methods']
                st.session_state.processor.llm_reranking = eval_config['llm_reranking']
                st.session_state.processor.llm_reranking_sample_size = eval_config['rerank_sample_size']
                st.session_state.processor.expand_upstream = eval_config['expand_upstream']
                st.session_state.processor.expand_top_k = eval_config['expand_top_k']
                st.session_state.processor.expand_context_size = eval_config['expand_context_size']
                st.session_state.processor.top_n_retrieval = eval_config['top_n_retrieval']
            else:
                st.info("📋 使用当前流程配置进行评估")
                # 使用当前配置
                config_info = {
                    'top_n_retrieval': config.get('top_n_retrieval', 10),
                    'use_hyde': config.get('use_hyde', True),
                    'use_multi_query': config.get('use_multi_query', True),
                    'multi_query_methods': config.get('multi_query_methods', {'synonym': True, 'subquestion': False, 'variant': False}),
                    'llm_reranking': config.get('llm_reranking', True),
                    'rerank_sample_size': config.get('llm_reranking_sample_size', 20),
                    'expand_upstream': config.get('expand_upstream', False),
                    'expand_top_k': config.get('expand_top_k', 5),
                    'expand_context_size': config.get('expand_context_size', 2),
                    'parent_document_retrieval': True,  # 默认启用父文档检索
                    'parallel_requests': config.get('parallel_requests', 4),
                    'answering_model': config.get('answering_model', 'qwen-max'),
                    'api_provider': config.get('api_provider', 'qwen')
                }
            
            # 显示超参数确认对话框
            st.markdown("---")
            st.markdown("### 📋 超参数配置确认")
            
            # 创建两列显示配置
            conf_col1, conf_col2 = st.columns(2)
            
            with conf_col1:
                st.markdown("#### 🚀 检索增强配置")
                st.markdown(f"- **HYDE**: {'✅ 启用' if config_info['use_hyde'] else '❌ 关闭'}")
                st.markdown(f"- **Multi-Query**: {'✅ 启用' if config_info['use_multi_query'] else '❌ 关闭'}")
                if config_info['use_multi_query']:
                    mq_methods = config_info['multi_query_methods']
                    st.markdown(f"  - 名词解释: {'✅' if mq_methods.get('synonym', False) else '❌'}")
                    st.markdown(f"  - 指标拆分: {'✅' if mq_methods.get('subquestion', False) else '❌'}")
                    st.markdown(f"  - 情景变体: {'✅' if mq_methods.get('variant', False) else '❌'}")
                
                st.markdown("#### 🎯 重排序配置")
                st.markdown(f"- **LLM重排序**: {'✅ 启用' if config_info['llm_reranking'] else '❌ 关闭'}")
                if config_info['llm_reranking']:
                    st.markdown(f"  - 样本数: {config_info['rerank_sample_size']}")
            
            with conf_col2:
                st.markdown("#### 📊 检索参数")
                st.markdown(f"- **最终检索数量**: {config_info['top_n_retrieval']}")
                st.markdown(f"- **父文档检索**: {'✅ 启用' if config_info['parent_document_retrieval'] else '❌ 关闭'}")
                
                st.markdown("#### 🔄 上下游扩充")
                st.markdown(f"- **上下游扩充**: {'✅ 启用' if config_info['expand_upstream'] else '❌ 关闭'}")
                if config_info['expand_upstream']:
                    st.markdown(f"  - 扩充 top-k: {config_info['expand_top_k']}")
                    st.markdown(f"  - 扩充大小: ±{config_info['expand_context_size']} 页")
                
                st.markdown("#### 🤖 模型配置")
                st.markdown(f"- **回答模型**: {config_info['answering_model']}")
                st.markdown(f"- **并发数**: {config_info['parallel_requests']}")
            
            st.markdown("---")
            st.warning("⚠️ 评估将使用上述配置运行，预计耗时较长。请确认配置无误后继续。")
            
            # 遍历所有问题
            for idx, row in questions_df.iterrows():
                question = str(row.get('提问内容', '')).strip()
                if not question:
                    continue
                
                # 更新进度
                progress = (idx + 1) / total_questions
                status_text.text(f"正在评估第 {idx + 1}/{total_questions} 个问题: {question[:50]}...")
                progress_bar.progress(progress)
                
                # 获取标准答案
                standard_answer = get_standard_answer(question, questions_df, benchmark_map)
                if not standard_answer:
                    # 如果没有标准答案，跳过
                    evaluation_results.append({
                        'question': question,
                        'standard_answer': '',
                        'rag_answer': '',
                        'score': 0.0,
                        'reasoning': '无标准答案，跳过评估',
                        'is_correct': False,
                        'skipped': True,
                        'timing': {}
                    })
                    continue
                
                try:
                    # 调用RAG系统获取答案
                    full_question = f"{company_name}{question}" if company_name not in question else question
                    answer_dict = st.session_state.processor.get_answer_for_company(
                        company_name=company_name,
                        question=full_question,
                        schema="jingpan",
                        conversation_history=None,
                        progress_callback=None,
                        selected_years=None
                    )
                    
                    rag_answer = str(answer_dict.get("final_answer", answer_dict.get("answer", "N/A")))
                    
                    # 提取时间信息
                    timing = answer_dict.get('timing', {})
                    if timing:
                        for key in timing_accumulator:
                            if key in timing:
                                timing_accumulator[key].append(timing[key])
                    
                    # 使用LLM as Judge评估
                    try:
                        eval_result = api_processor.evaluate_answer(
                            question=question,
                            standard_answer=standard_answer,
                            rag_answer=rag_answer,
                            model="qwen-turbo"
                        )
                        
                        # 验证评估结果的有效性
                        if not eval_result or not isinstance(eval_result, dict):
                            raise ValueError("评估结果为空或格式错误")
                        
                        score = eval_result.get('score', 0.0)
                        reasoning = eval_result.get('reasoning', '')
                        
                        # 验证reasoning不为空
                        if not reasoning or not reasoning.strip():
                            raise ValueError(f"评估返回的reasoning为空，score={score}")
                        
                        total_score += score
                        is_correct = score >= 0.8
                        if is_correct:
                            correct_count += 1
                        
                        evaluation_results.append({
                            'question': question,
                            'standard_answer': standard_answer,
                            'rag_answer': rag_answer,
                            'score': score,
                            'reasoning': reasoning,
                            'is_correct': is_correct,
                            'skipped': False,
                            'timing': timing
                        })
                        
                    except Exception as eval_error:
                        # 评估失败时的降级处理
                        error_msg = str(eval_error)
                        print(f"[WARNING] 评估失败 (问题: {question[:50]}...): {error_msg}")
                        st.warning(f"⚠️ 问题评估失败: {error_msg}")
                        
                        # 使用默认值，但保留RAG答案和时间信息
                        evaluation_results.append({
                            'question': question,
                            'standard_answer': standard_answer,
                            'rag_answer': rag_answer,  # 保留RAG答案
                            'score': 0.0,
                            'reasoning': f'评估失败: {error_msg}',
                            'is_correct': False,
                            'skipped': False,
                            'timing': timing  # 保留时间信息
                        })
                    
                except Exception as e:
                    evaluation_results.append({
                        'question': question,
                        'standard_answer': standard_answer,
                        'rag_answer': '',
                        'score': 0.0,
                        'reasoning': f'评估失败: {str(e)}',
                        'is_correct': False,
                        'skipped': False,
                        'error': str(e),
                        'timing': {}
                    })
            
            # 完成评估
            progress_bar.progress(1.0)
            status_text.text("✅ 评估完成！")
            
            # 统计结果
            evaluated_count = len([r for r in evaluation_results if not r.get('skipped', False)])
            accuracy = correct_count / evaluated_count if evaluated_count > 0 else 0.0
            average_score = total_score / evaluated_count if evaluated_count > 0 else 0.0
            
            # 计算各阶段平均用时（精确到秒）
            avg_timing = {}
            for key, times in timing_accumulator.items():
                if times:
                    avg_time = sum(times) / len(times)
                    avg_timing[key] = round(avg_time, 2)  # 保留2位小数（精确到0.01秒）
                else:
                    avg_timing[key] = 0.0
            
            # 获取最终检索数量（从配置中）
            final_retrieval_count = config_info['top_n_retrieval']
            if config_info.get('expand_upstream', False):
                # 如果有上游扩充，检索数量会更多
                final_retrieval_count = f"{config_info['top_n_retrieval']} + 扩充"
            
            # 保存结果
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_file = val_result_dir / f"evaluation_{timestamp}.json"
            
            result_data = {
                'timestamp': timestamp,
                'total_questions': total_questions,
                'evaluated_count': evaluated_count,
                'correct_count': correct_count,
                'accuracy': accuracy,
                'average_score': average_score,
                'config': config_info,
                'final_retrieval_count': final_retrieval_count,
                'average_timing': avg_timing,
                'results': evaluation_results
            }
            
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, ensure_ascii=False, indent=2)
            
            # 显示统计结果
            with results_container:
                st.success(f"✅ 评估完成！结果已保存到: {result_file}")
                
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("总问题数", total_questions)
                with col2:
                    st.metric("已评估", evaluated_count)
                with col3:
                    st.metric("正确答案", correct_count)
                with col4:
                    st.metric("正确率", f"{accuracy*100:.2f}%")
                with col5:
                    st.metric("平均得分", f"{average_score:.3f}")
                
                # 显示各阶段平均用时
                st.markdown("### ⏱️ 各阶段平均用时（秒）")
                timing_df = pd.DataFrame([
                    {'阶段': '初始化检索器', '平均用时(秒)': avg_timing.get('init_retriever', 0.0)},
                    {'阶段': '向量检索', '平均用时(秒)': avg_timing.get('retrieval', 0.0)},
                    {'阶段': 'HYDE扩展', '平均用时(秒)': avg_timing.get('hyde_expansion', 0.0)},
                    {'阶段': 'Multi-Query扩展', '平均用时(秒)': avg_timing.get('multi_query_expansion', 0.0)},
                    {'阶段': '向量搜索', '平均用时(秒)': avg_timing.get('vector_search', 0.0)},
                    {'阶段': 'LLM重排序', '平均用时(秒)': avg_timing.get('llm_reranking', 0.0)},
                    {'阶段': '上游扩充', '平均用时(秒)': avg_timing.get('upstream_expansion', 0.0)},
                    {'阶段': '格式化结果', '平均用时(秒)': avg_timing.get('format_results', 0.0)},
                    {'阶段': '生成答案', '平均用时(秒)': avg_timing.get('generate_answer', 0.0)},
                    {'阶段': '总用时', '平均用时(秒)': avg_timing.get('total_time', 0.0)},
                ])
                st.dataframe(timing_df, use_container_width=True, hide_index=True)
                
                # 显示详细结果表格
                st.markdown("### 📋 详细评估结果")
                results_df = pd.DataFrame([
                    {
                        '问题': r['question'][:50] + '...' if len(r['question']) > 50 else r['question'],
                        '标准答案': r['standard_answer'][:50] + '...' if len(r.get('standard_answer', '')) > 50 else r.get('standard_answer', ''),
                        'RAG答案': r['rag_answer'][:50] + '...' if len(r.get('rag_answer', '')) > 50 else r.get('rag_answer', ''),
                        '评分': r['score'],
                        '是否正确': '✅' if r['is_correct'] else '❌',
                        '状态': '跳过' if r.get('skipped', False) else '已评估'
                    }
                    for r in evaluation_results
                ])
                st.dataframe(results_df, use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ 评估过程出错: {str(e)}")
            with st.expander("查看详细错误"):
                st.code(traceback.format_exc())
    
    # 问答区域
    st.markdown("---")

# 如果点击了示例问题，显示提示
if st.session_state.get('example_clicked', False):
    st.success(f"✅ 已选择示例问题：**{st.session_state.current_question}**")
    st.session_state.example_clicked = False

# 固定使用 jingpan 模式（深度分析模式）
schema_type = "jingpan"

# 添加说明信息
st.info("💡 **深度分析模式**：系统将为您提供详细的答案、推理过程和数据来源，适用于所有类型的问题。")

# 使用动态 key 以便每次点击示例问题后重新渲染
question_key = f"question_input_{st.session_state.widget_key_counter}"
question_input = st.text_input(
    "💬 请输入您的问题",
    value=st.session_state.current_question,
    placeholder="例如：2024年第一季度的营业收入是多少？",
    help="系统会自动添加公司名称（金盘科技）到问题前",
    key=question_key
)

# 问答按钮
if st.button("🚀 获取答案", type="primary", use_container_width=True):
    if not question_input.strip():
        st.warning("⚠️ 请输入问题")
    else:
        # 确保问题包含公司名称
        company_name = st.session_state.company_name
        if company_name not in question_input:
            full_question = f"{company_name}{question_input}"
        else:
            full_question = question_input
        
        # 显示问题
        st.markdown(f'<div class="question-box"><b>❓ 问题:</b> {full_question}<br><b>📝 分析模式:</b> 深度分析 (jingpan)</div>', 
                   unsafe_allow_html=True)
        
        try:
            # 准备对话历史（如果启用多轮对话）
            conversation_history = prepare_conversation_history(st.session_state.context_turns)
            
            # 显示多轮对话状态
            if conversation_history:
                st.info(f"🔄 多轮对话模式：使用最近 {len(conversation_history)} 轮对话作为上下文")
            
            # 创建进度条容器
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # 定义进度回调函数
            def update_progress(stage: str, progress: int):
                """更新进度条的回调函数"""
                status_text.text(stage)
                progress_bar.progress(progress)
            
            # 获取选中的年份（如果有）
            selected_years = st.session_state.get("selected_years", None)
            
            # 调用问答系统，传入真实的进度回调
            answer_dict = st.session_state.processor.get_answer_for_company(
                company_name=company_name,
                question=full_question,
                schema=schema_type,
                conversation_history=conversation_history,
                progress_callback=update_progress,
                selected_years=selected_years
            )
            
            # 完成
            import time
            status_text.text("✅ 处理完成！")
            progress_bar.progress(100)
            time.sleep(0.5)
            
            # 清除进度显示
            progress_bar.empty()
            status_text.empty()
            
            # 显示答案（传入问题以便查找标准答案）
            format_answer_display(answer_dict, full_question)
            
            # 保存到历史
            st.session_state.history.append({
                'timestamp': datetime.now().isoformat(),
                'question': full_question,
                'schema': schema_type,
                'answer': answer_dict
            })
            
            st.success("✅ 问答完成！")
            
        except Exception as e:
            error_msg = str(e)
            
            # 特殊处理400错误（通常是Token超限）
            if "400" in error_msg or "Bad Request" in error_msg:
                st.error("❌ **API请求失败：400 Bad Request**")
                st.markdown("""
                **可能原因**：
                - 🚨 **Token超限**：上游扩充导致上下文过长（46页约36,800 tokens）
                - ⚠️ API参数错误或格式不正确
                
                **解决方法**：
                1. **降低 expand_top_k**：从 10 降至 3-5
                2. **降低 expand_context_size**：从 2 降至 1
                3. **关闭上游扩充**：使用传统的下游扩充模式
                4. 检查侧边栏的Token预估，确保不超过 25,000 tokens
                
                **推荐配置**（适合大部分场景）：
                - expand_top_k = 5
                - expand_context_size = 1
                - 预计Token: ~12,000 ✅
                """)
            else:
                st.error(f"❌ 处理问题时出错: {error_msg}")
            
            with st.expander("查看详细错误"):
                st.code(traceback.format_exc())

# 示例问题（从问题库加载）
st.markdown("---")
st.markdown("### 💡 投资者关注问题")

# 加载问题库
try:
    questions_df = pd.read_csv("data/val_set/questions_selected_100.csv")
    total_questions = len(questions_df)
    st.markdown(f"点击下方问题可自动填入输入框 | 当前共有 **{total_questions}** 个问题")

    ...
    
    # 获取所有问题类型
    question_types = questions_df['问题类型'].unique().tolist()
    
    # 创建问题类型选择器
    col_select, col_random = st.columns([3, 1])
    
    with col_select:
        selected_type = st.selectbox(
            "选择问题类型",
            options=["全部"] + sorted(question_types),
            index=0,
            key="question_type_selector"
        )
    
    with col_random:
        if st.button("🎲 随机问题", use_container_width=True):
            random_q = questions_df.sample(1)['提问内容'].values[0]
            st.session_state.current_question = random_q
            st.session_state.current_schema = "jingpan"
            st.session_state.example_clicked = True
            st.session_state.widget_key_counter += 1
            st.rerun()
    
    # 筛选问题
    if selected_type == "全部":
        filtered_questions = questions_df
    else:
        filtered_questions = questions_df[questions_df['问题类型'] == selected_type]
    
    # 显示问题统计
    st.info(f"📊 当前类型共 **{len(filtered_questions)}** 个问题")
    
    # 分3列展示问题
    col1, col2, col3 = st.columns(3)
    
    # 将问题分配到3列
    questions_list = filtered_questions['提问内容'].tolist()
    
    # 最多显示15个问题（避免页面过长）
    display_limit = 15
    if len(questions_list) > display_limit:
        questions_list = questions_list[:display_limit]
        st.warning(f"⚠️ 仅显示前 {display_limit} 个问题，可通过类型筛选查看更多")
    
    # 平均分配到3列
    questions_per_col = len(questions_list) // 3 + (1 if len(questions_list) % 3 > 0 else 0)
    
    for idx, col in enumerate([col1, col2, col3]):
        with col:
            start_idx = idx * questions_per_col
            end_idx = min((idx + 1) * questions_per_col, len(questions_list))
            
            for i in range(start_idx, end_idx):
                q = questions_list[i]
                # 截断过长的问题用于按钮显示
                button_text = q if len(q) <= 50 else q[:47] + "..."
                
                if st.button(button_text, key=f"ex_q_{i}_{hash(q) % 10000}", use_container_width=True):
                    # 更新问题（固定使用 jingpan）
                    st.session_state.current_question = q
                    st.session_state.current_schema = "jingpan"
                    st.session_state.example_clicked = True
                    # 增加计数器，强制重新渲染输入框
                    st.session_state.widget_key_counter += 1
                    st.rerun()

except FileNotFoundError:
    st.warning("⚠️ 问题库文件未找到，显示默认示例问题")
    
    # 默认示例问题（作为后备）
    example_col1, example_col2 = st.columns(2)
    
    with example_col1:
        st.markdown("**📊 财务数据类**")
        examples_financial = [
            "2024年第一季度的营业收入是多少？",
            "2023年到2025年的净利润对比",
            "截至2025年9月30日的总资产",
        ]
        for q in examples_financial:
            if st.button(q, key=f"ex_fin_{q[:10]}", use_container_width=True):
                st.session_state.current_question = q
                st.session_state.current_schema = "jingpan"
                st.session_state.example_clicked = True
                st.session_state.widget_key_counter += 1
                st.rerun()
    
    with example_col2:
        st.markdown("**📝 信息查询类**")
        examples_info = [
            "公司的法定代表人是谁？",
            "2024年有哪些主要产品？",
            "公司是否有海外业务？",
        ]
        for q in examples_info:
            if st.button(q, key=f"ex_info_{q[:10]}", use_container_width=True):
                st.session_state.current_question = q
                st.session_state.current_schema = "jingpan"
                st.session_state.example_clicked = True
                st.session_state.widget_key_counter += 1
                st.rerun()

except Exception as e:
    st.error(f"❌ 加载问题库时出错: {str(e)}")

# 历史记录展示
if st.session_state.history:
    st.markdown("---")
    st.markdown("### 📜 问答历史")
    
    with st.expander(f"查看历史记录（共 {len(st.session_state.history)} 条）", expanded=False):
        for i, record in enumerate(reversed(st.session_state.history), 1):
            # 使用更清晰的容器展示每条记录
            with st.container():
                st.markdown(f"#### 📋 记录 {i}")
                st.markdown(f"🕐 **时间**: {record['timestamp']}")
                st.markdown(f"❓ **问题**: {record['question']}")
                st.markdown(f"📝 **类型**: `{record['schema']}`")
                
                answer = record['answer'].get('final_answer', record['answer'].get('answer', 'N/A'))
                st.markdown(f"💡 **答案**: **{answer}**")
                st.markdown("---")

with main_tab2:
    st.title("📊 评估结果分析")
    st.markdown("分析不同参数配置下的评估结果，对比答案差异和统计指标")
    
    # 加载评估结果
    evaluation_results = load_evaluation_results()
    
    if not evaluation_results:
        st.warning("⚠️ 未找到评估结果文件。请先运行批量评估。")
        st.info("评估结果文件应位于: `data/val_set/val_result/evaluation_*.json`")
    else:
        st.success(f"✅ 已加载 {len(evaluation_results)} 个评估结果文件")
        
        # 功能选择
        analysis_mode = st.radio(
            "选择分析模式",
            ["问题对比", "配置统计"],
            horizontal=True,
            key="analysis_mode"
        )
        
        if analysis_mode == "问题对比":
            st.markdown("### 🔍 问题对比分析")
            st.markdown("查看某个问题在不同参数配置下的回答差异")
            
            # 获取所有问题列表
            all_questions = set()
            for eval_data in evaluation_results:
                for result in eval_data.get('results', []):
                    all_questions.add(result.get('question', '').strip())
            
            if all_questions:
                selected_question = st.selectbox(
                    "选择要对比的问题",
                    sorted(all_questions),
                    key="question_compare_select"
                )
                
                if selected_question:
                    # 查找该问题在所有评估结果中的答案
                    matches = find_question_across_results(selected_question, evaluation_results)
                    
                    if matches:
                        st.markdown(f"#### 📋 找到 {len(matches)} 个配置下的答案")
                        
                        # 显示标准答案
                        if matches[0].get('standard_answer'):
                            st.info(f"📌 **标准答案**: {matches[0]['standard_answer']}")
                        
                        # 显示每个配置的答案
                        for i, match in enumerate(matches, 1):
                            with st.expander(
                                f"配置 {i}: {match['config_summary']} | "
                                f"得分: {match['score']:.2f} | "
                                f"{'✅ 正确' if match['is_correct'] else '❌ 错误'} | "
                                f"时间: {match['timestamp']}",
                                expanded=(i == 1)
                            ):
                                col1, col2 = st.columns([2, 1])
                                
                                with col1:
                                    st.markdown("**RAG生成的答案:**")
                                    st.write(match['rag_answer'])
                                
                                with col2:
                                    st.metric("评分", f"{match['score']:.2f}")
                                    st.metric("是否正确", "✅" if match['is_correct'] else "❌")
                                
                                if match.get('reasoning'):
                                    st.markdown("**评估理由:**")
                                    st.caption(match['reasoning'])
                                
                                st.caption(f"文件: {match['file_name']}")
                        
                        # 对比表格
                        st.markdown("#### 📊 对比表格")
                        compare_df = pd.DataFrame([
                            {
                                '配置': match['config_summary'],
                                '得分': match['score'],
                                '是否正确': '✅' if match['is_correct'] else '❌',
                                'RAG答案': match['rag_answer'][:100] + '...' if len(match['rag_answer']) > 100 else match['rag_answer'],
                                '时间': match['timestamp']
                            }
                            for match in matches
                        ])
                        st.dataframe(compare_df, use_container_width=True, hide_index=True)
                    else:
                        st.warning("未找到该问题的评估结果")
            else:
                st.warning("未找到任何问题")
        
        elif analysis_mode == "配置统计":
            st.markdown("### 📈 配置统计信息")
            st.markdown("查看某个参数配置的评估统计结果")
            
            # 选择评估文件
            eval_options = [
                f"{eval_data['timestamp']} | {format_config_summary(eval_data.get('config', {}))} | "
                f"准确率: {eval_data.get('accuracy', 0)*100:.1f}%"
                for eval_data in evaluation_results
            ]
            
            selected_idx = st.selectbox(
                "选择评估结果",
                range(len(evaluation_results)),
                format_func=lambda x: eval_options[x],
                key="config_stats_select"
            )
            
            if selected_idx is not None:
                selected_eval = evaluation_results[selected_idx]
                config = selected_eval.get('config', {})
                
                # 显示配置信息
                st.markdown("#### ⚙️ 配置参数")
                config_cols = st.columns(3)
                
                with config_cols[0]:
                    st.markdown("**检索增强:**")
                    st.write(f"- HYDE: {'✅' if config.get('use_hyde') else '❌'}")
                    st.write(f"- Multi-Query: {'✅' if config.get('use_multi_query') else '❌'}")
                    if config.get('use_multi_query'):
                        mq = config.get('multi_query_methods', {})
                        st.write(f"  - 名词解释: {'✅' if mq.get('synonym') else '❌'}")
                        st.write(f"  - 指标拆分: {'✅' if mq.get('subquestion') else '❌'}")
                        st.write(f"  - 情景变体: {'✅' if mq.get('variant') else '❌'}")
                
                with config_cols[1]:
                    st.markdown("**重排序与扩充:**")
                    st.write(f"- LLM重排序: {'✅' if config.get('llm_reranking') else '❌'}")
                    if config.get('llm_reranking'):
                        st.write(f"  - 样本数: {config.get('rerank_sample_size', 'N/A')}")
                    st.write(f"- 上游扩充: {'✅' if config.get('expand_upstream') else '❌'}")
                    if config.get('expand_upstream'):
                        st.write(f"  - Top-K: {config.get('expand_top_k', 'N/A')}")
                        st.write(f"  - 扩充大小: ±{config.get('expand_context_size', 'N/A')}页")
                
                with config_cols[2]:
                    st.markdown("**其他参数:**")
                    st.write(f"- 最终检索数: {config.get('top_n_retrieval', 'N/A')}")
                    st.write(f"- 回答模型: {config.get('answering_model', 'N/A')}")
                    st.write(f"- 并发数: {config.get('parallel_requests', 'N/A')}")
                
                st.markdown("---")
                
                # 显示统计指标
                st.markdown("#### 📊 评估统计")
                stat_cols = st.columns(5)
                
                with stat_cols[0]:
                    st.metric("总问题数", selected_eval.get('total_questions', 0))
                with stat_cols[1]:
                    st.metric("已评估", selected_eval.get('evaluated_count', 0))
                with stat_cols[2]:
                    st.metric("正确答案", selected_eval.get('correct_count', 0))
                with stat_cols[3]:
                    accuracy = selected_eval.get('accuracy', 0)
                    st.metric("准确率", f"{accuracy*100:.2f}%")
                with stat_cols[4]:
                    st.metric("平均得分", f"{selected_eval.get('average_score', 0):.3f}")
                
                # 显示时间统计
                avg_timing = selected_eval.get('average_timing', {})
                if avg_timing:
                    st.markdown("#### ⏱️ 平均用时（秒）")
                    timing_df = pd.DataFrame([
                        {'阶段': '初始化检索器', '平均用时(秒)': avg_timing.get('init_retriever', 0.0)},
                        {'阶段': '向量检索', '平均用时(秒)': avg_timing.get('retrieval', 0.0)},
                        {'阶段': 'HYDE扩展', '平均用时(秒)': avg_timing.get('hyde_expansion', 0.0)},
                        {'阶段': 'Multi-Query扩展', '平均用时(秒)': avg_timing.get('multi_query_expansion', 0.0)},
                        {'阶段': 'LLM重排序', '平均用时(秒)': avg_timing.get('llm_reranking', 0.0)},
                        {'阶段': '上游扩充', '平均用时(秒)': avg_timing.get('upstream_expansion', 0.0)},
                        {'阶段': '生成答案', '平均用时(秒)': avg_timing.get('generate_answer', 0.0)},
                        {'阶段': '总用时', '平均用时(秒)': avg_timing.get('total_time', 0.0)},
                    ])
                    st.dataframe(timing_df, use_container_width=True, hide_index=True)
                
                # 显示详细结果
                st.markdown("#### 📋 详细评估结果")
                results = selected_eval.get('results', [])
                if results:
                    results_df = pd.DataFrame([
                        {
                            '问题': r.get('question', '')[:60] + '...' if len(r.get('question', '')) > 60 else r.get('question', ''),
                            '标准答案': r.get('standard_answer', '')[:60] + '...' if len(r.get('standard_answer', '')) > 60 else r.get('standard_answer', ''),
                            'RAG答案': r.get('rag_answer', '')[:60] + '...' if len(r.get('rag_answer', '')) > 60 else r.get('rag_answer', ''),
                            '评分': r.get('score', 0.0),
                            '是否正确': '✅' if r.get('is_correct', False) else '❌',
                            '评估理由': r.get('reasoning', '')[:80] + '...' if len(r.get('reasoning', '')) > 80 else r.get('reasoning', '')
                        }
                        for r in results
                    ])
                    st.dataframe(results_df, use_container_width=True, hide_index=True)
                    
                    # 下载按钮
                    st.download_button(
                        label="📥 下载完整评估结果 (JSON)",
                        data=json.dumps(selected_eval, ensure_ascii=False, indent=2),
                        file_name=selected_eval.get('file_name', 'evaluation_result.json'),
                        mime="application/json"
                    )
                else:
                    st.warning("该评估结果中没有详细数据")

# 页脚
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>🔧 金盘科技 RAG 问答系统 | 基于 FAISS + Qwen-max + 时间智能路由</p>
    <p>💡 支持多年份对比、智能检索增强（HYDE + Multi-Query + LLM重排序）</p>
</div>
""", unsafe_allow_html=True)
