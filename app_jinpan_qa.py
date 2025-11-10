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
    st.session_state.enable_multi_turn = True  # 默认启用多轮对话
if 'context_turns' not in st.session_state:
    st.session_state.context_turns = 3  # 默认保留3轮历史

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
                parent_document_retrieval=False,
                llm_reranking=config['llm_reranking'],
                llm_reranking_sample_size=config.get('rerank_sample_size', 50),
                top_n_retrieval=config['top_n_retrieval'],
                parallel_requests=1,
                api_provider=config['api_provider'],
                answering_model=config['answering_model'],
                full_context=False,
                use_hyde=config['use_hyde'],
                use_multi_query=config['use_multi_query'],
                expand_upstream=config.get('expand_upstream', False),
                expand_top_k=config.get('expand_top_k', 5),
                expand_context_size=config.get('expand_context_size', 2)
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

def format_answer_display(answer_dict: dict):
    """格式化并显示答案"""
    # 获取答案
    answer = answer_dict.get("final_answer", answer_dict.get("answer", "N/A"))
    
    # 主答案 - 使用更明显的对比色
    st.markdown("### 📊 答案")
    st.markdown(f'<div class="answer-box"><h2 style="color: #0d6efd; margin-top: 0; margin-bottom: 0;">💡 {answer}</h2></div>', 
                unsafe_allow_html=True)
    
    # 创建标签页
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 分析过程", "📝 推理总结", "📚 LLM选用的参考", "🗂️ 所有检索结果"])
    
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
    
    # 显示推荐配置提示
    with st.expander("✨ 当前配置 - 推荐配置", expanded=False):
        st.markdown("""
        **🎯 推荐设置（已应用）**
        
        ✅ 检索数量：10  
        ✅ HYDE：开启  
        ✅ Multi-Query：开启  
        ✅ LLM重排序：开启  
        ✅ 初始召回：50  
        ✅ 上游扩充：开启  
        ✅ 核心页面：5  
        ✅ 扩充页数：上下各2页  
        ✅ 多轮对话：关闭  
        
        💡 这些配置在大多数场景下效果最佳
        """)
    
    # 初始化默认配置
    if 'config' not in st.session_state:
        st.session_state.config = {
            'api_provider': 'qwen',
            'answering_model': 'qwen-max',
            'top_n_retrieval': 10,
            'use_hyde': True,  # ✅ 已改用 Qwen API
            'use_multi_query': True,  # ✅ 已改用 Qwen API
            'llm_reranking': True,
            'rerank_sample_size': 50
        }
    
    st.markdown("---")
    st.subheader("🤖 模型配置")
    
    api_provider = st.selectbox(
        "API 提供商",
        options=['qwen', 'openai', 'gemini'],
        index=0,
        help="选择大语言模型API提供商"
    )
    
    # 根据API提供商显示不同的模型选项
    if api_provider == 'qwen':
        model_options = ['qwen-max', 'qwen-plus', 'qwen-turbo']
        default_model = 'qwen-max'
    elif api_provider == 'openai':
        model_options = ['gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo']
        default_model = 'gpt-4o-mini'
    else:  # gemini
        model_options = ['gemini-1.5-pro', 'gemini-1.5-flash']
        default_model = 'gemini-1.5-pro'
    
    answering_model = st.selectbox(
        "回答模型",
        options=model_options,
        index=0,
        help="用于生成答案的模型"
    )
    
    st.markdown("---")
    
    st.markdown("### ⚙️ 基础检索")
    
    top_n_retrieval = st.slider(
        "📊 最终检索数量",
        min_value=5,
        max_value=30,
        value=10,
        step=5,
        help="经过重排序后最终返回给LLM的文档块数量"
    )
    
    st.markdown("---")
    st.markdown("### 🚀 检索增强")
    
    use_hyde = st.checkbox(
        "✨ HYDE（假设性文档扩展）",
        value=True,
        help="生成假设性答案辅助检索，提高语义匹配度"
    )
    
    use_multi_query = st.checkbox(
        "🔄 Multi-Query（多查询扩展）",
        value=True,
        help="生成多个相关查询并行检索，提高召回率"
    )
    
    st.markdown("---")
    st.markdown("### 🎯 LLM重排序")
    
    llm_reranking = st.checkbox(
        "🧠 启用 LLM 重排序",
        value=True,
        help="使用LLM智能评估相关性并重新排序，显著提高精确度"
    )
    
    if llm_reranking:
        rerank_sample_size = st.slider(
            "🔍 初始召回数量",
            min_value=20,
            max_value=100,
            value=50,
            step=10,
            help="LLM重排序前先召回的候选chunks数量（更多=更全面但更慢）"
        )
        st.success(f"✅ **推荐配置**\n\n🎯 检索流程：召回 **{rerank_sample_size}** 个候选 → LLM重排序 → 返回前 **{top_n_retrieval}** 个")
        
        # 上游扩充配置
        st.markdown("---")
        st.markdown("### 🔄 上游扩充（推荐）")
        
        expand_upstream = st.checkbox(
            "📈 启用上游扩充",
            value=True,
            help="✨ 推荐开启！在答案生成前扩充页面组合，让LLM基于更完整的上下文生成高质量答案"
        )
        
        if expand_upstream:
            col1, col2 = st.columns(2)
            with col1:
                expand_top_k = st.slider(
                    "核心页面数",
                    min_value=3,
                    max_value=10,
                    value=5,
                    help="选取重排序后的前K个页面作为核心"
                )
            with col2:
                expand_context_size = st.slider(
                    "上下扩充页数",
                    min_value=1,
                    max_value=3,
                    value=2,
                    help="每个核心页面上下各扩充N页"
                )
            
            estimated_pages = expand_top_k * (2 * expand_context_size + 1)
            st.info(f"📊 **扩充预览**\n\n{expand_top_k} 个核心页 × {2*expand_context_size+1} 页/组 ≈ **{estimated_pages}** 页 → 去重后约 **20-40** 页")
            
            # Token估算和警告
            estimated_tokens = estimated_pages * 800  # 假设每页平均800 tokens
            if estimated_tokens > 25000:
                st.error(f"🚨 **Token超限警告**\n\n预计 **{estimated_tokens:,}** tokens，可能超过API限制！\n\n💡 **建议**：expand_top_k ≤ 5 或 expand_context_size = 1")
            elif estimated_tokens > 15000:
                st.warning(f"⚠️ **Token消耗较高**\n\n预计 **{estimated_tokens:,}** tokens，响应时间可能较长")
            else:
                st.success(f"✅ **Token消耗适中**\n\n预计 **{estimated_tokens:,}** tokens")
        else:
            expand_top_k = 5
            expand_context_size = 2
            st.info("💡 **下游扩充模式**\n\nLLM生成答案后扩充，仅用于展示参考资料（不影响答案质量）")
    else:
        rerank_sample_size = 10  # 不启用时默认值
        expand_upstream = False
        expand_top_k = 5
        expand_context_size = 2
    
    # 多轮对话设置
    st.markdown("---")
    st.markdown("### 💬 多轮对话设置")
    
    enable_multi_turn = st.checkbox(
        "启用多轮对话",
        value=False,
        help="启用后，系统会记住历史对话，理解上下文和指代关系（可能增加token消耗）",
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
            help="设置保留多少轮历史对话作为上下文（轮数越多，token消耗越大）",
            key="context_turns_slider"
        )
        st.session_state.context_turns = context_turns
        
        st.info(f"💡 当前将保留最近 **{context_turns}** 轮对话作为上下文")
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
        'expand_context_size': expand_context_size
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

# ==================== 主界面 ====================
st.title("🏢 金盘科技 RAG 问答系统")
st.markdown("基于 **FAISS + Qwen + 时间路由** 的智能财务问答系统")

# 初始化系统
if not st.session_state.initialized:
    if initialize_system():
        st.success("✅ 系统初始化成功！")
        st.rerun()
    else:
        st.stop()

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
            
            # 调用问答系统，传入真实的进度回调
            answer_dict = st.session_state.processor.get_answer_for_company(
                company_name=company_name,
                question=full_question,
                schema=schema_type,
                conversation_history=conversation_history,
                progress_callback=update_progress
            )
            
            # 完成
            import time
            status_text.text("✅ 处理完成！")
            progress_bar.progress(100)
            time.sleep(0.5)
            
            # 清除进度显示
            progress_bar.empty()
            status_text.empty()
            
            # 显示答案
            format_answer_display(answer_dict)
            
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
                - expand_context_size = 2
                - 预计Token: ~20,000 ✅
                """)
            else:
                st.error(f"❌ 处理问题时出错: {error_msg}")
            
            with st.expander("查看详细错误"):
                st.code(traceback.format_exc())

# 示例问题（从问题库加载）
st.markdown("---")
st.markdown("### 💡 投资者关注问题")
st.markdown("点击下方问题可自动填入输入框 | 共127个真实投资者问题")

# 加载问题库
try:
    questions_df = pd.read_csv("data/val_set/questions.csv")
    
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

# 页脚
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>🔧 金盘科技 RAG 问答系统 | 基于 FAISS + Qwen-max + 时间智能路由</p>
    <p>💡 支持多年份对比、智能检索增强（HYDE + Multi-Query + LLM重排序）</p>
</div>
""", unsafe_allow_html=True)
