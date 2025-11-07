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
                llm_reranking_sample_size=50 if config['llm_reranking'] else 10,
                top_n_retrieval=config['top_n_retrieval'],
                parallel_requests=1,
                api_provider=config['api_provider'],
                answering_model=config['answering_model'],
                full_context=False,
                use_hyde=config['use_hyde'],
                use_multi_query=config['use_multi_query']
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
    tab1, tab2, tab3 = st.tabs(["🔍 分析过程", "📝 推理总结", "📚 参考来源"])
    
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
            
            # 按文档分组并按页码排序
            from collections import defaultdict
            doc_groups = defaultdict(list)
            for ref in refs:
                sha1 = ref.get("pdf_sha1", "")
                page = ref.get("page_index", "N/A")
                chunk_text = ref.get("chunk_text", "")
                if sha1 and page != "N/A":
                    doc_groups[sha1].append({
                        'page': page,
                        'text': chunk_text
                    })
            
            # 按文档显示，每个文档内部按页码排序
            for doc_sha1, pages_data in doc_groups.items():
                # 按页码排序
                pages_data.sort(key=lambda x: x['page'])
                
                # 显示文档标题
                st.markdown(f"### 📄 文档 {doc_sha1[:8]}... ({len(pages_data)}个引用)")
                
                # 为每个页码显示图片和文本
                for idx, page_data in enumerate(pages_data, 1):
                    page_num = page_data['page']
                    chunk_text = page_data['text']
                    
                    with st.expander(f"引用 {idx}: 页码 {page_num}", expanded=(idx == 1)):
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
    
    # 初始化默认配置
    if 'config' not in st.session_state:
        st.session_state.config = {
            'api_provider': 'qwen',
            'answering_model': 'qwen-max',
            'top_n_retrieval': 10,
            'use_hyde': True,  # ✅ 已改用 Qwen API
            'use_multi_query': True,  # ✅ 已改用 Qwen API
            'llm_reranking': True
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
    st.subheader("🔍 检索配置")
    
    top_n_retrieval = st.slider(
        "检索数量",
        min_value=5,
        max_value=30,
        value=10,
        step=5,
        help="每次检索返回的文档块数量"
    )
    
    use_hyde = st.checkbox(
        "启用 HYDE",
        value=True,
        help="假设性文档扩展，生成假设性答案辅助检索"
    )
    
    use_multi_query = st.checkbox(
        "启用 Multi-Query",
        value=True,
        help="多查询扩展，生成多个相关查询提高召回率"
    )
    
    llm_reranking = st.checkbox(
        "启用 LLM 重排序",
        value=True,
        help="使用LLM对检索结果重新排序，提高相关性"
    )
    
    if llm_reranking:
        st.info("🎯 启用重排序时，初始检索50个chunks，最终返回前N个")
    
    # 多轮对话设置
    st.markdown("---")
    st.markdown("### 💬 多轮对话设置")
    
    enable_multi_turn = st.checkbox(
        "启用多轮对话",
        value=st.session_state.enable_multi_turn,
        help="启用后，系统会记住历史对话，理解上下文和指代关系",
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
    
    # 更新配置
    st.session_state.config = {
        'api_provider': api_provider,
        'answering_model': answering_model,
        'top_n_retrieval': top_n_retrieval,
        'use_hyde': use_hyde,
        'use_multi_query': use_multi_query,
        'llm_reranking': llm_reranking
    }
    
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
            st.error(f"❌ 处理问题时出错: {str(e)}")
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
