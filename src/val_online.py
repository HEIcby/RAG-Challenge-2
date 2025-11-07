#!/usr/bin/env python3
"""
Interactive Q&A script for val_set (金盘科技)
Allows users to ask questions in real-time and get answers from the RAG system.
"""

import sys
from pathlib import Path
from src.questions_processing import QuestionsProcessor
from src.pipeline import RunConfig
import json
from datetime import datetime

class ValOnline:
    def __init__(self, 
                 root_path: Path,
                 use_hyde: bool = True,
                 use_multi_query: bool = True,
                 llm_reranking: bool = True,
                 top_n_retrieval: int = 10,
                 api_provider: str = "qwen",
                 answering_model: str = "qwen-max"):
        """
        Initialize the interactive Q&A system for val_set.
        
        Args:
            root_path: Path to val_set directory
            use_hyde: Enable HYDE hypothetical document expansion
            use_multi_query: Enable multi-query expansion
            llm_reranking: Enable LLM-based reranking
            top_n_retrieval: Number of chunks to retrieve
            api_provider: API provider ("qwen", "openai", "gemini")
            answering_model: Model name for answering
        """
        self.root_path = root_path
        self.company_name = "金盘科技"
        
        # Initialize paths
        self.vector_db_dir = root_path / "databases" / "vector_dbs"
        self.documents_dir = root_path / "databases" / "chunked_reports"
        self.subset_path = root_path / "subset.csv"
        
        # Check if databases exist
        if not self.documents_dir.exists() or not self.vector_db_dir.exists():
            print("❌ 错误: 数据库不存在！")
            print(f"请先运行以下命令处理 PDF 文件:")
            print(f"  cd {root_path}")
            print(f"  python main.py parse-pdfs")
            print(f"  python main.py process-reports")
            sys.exit(1)
        
        # Initialize processor
        print("🔧 初始化问答系统...")
        print(f"📁 数据目录: {root_path}")
        print(f"🏢 公司: {self.company_name}")
        print(f"🤖 API提供商: {api_provider}")
        print(f"🧠 模型: {answering_model}")
        print(f"🔍 检索数量: {top_n_retrieval}")
        print(f"💡 HYDE: {'启用' if use_hyde else '禁用'}")
        print(f"🔄 Multi-Query: {'启用' if use_multi_query else '禁用'}")
        print(f"🎯 LLM重排序: {'启用' if llm_reranking else '禁用'}")
        if llm_reranking:
            print(f"   └─ 初始检索: 50个chunks (多文档时自动平均分配)")
        print()
        
        self.processor = QuestionsProcessor(
            vector_db_dir=self.vector_db_dir,
            documents_dir=self.documents_dir,
            questions_file_path=None,  # No file, we'll ask interactively
            new_challenge_pipeline=True,
            subset_path=self.subset_path,
            parent_document_retrieval=False,
            llm_reranking=llm_reranking,
            llm_reranking_sample_size=50 if llm_reranking else 10,  # 增加到50以确保多文档时每个都能被充分检索
            top_n_retrieval=top_n_retrieval,
            parallel_requests=1,  # Sequential for interactive mode
            api_provider=api_provider,
            answering_model=answering_model,
            full_context=False,
            use_hyde=use_hyde,
            use_multi_query=use_multi_query
        )
        
        print("✅ 系统初始化完成！\n")
    
    def format_answer(self, answer_dict: dict) -> str:
        """Format the answer for display."""
        output = []
        output.append("=" * 80)
        output.append("📊 答案")
        output.append("=" * 80)
        
        # Main answer - check both 'final_answer' and 'answer' fields
        answer = answer_dict.get("final_answer", answer_dict.get("answer", "N/A"))
        output.append(f"💡 答案: {answer}")
        output.append("")
        
        # Step by step analysis
        if "step_by_step_analysis" in answer_dict:
            output.append("🔍 分析过程:")
            output.append("-" * 80)
            analysis = answer_dict["step_by_step_analysis"]
            if isinstance(analysis, list):
                for i, step in enumerate(analysis, 1):
                    output.append(f"{i}. {step}")
            else:
                output.append(str(analysis))
            output.append("")
        
        # Reasoning summary
        if "reasoning_summary" in answer_dict:
            output.append("📝 推理总结:")
            output.append("-" * 80)
            output.append(answer_dict["reasoning_summary"])
            output.append("")
        
        # References
        if "references" in answer_dict:
            refs = answer_dict["references"]
            if refs:
                output.append("📚 参考来源:")
                output.append("-" * 80)
                for i, ref in enumerate(refs, 1):
                    sha1 = ref.get("pdf_sha1", "N/A")[:8]
                    page = ref.get("page_index", "N/A")
                    output.append(f"{i}. 文档: {sha1}... | 页码: {page}")
                output.append("")
        
        # Source SHA1 (if available)
        if "source_sha1" in answer_dict:
            output.append(f"📄 来源文档: {answer_dict['source_sha1']}")
            output.append("")
        
        output.append("=" * 80)
        return "\n".join(output)

    def ask_question(self, question: str, schema: str = "jingpan") -> dict:
        """
        Ask a question and get an answer.
        
        Args:
            question: The question to ask
            schema: Expected answer type ("jingpan", "number", "boolean", "name")
        
        Returns:
            Dictionary containing the answer and related information
        """
        # Ensure company name is in the question
        if self.company_name not in question:
            question = f"{self.company_name}{question}"
        
        print(f"❓ 问题: {question}")
        print(f"📝 答案类型: {schema}")
        print("⏳ 处理中...\n")
        
        try:
            # Get answer
            answer_dict = self.processor.get_answer_for_company(
                company_name=self.company_name,
                question=question,
                schema=schema
            )
            
            return answer_dict
            
        except Exception as e:
            print(f"❌ 错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}
    
    def interactive_mode(self):
        """Run in interactive mode, continuously asking for questions."""
        print("🎯 交互式问答模式")
        print("=" * 80)
        print(f"📌 当前公司: {self.company_name}")
        print("📌 可以直接输入问题，系统会自动添加公司名称")
        print("📌 默认使用 'jingpan' schema（中文财务问答专用）")
        print("📌 输入 'quit' 或 'exit' 退出")
        print("📌 输入 'save' 保存历史记录")
        print("=" * 80)
        print()
        
        history = []
        
        while True:
            try:
                # Get question from user
                question = input("💬 请输入问题 (或命令): ").strip()
                
                if not question:
                    continue
                
                # Check for exit commands
                if question.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 再见！")
                    break
                
                # Check for save command
                if question.lower() == 'save':
                    self.save_history(history)
                    continue
                
                # Ask for schema (optional)
                print("📝 答案类型 (直接回车使用 'jingpan')【可选项目['jingpan', 'number', 'boolean', 'name']】: ", end="")
                schema_input = input().strip().lower()
                schema = schema_input if schema_input in ['jingpan', 'number', 'boolean', 'name'] else 'jingpan'
                
                print()
                
                # Get answer
                answer_dict = self.ask_question(question, schema)

                print("answer_dict:", answer_dict)
                
                # Display answer
                print(self.format_answer(answer_dict))
                print()
                
                # Save to history
                history.append({
                    "timestamp": datetime.now().isoformat(),
                    "question": question,
                    "schema": schema,
                    "answer": answer_dict
                })

                _ = input("输入任意值开启下一轮对话✅ ").strip()
                
            except KeyboardInterrupt:
                print("\n\n👋 再见！")
                break
            except Exception as e:
                print(f"❌ 错误: {str(e)}")
                import traceback
                traceback.print_exc()
                print()
    
    def batch_mode(self, questions: list):
        """
        Process a batch of questions.
        
        Args:
            questions: List of dicts with 'question' and optionally 'schema'
        """
        print("📦 批量处理模式")
        print(f"📊 总问题数: {len(questions)}")
        print("=" * 80)
        print()
        
        results = []
        
        for i, q in enumerate(questions, 1):
            question = q.get("question", q.get("text", ""))
            schema = q.get("schema", q.get("kind", "jingpan"))  # Default to jingpan
            
            print(f"[{i}/{len(questions)}] 处理中...")
            answer_dict = self.ask_question(question, schema)
            print(self.format_answer(answer_dict))
            print()
            
            results.append({
                "question": question,
                "schema": schema,
                "answer": answer_dict
            })
        
        return results
    
    def save_history(self, history: list):
        """Save question history to a JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.root_path / f"qa_history_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 历史记录已保存到: {filename}\n")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Interactive Q&A for 金盘科技 (val_set)")
    parser.add_argument("--root", type=str, default="data/val_set",
                       help="Root path to val_set directory (default: data/val_set)")
    parser.add_argument("--mode", type=str, default="interactive",
                       choices=["interactive", "batch"],
                       help="Run mode: interactive or batch (default: interactive)")
    parser.add_argument("--questions-file", type=str, default=None,
                       help="JSON file with questions for batch mode")
    parser.add_argument("--hyde", action="store_true", default=True,
                       help="Enable HYDE expansion (default: True)")
    parser.add_argument("--no-hyde", action="store_false", dest="hyde",
                       help="Disable HYDE expansion")
    parser.add_argument("--multi-query", action="store_true", default=True,
                       help="Enable multi-query expansion (default: True)")
    parser.add_argument("--no-multi-query", action="store_false", dest="multi_query",
                       help="Disable multi-query expansion")
    parser.add_argument("--rerank", action="store_true", default=True,
                       help="Enable LLM reranking (default: True)")
    parser.add_argument("--no-rerank", action="store_false", dest="rerank",
                       help="Disable LLM reranking")
    parser.add_argument("--top-n", type=int, default=10,
                       help="Number of chunks to retrieve (default: 10)")
    parser.add_argument("--api-provider", type=str, default="qwen",
                       choices=["qwen", "openai", "gemini"],
                       help="API provider (default: qwen)")
    parser.add_argument("--model", type=str, default="qwen-max",
                       help="Model name (default: qwen-max)")
    
    args = parser.parse_args()
    
    # Convert root path to Path object
    root_path = Path(args.root)
    
    if not root_path.exists():
        print(f"❌ 错误: 目录不存在: {root_path}")
        sys.exit(1)
    
    # Initialize system
    val_online = ValOnline(
        root_path=root_path,
        use_hyde=args.hyde,
        use_multi_query=args.multi_query,
        llm_reranking=args.rerank,
        top_n_retrieval=args.top_n,
        api_provider=args.api_provider,
        answering_model=args.model
    )
    
    # Run in selected mode
    if args.mode == "interactive":
        val_online.interactive_mode()
    elif args.mode == "batch":
        if not args.questions_file:
            print("❌ 错误: 批量模式需要 --questions-file 参数")
            sys.exit(1)
        
        with open(args.questions_file, 'r', encoding='utf-8') as f:
            questions = json.load(f)
        
        results = val_online.batch_mode(questions)
        
        # Save results
        output_file = root_path / f"answers_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 结果已保存到: {output_file}")


if __name__ == "__main__":
    main()
