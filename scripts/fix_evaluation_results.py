#!/usr/bin/env python3
"""
重新评估脚本 - 修复评估结果中reasoning为空的问题

该脚本会：
1. 扫描所有评估JSON文件
2. 识别reasoning为空的问题
3. 使用APIProcessor重新评估这些问题
4. 备份原文件并生成修复后的新文件
"""

import json
import sys
import shutil
from pathlib import Path
from typing import List, Dict
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.api_requests import APIProcessor


def find_evaluation_files(val_result_dir: Path) -> List[Path]:
    """查找所有评估JSON文件"""
    return sorted(val_result_dir.glob("evaluation_*.json"))


def backup_file(file_path: Path) -> Path:
    """备份文件，添加.backup后缀"""
    backup_path = file_path.with_suffix(file_path.suffix + ".backup")
    if backup_path.exists():
        print(f"  ⚠️  备份文件已存在，跳过备份: {backup_path.name}")
    else:
        shutil.copy2(file_path, backup_path)
        print(f"  ✅ 已备份: {backup_path.name}")
    return backup_path


def identify_failed_evaluations(data: Dict) -> List[int]:
    """识别所有reasoning为空或包含评估失败的问题索引"""
    failed_indices = []
    for i, result in enumerate(data.get("results", [])):
        reasoning = result.get("reasoning", "")
        score = result.get("score", 0.0)
        # 检查reasoning为空，或包含"评估失败"，或score=0.0且reasoning包含错误信息
        if (not reasoning or not reasoning.strip() or 
            "评估失败" in reasoning or 
            (score == 0.0 and ("评估返回的reasoning为空" in reasoning or "评估失败" in reasoning))):
            failed_indices.append(i)
    return failed_indices


def recalculate_statistics(data: Dict) -> Dict:
    """重新计算统计指标"""
    results = data.get("results", [])
    total_questions = len(results)
    
    if total_questions == 0:
        return data
    
    correct_count = sum(1 for r in results if r.get("is_correct", False))
    total_score = sum(r.get("score", 0.0) for r in results)
    
    data["evaluated_count"] = total_questions
    data["correct_count"] = correct_count
    data["accuracy"] = correct_count / total_questions if total_questions > 0 else 0.0
    data["average_score"] = total_score / total_questions if total_questions > 0 else 0.0
    
    return data


def fix_evaluation_file(
    file_path: Path,
    api_processor: APIProcessor,
    model: str = "qwen-turbo",
    dry_run: bool = False
) -> Dict[str, any]:
    """修复单个评估文件"""
    print(f"\n{'='*80}")
    print(f"📄 处理文件: {file_path.name}")
    print(f"{'='*80}")
    
    # 读取评估结果
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 识别需要修复的问题
    failed_indices = identify_failed_evaluations(data)
    
    if not failed_indices:
        print("  ✅ 没有发现需要修复的问题")
        return {
            "file": file_path.name,
            "total_questions": len(data.get("results", [])),
            "fixed_count": 0,
            "success_count": 0,
            "error_count": 0
        }
    
    print(f"  📊 发现 {len(failed_indices)} 个需要修复的问题")
    
    # 备份文件
    if not dry_run:
        backup_file(file_path)
    
    # 修复每个问题
    results = data.get("results", [])
    success_count = 0
    error_count = 0
    
    for idx, failed_idx in enumerate(failed_indices, 1):
        result = results[failed_idx]
        question = result.get("question", "")
        standard_answer = result.get("standard_answer", "")
        rag_answer = result.get("rag_answer", "")
        
        print(f"\n  [{idx}/{len(failed_indices)}] 修复问题: {question[:50]}...")
        print(f"    RAG答案: {rag_answer[:50]}...")
        
        if not rag_answer or rag_answer.strip() == "":
            print("    ⚠️  RAG答案为空，跳过")
            error_count += 1
            continue
        
        try:
            if dry_run:
                print("    [DRY RUN] 跳过实际评估")
                success_count += 1
            else:
                # 重新评估
                eval_result = api_processor.evaluate_answer(
                    question=question,
                    standard_answer=standard_answer,
                    rag_answer=rag_answer,
                    model=model
                )
                
                # 验证结果
                if not eval_result or not isinstance(eval_result, dict):
                    raise ValueError("评估结果为空或格式错误")
                
                score = eval_result.get("score", 0.0)
                reasoning = eval_result.get("reasoning", "")
                
                if not reasoning or not reasoning.strip():
                    raise ValueError("评估返回的reasoning仍为空")
                
                # 更新结果
                result["score"] = score
                result["reasoning"] = reasoning
                result["is_correct"] = score >= 0.8
                
                print(f"    ✅ 修复成功: score={score:.2f}, is_correct={result['is_correct']}")
                print(f"    📝 Reasoning: {reasoning[:80]}...")
                success_count += 1
                
        except Exception as e:
            print(f"    ❌ 修复失败: {str(e)}")
            error_count += 1
    
    # 重新计算统计指标
    if not dry_run and success_count > 0:
        data = recalculate_statistics(data)
        
        # 保存修复后的文件
        fixed_file_path = file_path.with_name(
            file_path.stem + "_fixed" + file_path.suffix
        )
        with open(fixed_file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"\n  💾 修复后的文件已保存: {fixed_file_path.name}")
        print(f"  📈 更新后的统计:")
        print(f"     - 正确数: {data['correct_count']}/{data['total_questions']}")
        print(f"     - 准确率: {data['accuracy']:.2%}")
        print(f"     - 平均得分: {data['average_score']:.3f}")
    
    return {
        "file": file_path.name,
        "total_questions": len(results),
        "failed_count": len(failed_indices),
        "fixed_count": success_count,
        "error_count": error_count
    }


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="修复评估结果中reasoning为空的问题")
    parser.add_argument(
        "--val-result-dir",
        type=str,
        default="data/val_set/val_result",
        help="评估结果目录路径（默认: data/val_set/val_result）"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen-turbo",
        help="评估使用的模型（默认: qwen-turbo）"
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="qwen",
        choices=["qwen", "openai", "gemini", "ibm"],
        help="API提供商（默认: qwen）"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="试运行模式，不实际修复文件"
    )
    parser.add_argument(
        "--file",
        type=str,
        help="只处理指定的文件（相对于val_result_dir）"
    )
    
    args = parser.parse_args()
    
    # 确定评估结果目录
    val_result_dir = Path(project_root) / args.val_result_dir
    if not val_result_dir.exists():
        print(f"❌ 错误: 评估结果目录不存在: {val_result_dir}")
        sys.exit(1)
    
    print(f"📂 评估结果目录: {val_result_dir}")
    print(f"🤖 评估模型: {args.model} ({args.provider})")
    if args.dry_run:
        print("🔍 试运行模式: 不会实际修改文件")
    print()
    
    # 查找评估文件
    if args.file:
        eval_files = [val_result_dir / args.file]
        if not eval_files[0].exists():
            print(f"❌ 错误: 文件不存在: {eval_files[0]}")
            sys.exit(1)
    else:
        eval_files = find_evaluation_files(val_result_dir)
    
    if not eval_files:
        print("❌ 未找到评估文件")
        sys.exit(1)
    
    print(f"📋 找到 {len(eval_files)} 个评估文件")
    
    # 初始化API处理器
    print("\n🔧 初始化API处理器...")
    api_processor = APIProcessor(provider=args.provider)
    print("✅ API处理器初始化完成\n")
    
    # 处理每个文件
    summary = []
    for eval_file in eval_files:
        try:
            result = fix_evaluation_file(
                eval_file,
                api_processor,
                model=args.model,
                dry_run=args.dry_run
            )
            summary.append(result)
        except Exception as e:
            print(f"\n❌ 处理文件 {eval_file.name} 时出错: {e}")
            import traceback
            traceback.print_exc()
            summary.append({
                "file": eval_file.name,
                "error": str(e)
            })
    
    # 打印总结
    print(f"\n{'='*80}")
    print("📊 修复总结")
    print(f"{'='*80}")
    
    total_failed = 0
    total_fixed = 0
    total_errors = 0
    
    for result in summary:
        if "error" in result:
            print(f"❌ {result['file']}: 处理出错 - {result['error']}")
        else:
            print(f"📄 {result['file']}:")
            print(f"   总问题数: {result['total_questions']}")
            print(f"   需要修复: {result.get('failed_count', 0)}")
            print(f"   成功修复: {result.get('fixed_count', 0)}")
            print(f"   修复失败: {result.get('error_count', 0)}")
            
            total_failed += result.get('failed_count', 0)
            total_fixed += result.get('fixed_count', 0)
            total_errors += result.get('error_count', 0)
    
    print(f"\n总计:")
    print(f"  - 需要修复的问题: {total_failed}")
    print(f"  - 成功修复: {total_fixed}")
    print(f"  - 修复失败: {total_errors}")
    
    if not args.dry_run and total_fixed > 0:
        print(f"\n✅ 修复完成！修复后的文件已保存（_fixed后缀）")
        print(f"   原文件已备份（.backup后缀）")


if __name__ == "__main__":
    main()

