#!/usr/bin/env python3
"""
单文档测试脚本 - 只处理一个文档进行测试
"""

from pathlib import Path
from generate_similarity_matrix import SimilarityMatrixGenerator

def main():
    """测试单个文档"""
    print("\n" + "="*70)
    print("  单文档测试 - J2023 (最小的 FAISS 文件)")
    print("="*70)
    
    # 设置路径
    base_path = Path(__file__).parent.parent / "data" / "val_set"
    vector_db_dir = base_path / "databases" / "vector_dbs"
    documents_dir = base_path / "databases" / "chunked_reports"
    output_dir = Path(__file__).parent / "outputs"
    
    # 检查路径
    print(f"\n📂 检查数据路径...")
    if not vector_db_dir.exists():
        print(f"❌ 向量数据库目录不存在: {vector_db_dir}")
        return
    print(f"✓ 向量数据库目录: {vector_db_dir}")
    
    if not documents_dir.exists():
        print(f"❌ 文档目录不存在: {documents_dir}")
        return
    print(f"✓ 文档目录: {documents_dir}")
    print(f"✓ 输出目录: {output_dir}")
    
    # 创建生成器
    generator = SimilarityMatrixGenerator(vector_db_dir, documents_dir, output_dir)
    
    # 只处理 J2023（最小的文档）
    test_doc = "J2023"
    print(f"\n🧪 测试文档: {test_doc}")
    print("="*70)
    
    try:
        generator.process_document(test_doc)
        print(f"\n✅ 测试成功！")
        print(f"📁 输出文件: {output_dir / f'{test_doc}_similarity_matrix.html'}")
        print(f"💡 用浏览器打开 HTML 文件查看交互式热度图\n")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

