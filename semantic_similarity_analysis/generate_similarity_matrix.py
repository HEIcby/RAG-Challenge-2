#!/usr/bin/env python3
"""
语义相似度矩阵生成器
为财报文档构建语义相似度矩阵，并生成交互式热度图可视化
"""

import faiss
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import plotly.graph_objects as go
from typing import Dict, Tuple
import time
import warnings
warnings.filterwarnings('ignore')


class SimilarityMatrixGenerator:
    """语义相似度矩阵生成器"""
    
    def __init__(self, vector_db_dir: Path, documents_dir: Path, output_dir: Path):
        """
        初始化生成器
        
        Args:
            vector_db_dir: FAISS 向量数据库目录
            documents_dir: 文档 JSON 目录
            output_dir: 输出目录
        """
        self.vector_db_dir = Path(vector_db_dir)
        self.documents_dir = Path(documents_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.stats = {}  # 存储统计信息
    
    def load_vectors_from_faiss(self, faiss_path: Path) -> np.ndarray:
        """
        从 FAISS 索引中提取所有向量（优化版，批量提取）
        
        Args:
            faiss_path: FAISS 文件路径
            
        Returns:
            向量数组 (n_vectors, dimension)
        """
        index = faiss.read_index(str(faiss_path))
        n_vectors = index.ntotal
        
        print(f"   - 提取 {n_vectors:,} 个向量（维度: {index.d}）...")
        
        # 优化：批量重构向量
        vectors = np.zeros((n_vectors, index.d), dtype=np.float32)
        batch_size = 1000  # 每次提取1000个向量
        
        with tqdm(total=n_vectors, desc="   提取向量", ncols=80, leave=False) as pbar:
            for start_idx in range(0, n_vectors, batch_size):
                end_idx = min(start_idx + batch_size, n_vectors)
                batch_ids = np.arange(start_idx, end_idx, dtype=np.int64)
                vectors[start_idx:end_idx] = index.reconstruct_batch(batch_ids)
                pbar.update(end_idx - start_idx)
        
        return vectors
    
    def compute_cosine_similarity_matrix(self, vectors: np.ndarray) -> np.ndarray:
        """
        计算余弦相似度矩阵（优化版）
        
        Args:
            vectors: 向量数组 (n_vectors, dimension)
            
        Returns:
            相似度矩阵 (n_vectors, n_vectors)
        """
        n = vectors.shape[0]
        
        # 估计内存使用
        matrix_size_mb = (n * n * 4) / (1024 * 1024)  # float32 = 4 bytes
        print(f"   - 矩阵大小: {n} × {n} (~{matrix_size_mb:.1f} MB)")
        
        # 归一化向量
        print("   - 归一化向量...")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        normalized_vectors = vectors / (norms + 1e-8)  # 避免除零
        
        # 计算余弦相似度 (点积) - 使用 @ 运算符（更快）
        print("   - 计算余弦相似度矩阵...")
        start_time = time.time()
        similarity_matrix = normalized_vectors @ normalized_vectors.T
        elapsed = time.time() - start_time
        print(f"   - 矩阵计算完成（耗时: {elapsed:.2f}秒）")
        
        # 确保对角线为 1，处理数值误差
        np.fill_diagonal(similarity_matrix, 1.0)
        
        # 限制范围在 [-1, 1]
        similarity_matrix = np.clip(similarity_matrix, -1.0, 1.0)
        
        return similarity_matrix
    
    def load_document_info(self, doc_path: Path) -> Dict:
        """
        加载文档元信息
        
        Args:
            doc_path: 文档 JSON 路径
            
        Returns:
            文档信息字典
        """
        with open(doc_path, 'r', encoding='utf-8') as f:
            doc = json.load(f)
        
        return {
            'name': doc['metainfo']['sha1_name'],
            'company': doc['metainfo'].get('company_name', 'Unknown'),
            'chunks_count': len(doc['content']['chunks']),
            'pages': doc['metainfo'].get('pages_amount', 0),
            'text_blocks': doc['metainfo'].get('text_blocks_amount', 0),
            'tables': doc['metainfo'].get('tables_amount', 0),
        }
    
    def generate_heatmap(
        self, 
        similarity_matrix: np.ndarray, 
        doc_info: Dict,
        output_path: Path
    ):
        """
        生成交互式热度图
        
        Args:
            similarity_matrix: 相似度矩阵
            doc_info: 文档信息
            output_path: 输出 HTML 文件路径
        """
        n = similarity_matrix.shape[0]
        
        # 创建 Plotly 热度图
        fig = go.Figure(data=go.Heatmap(
            z=similarity_matrix,
            x=list(range(n)),
            y=list(range(n)),
            colorscale=[
                [0.0, 'rgb(0, 0, 255)'],      # 蓝色 (低相似度)
                [0.5, 'rgb(255, 255, 255)'],  # 白色 (中等)
                [1.0, 'rgb(255, 0, 0)']       # 红色 (高相似度)
            ],
            zmid=0.5,  # 中间值设置为 0.5
            zmin=0,
            zmax=1,
            colorbar=dict(
                title=dict(text='相似度', side='right'),
                tickmode='linear',
                tick0=0,
                dtick=0.1
            ),
            hovertemplate='Chunk %{x} ↔ Chunk %{y}<br>相似度: %{z:.4f}<extra></extra>'
        ))
        
        # 计算统计信息
        # 排除对角线的统计
        mask = ~np.eye(n, dtype=bool)
        off_diagonal = similarity_matrix[mask]
        
        avg_similarity = np.mean(off_diagonal)
        median_similarity = np.median(off_diagonal)
        std_similarity = np.std(off_diagonal)
        max_similarity = np.max(off_diagonal)
        min_similarity = np.min(off_diagonal)
        
        # 存储统计信息
        self.stats[doc_info['name']] = {
            'chunks_count': n,
            'avg_similarity': float(avg_similarity),
            'median_similarity': float(median_similarity),
            'std_similarity': float(std_similarity),
            'max_similarity': float(max_similarity),
            'min_similarity': float(min_similarity),
            'company': doc_info['company'],
            'pages': doc_info['pages'],
            'text_blocks': doc_info['text_blocks'],
            'tables': doc_info['tables']
        }
        
        # 设置布局
        title_text = (
            f"<b>{doc_info['name']} - {doc_info['company']}</b><br>"
            f"<sub>Chunks: {n:,} | 平均相似度: {avg_similarity:.4f} | "
            f"中位数: {median_similarity:.4f} | 标准差: {std_similarity:.4f}</sub>"
        )
        
        fig.update_layout(
            title={
                'text': title_text,
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 16}
            },
            xaxis_title='Chunk 索引',
            yaxis_title='Chunk 索引',
            width=1200,
            height=1200,
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=True,
                tickfont=dict(size=10)
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=True,
                tickfont=dict(size=10),
                autorange='reversed'  # Y 轴从上到下
            ),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # 保存为 HTML
        print("   - 正在保存 HTML 文件...")
        save_start = time.time()
        fig.write_html(
            str(output_path),
            config={
                'displayModeBar': True,
                'displaylogo': False,
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': f'{doc_info["name"]}_similarity_matrix',
                    'height': 1200,
                    'width': 1200,
                    'scale': 2
                }
            }
        )
        save_elapsed = time.time() - save_start
        
        # 获取文件大小
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        
        print(f"✅ 生成热度图: {output_path.name}")
        print(f"   - Chunks 数量: {n:,}")
        print(f"   - 平均相似度: {avg_similarity:.4f}")
        print(f"   - 相似度范围: [{min_similarity:.4f}, {max_similarity:.4f}]")
        print(f"   - HTML 文件大小: {file_size_mb:.1f} MB")
        print(f"   - 保存耗时: {save_elapsed:.2f}秒")
    
    def process_document(self, doc_name: str):
        """
        处理单个文档
        
        Args:
            doc_name: 文档名称（不含扩展名）
        """
        doc_start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"📄 处理文档: {doc_name}")
        print(f"{'='*60}")
        
        # 文件路径
        faiss_path = self.vector_db_dir / f"{doc_name}.faiss"
        doc_path = self.documents_dir / f"{doc_name}.json"
        output_path = self.output_dir / f"{doc_name}_similarity_matrix.html"
        
        # 检查文件是否存在
        if not faiss_path.exists():
            print(f"❌ 未找到 FAISS 文件: {faiss_path}")
            return
        
        if not doc_path.exists():
            print(f"❌ 未找到文档文件: {doc_path}")
            return
        
        # 加载文档信息
        print("\n📖 [1/3] 加载文档信息...")
        doc_info = self.load_document_info(doc_path)
        
        # 从 FAISS 提取向量
        print("\n🔍 [2/3] 从 FAISS 提取向量...")
        vectors = self.load_vectors_from_faiss(faiss_path)
        
        # 计算相似度矩阵
        print("\n🧮 [3/3] 计算相似度矩阵...")
        similarity_matrix = self.compute_cosine_similarity_matrix(vectors)
        
        # 生成热度图
        print("\n🎨 生成交互式热度图...")
        self.generate_heatmap(similarity_matrix, doc_info, output_path)
        
        # 总耗时
        total_time = time.time() - doc_start_time
        print(f"\n⏱️  文档处理总耗时: {total_time:.2f}秒 ({total_time/60:.1f}分钟)")
    
    def process_all_documents(self):
        """处理所有文档"""
        overall_start_time = time.time()
        
        # 获取所有 FAISS 文件
        faiss_files = sorted(self.vector_db_dir.glob("*.faiss"))
        doc_names = [f.stem for f in faiss_files]
        
        print(f"\n{'='*60}")
        print(f"🚀 语义相似度矩阵生成器")
        print(f"{'='*60}")
        print(f"📂 输出目录: {self.output_dir}")
        print(f"📊 待处理文档: {len(doc_names)} 个")
        print(f"{'='*60}\n")
        
        success_count = 0
        failed_docs = []
        
        for idx, doc_name in enumerate(doc_names, 1):
            print(f"\n🔄 进度: [{idx}/{len(doc_names)}]")
            try:
                self.process_document(doc_name)
                success_count += 1
            except Exception as e:
                print(f"\n❌ 处理 {doc_name} 时出错: {e}")
                failed_docs.append(doc_name)
                import traceback
                traceback.print_exc()
        
        # 总结
        overall_time = time.time() - overall_start_time
        print(f"\n{'='*60}")
        print(f"✅ 处理完成！")
        print(f"{'='*60}")
        print(f"✓ 成功: {success_count}/{len(doc_names)} 个文档")
        if failed_docs:
            print(f"✗ 失败: {', '.join(failed_docs)}")
        print(f"⏱️  总耗时: {overall_time:.2f}秒 ({overall_time/60:.1f}分钟)")
        print(f"{'='*60}\n")
        
        # 保存统计信息
        stats_path = self.output_dir / "statistics.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, ensure_ascii=False, indent=2)
        print(f"📊 统计信息已保存至: {stats_path}")
        
        return self.stats


def main():
    """主函数"""
    print("\n" + "="*70)
    print("  语义相似度矩阵生成器 v2.0 (优化版)")
    print("  Semantic Similarity Matrix Generator")
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
    
    # 创建生成器并处理所有文档
    generator = SimilarityMatrixGenerator(vector_db_dir, documents_dir, output_dir)
    stats = generator.process_all_documents()
    
    print(f"\n🎉 所有任务完成！")
    print(f"📁 输出文件位置: {output_dir}")
    print(f"💡 提示: 用浏览器打开 HTML 文件即可查看交互式热度图\n")


if __name__ == "__main__":
    main()

