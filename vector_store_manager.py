"""
向量库管理器
负责文档加载、切分、向量化和检索
使用BGE embedding模型和FAISS向量库
"""

import os
import json
import faiss
import numpy as np
import torch
from typing import List, Dict, Any, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import FAISS
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VectorStoreManager:
    """
    管理向量知识库的创建、加载和检索
    """

    def __init__(self, embedding_model_name: str = "BAAI/bge-large-zh-v1.5"):
        """
        初始化VectorStoreManager

        Args:
            embedding_model_name (str): HuggingFace上的BGE模型名称
        """
        self.embedding_model_name = embedding_model_name
        self.embedding_model = None
        self.vector_store = None
        self.documents = []

    def _load_embedding_model(self):
        """加载BGE嵌入模型"""
        if self.embedding_model is None:
            logger.info(f"开始加载BGE嵌入模型: {self.embedding_model_name}")
            try:
                # BGE模型需要设置`query_instruction`
                model_kwargs = {'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
                encode_kwargs = {'normalize_embeddings': True} # 归一化以获得更好的性能
                self.embedding_model = HuggingFaceBgeEmbeddings(
                    model_name=self.embedding_model_name,
                    model_kwargs=model_kwargs,
                    encode_kwargs=encode_kwargs,
                    query_instruction="为这个句子生成表示以用于检索相关文章："
                )
                logger.info("BGE嵌入模型加载成功")
            except Exception as e:
                logger.error(f"加载嵌入模型失败: {e}")
                raise

    def create_vector_store(self, documents: List[str], store_path: str):
        """
        根据给定的文档列表创建并保存FAISS向量库。

        Args:
            documents (List[str]): 文档内容的列表。
            store_path (str): 向量库的保存路径。
        """
        if not self.embedding_model:
            logger.error("Embedding model not loaded. Cannot create vector store.")
            return

        logger.info(f"开始为 {len(documents)} 个文档创建向量库...")

        # 步骤 1: 切分文档以满足200-300 token的要求
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=256,
            chunk_overlap=50,
            length_function=len,
            is_separator_regex=False,
        )
        split_docs = text_splitter.create_documents(documents)
        logger.info(f"文档被切分为 {len(split_docs)} 个chunks。")

        if not split_docs:
            logger.error("切分后没有可处理的文档块，向量库构建失败。")
            return

        # 步骤 2: 创建FAISS向量库并直接保存
        try:
            logger.info("开始生成文本向量并构建FAISS索引...")
            vector_store = FAISS.from_documents(documents=split_docs, embedding=self.embedding_model)
            logger.info("FAISS索引构建完成。")

            vector_store.save_local(store_path)
            logger.info(f"向量库已成功保存至: {store_path}")

        except Exception as e:
            logger.error(f"创建或保存FAISS向量库失败: {e}", exc_info=True)

    def load_vector_store(self, store_path: str):
        """
        从本地加载FAISS向量库

        Args:
            store_path (str): 向量库保存路径
        """
        if not os.path.exists(store_path):
            logger.error(f"向量库路径不存在: {store_path}")
            raise FileNotFoundError(f"向量库路径不存在: {store_path}")

        logger.info(f"从 {store_path} 加载FAISS向量库...")
        
        # 加载嵌入模型
        self._load_embedding_model()

        try:
            self.vector_store = FAISS.load_local(
                folder_path=store_path,
                embeddings=self.embedding_model,
                allow_dangerous_deserialization=True # FAISS需要此项
            )
            logger.info("FAISS向量库加载成功")
        except Exception as e:
            logger.error(f"加载FAISS向量库失败: {e}")
            raise

    def search(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        """
        在向量库中进行相似性搜索

        Args:
            query (str): 查询语句
            k (int): 返回最相似结果的数量

        Returns:
            List[Dict[str, Any]]: 检索到的文档块列表，包含内容和元数据
        """
        if self.vector_store is None:
            logger.error("向量库未加载，无法执行搜索")
            return []

        logger.info(f"执行相似性搜索: '{query[:50]}...' (k={k})")
        try:
            # FAISS的similarity_search_with_score返回(Document, score)元组
            results_with_scores = self.vector_store.similarity_search_with_score(query, k=k)
            
            # 格式化输出
            formatted_results = []
            for doc, score in results_with_scores:
                formatted_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score)  # score越小越好（L2距离）
                })
            
            return formatted_results
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return []

def main_test():
    """
    测试VectorStoreManager功能
    """
    logger.info("=== 开始测试 VectorStoreManager ===")
    
    # 0. 初始化管理器
    manager = VectorStoreManager()
    
    # 1. 准备测试数据
    test_docs = [
        "人工智能（AI）是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。",
        "深度学习是机器学习的一个子领域，它使用多层神经网络来模拟人脑的分析和学习过程。",
        "BGE（BAAI General Embedding）是由北京智源人工智能研究院开发的一系列通用嵌入模型。",
        "FAISS (Facebook AI Similarity Search) 是一个用于高效相似性搜索和密集向量聚类的库。",
        "自然语言处理（NLP）是人工智能和语言学领域的分支学科，它研究如何让计算机理解和生成人类语言。"
    ]
    store_directory = "./test_vector_store"

    # 2. 创建并保存向量库
    logger.info("\n--- 测试创建和保存向量库 ---")
    if not os.path.exists(store_directory):
        manager.create_vector_store(test_docs, store_directory)
    else:
        logger.info("向量库已存在，跳过创建。")

    # 3. 加载向量库
    logger.info("\n--- 测试加载向量库 ---")
    manager.load_vector_store(store_directory)

    # 4. 执行搜索
    logger.info("\n--- 测试相似性搜索 ---")
    query = "什么是AI？"
    search_results = manager.search(query, k=2)
    
    print(f"\n查询: '{query}'")
    print("搜索结果:")
    for i, result in enumerate(search_results):
        print(f"  {i+1}. Score: {result['score']:.4f}")
        print(f"     Content: {result['content'][:80]}...")
        
    query_2 = "介绍一下FAISS"
    search_results_2 = manager.search(query_2, k=2)

    print(f"\n查询: '{query_2}'")
    print("搜索结果:")
    for i, result in enumerate(search_results_2):
        print(f"  {i+1}. Score: {result['score']:.4f}")
        print(f"     Content: {result['content'][:80]}...")

    logger.info("\n=== VectorStoreManager 测试完成 ===")

if __name__ == "__main__":
    main_test() 
