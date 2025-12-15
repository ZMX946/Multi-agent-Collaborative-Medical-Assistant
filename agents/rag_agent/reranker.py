import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from sentence_transformers import CrossEncoder

class Reranker:
    """
    使用交叉编码器模型对检索到的文档重新排序，以获得更准确的结果。
    """
    def __init__(self, config):
        """
        用configuration初始化rerank。

        参数:
            config：包含重新排序设置的配置对象
        """
        self.logger = logging.getLogger(__name__)

        # 加载交叉编码器模型重新排序
        # 对于医疗数据，使用像“pritamdeka/S-PubMedBert-MS-MARCO”这样的专用模型
        # 是最理想的，但为了简单起见，这里使用一个通用的模型
        try:
            self.model_name = config.rag.reranker_model
            self.logger.info(f"Loading reranker model: {self.model_name}")
            self.model = CrossEncoder(self.model_name)
            self.top_k = config.rag.reranker_top_k
        except Exception as e:
            self.logger.error(f"Error loading reranker model: {e}")
            raise
    
    def rerank(self, query: str, documents: Union[List[Dict[str, Any]], List[str]], parsed_content_dir: str) -> List[Dict[str, Any]]:
        """
            使用交叉编码器基于查询相关性重新排序文档。

            参数:
                query：用户查询
                documents：文档列表（字典）或字符串列表

            返回:
                重新排序的文件列表与更新的分数
        """
        try:
            if not documents:
                return []
            
            # 处理不同的文档格式，确保结构一致
            if documents:
                # 如果检索到的文档只是一个字符串列表，我们添加一个默认分数
                if isinstance(documents[0], str):
                    # 将简单字符串转换为字典
                    docs_list = []
                    for i, doc_text in enumerate(documents):
                        docs_list.append({
                            "id": i,
                            "content": doc_text,
                            "score": 1.0  # Default score
                        })
                    documents = docs_list
                # 如果检索的文档是一个字典列表，我们使用原始分数
                elif isinstance(documents[0], dict):
                    # 确保字典中存在所有必需的字段
                    for i, doc in enumerate(documents):
                        # Ensure ID exists
                        if "id" not in doc:
                            doc["id"] = i
                        # Ensure score exists
                        if "score" not in doc:
                            doc["score"] = 1.0
                        # Ensure content exists (unlikely to be missing but just in case)
                        if "content" not in doc:
                            if "text" in doc:  # Some implementations might use "text" instead
                                doc["content"] = doc["text"]
                            else:
                                doc["content"] = f"Document {i}"
            
            # 创建用于评分的查询文档对
            pairs = [(query, doc["content"]) for doc in documents]
            
            # 获得相关性评分
            scores = self.model.predict(pairs)
            
            # 为文档添加分数
            for i, score in enumerate(scores):
                documents[i]["rerank_score"] = float(score)  # Store the new score from reranking
                # 如果原始文档没有分数，则使用重新排序的分数
                if "score" not in documents[i]:
                    documents[i]["score"] = 1.0
                # 合并（平均）原始分数和重新排序分数
                documents[i]["combined_score"] = (documents[i]["score"] + float(score)) / 2
            
            # Sort by combined score
            reranked_docs = sorted(documents, key=lambda x: x["combined_score"], reverse=True)
            
            # Limit to top_k if needed
            if self.top_k and len(reranked_docs) > self.top_k:
                reranked_docs = reranked_docs[:self.top_k]
            
            # 提取图片参考
            picture_reference_paths = []
            for doc in reranked_docs:
                matches = re.finditer(r"picture_counter_(\d+)", doc["content"])
                for match in matches:
                    counter_value = int(match.group(1))
                    # 创建基于文档源和计数器的图片路径
                    doc_basename = os.path.splitext(doc['source'])[0]  # Remove file extension
                    # picture_path = Path(os.path.abspath(parsed_content_dir + "/" + f"{doc_basename}-picture-{counter_value}.png")).as_uri()
                    picture_path = os.path.join("http://localhost:8000/", parsed_content_dir + "/" + f"{doc_basename}-picture-{counter_value}.png")
                    picture_reference_paths.append(picture_path)
            
            return reranked_docs, picture_reference_paths
            
        except Exception as e:
            self.logger.error(f"Error during reranking: {e}")
            # Fallback to original ranking if reranking fails
            self.logger.warning("Falling back to original ranking")
            return documents