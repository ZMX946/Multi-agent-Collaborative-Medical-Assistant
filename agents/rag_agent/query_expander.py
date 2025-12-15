import logging
from typing import List, Dict, Any

class QueryExpander:
    """
    用医学术语扩展用户查询，以改进检索。
    """
    def __init__(self, config):
        self.logger = logging.getLogger(f"{self.__module__}")
        self.config = config
        self.model = config.rag.llm
        
    def expand_query(self, original_query: str) -> Dict[str, Any]:
        """
        用相关的医学术语展开原始查询。

        参数:
            original_query：用户的原始查询

        返回:
            具有原始和扩展查询的字典
        """
        self.logger.info(f"Expanding query: {original_query}")
        
        # 生成扩展-执行以下策略之一
        expanded_query = self._generate_expansions(original_query)
        
        return {
            "original_query": original_query,
            "expanded_query": expanded_query.content
        }
    
    def _generate_expansions(self, query: str) -> str:
        """使用LLM用医学术语扩展查询。"""
        prompt = f"""
        作为医学专家，用相关医学术语扩展以下查询：
        同义词：有助于检索相关医学信息的同义词和相关概念：
        
        用户查询：{query}
        
        只有在需要时才展开查询，否则保持用户查询完整。
        请指定医疗或查询中提到的任何其他域，不要添加其他医疗域。
        如果用户查询要求以表格形式回答问题，请将其包含在扩展查询中，而不要自己以表格形式回答问题。
        只提供扩展后的查询而不提供解释。
        """
        expansion = self.model.invoke(prompt)
        
        return expansion