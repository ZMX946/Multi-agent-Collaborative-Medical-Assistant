import requests

class PubmedSearchAgent:
    """
    使用上下文感知分块处理RAG系统的医疗文档。
    """
    def __init__(self):
        """
        初始化Pubmed搜索代理。

        参数:
            query：用户查询
        """
        pass

    def search_pubmed(self, pubmed_api_url, query: str) -> str:
        """在PubMed搜索相关的医学文章。"""
        params = {
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": 5
        }
        
        try:
            response = requests.get(pubmed_api_url, params=params)             # 发送HTTP GET请求
            data = response.json()              # 解析响应数据
            article_ids = data.get("esearchresult", {}).get("idlist", [])             # 获取相关文章的ID列表
            if not article_ids:             # 如果没有相关文章
                return "No relevant PubMed articles found."
            
            article_links = [f"https://pubmed.ncbi.nlm.nih.gov/{article_id}/" for article_id in article_ids]             # 生成相关文章的链接
            return "\n".join(article_links)                # 返回相关文章的链接
        except Exception as e:
            return f"Error retrieving PubMed articles: {e}"