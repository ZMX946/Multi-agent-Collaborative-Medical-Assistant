import requests
from langchain_community.tools.tavily_search import TavilySearchResults

class TavilySearchAgent:
    """
    使用上下文感知分块处理RAG系统的一般文档。
    """
    def __init__(self):
        """
        初始化Tavily搜索代理。

        参数:
            query：用户查询
        """
        pass

    def search_tavily(self, query: str) -> str:
        """使用Tavily API执行一般的web搜索。"""

        tavily_search = TavilySearchResults(max_results = 5)     # 初始化Tavily搜索工具，用于执行网络搜索并获取结果

        # url = "https://api.tavily.com/search"
        # params = {
        #     "api_key": tavily_api_key,
        #     "query": query,
        #     "num_results": 5
        # }
        
        try:
            # response = requests.get(url, params=params)
            # Strip any surrounding quotes from the query
            query = query.strip('"\'')           # 去除引号
            # print("Printing query:", query)
            search_docs = tavily_search.invoke(query)         # 调用Tavily搜索工具
            # data = response.json()
            # if "results" in data:
            if len(search_docs):            # 检查结果
                return "\n".join(["title: " + str(res["title"]) + " - " + 
                                  "url: " + str(res["url"]) + " - " + 
                                  "content: " + str(res["content"]) + " - " + 
                                  "score: " + str(res["score"]) for res in search_docs])             # 返回结果
            return "No relevant results found."
        except Exception as e:
            return f"Error retrieving web search results: {e}"