import os
from .web_search_agent import WebSearchAgent
from typing import Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()

class WebSearchProcessor:
    """
    处理网络搜索结果并将其路由到相应的LLM以生成响应。
    """
    
    def __init__(self, config):
        self.web_search_agent = WebSearchAgent(config)
        
        # Initialize LLM for processing web search results
        self.llm = config.web_search.llm
    
    def _build_prompt_for_web_search(self, query: str, chat_history: List[Dict[str, str]] = None) -> str:
        """
        构建web搜索的提示符。

        参数:
            query：用户查询
            Chat_history：聊天记录

        返回:
            完整提示字符串
        """
        # Add chat history if provided
        # print("Chat History:", chat_history)
            
        # Build the prompt
        prompt = f"""以下是我们谈话的最后几句话：

        {chat_history}

        用户问了以下问题：

        {query}

        只有当过去的对话似乎与当前的查询相关时，才能将它们总结成一个单一的、格式良好的问题，以便用于网络搜索。
        保持简洁，确保它抓住了讨论背后的关键意图。
        """

        return prompt
    
    def process_web_results(self, query: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """
        获取网络搜索结果，使用LLM处理它们，并返回用户友好的响应。
        """
        # print(f"[WebSearchProcessor] Fetching web search results for: {query}")
        web_search_query_prompt = self._build_prompt_for_web_search(query=query, chat_history=chat_history)       # 构建web搜索的提示符
        # print("Web Search Query Prompt:", web_search_query_prompt)
        web_search_query = self.llm.invoke(web_search_query_prompt)              # 使用LLM处理web搜索的提示符
        # print("Web Search Query:", web_search_query)
        
        # Retrieve web search results
        web_results = self.web_search_agent.search(web_search_query.content)             # 获取web搜索结果

        # print(f"[WebSearchProcessor] Fetched results: {web_results}")
        
        # Construct prompt to LLM for processing the results
        llm_prompt = (
            "你是一个专门从事医疗信息的人工智能助手。以下是web搜索结果 "
            "为用户查询检索。总结并给出一个有用的、简洁的回答。 "
            "仅使用可靠的来源，并确保医疗准确性.\n\n"
            f"Query: {query}\n\nWeb Search Results:\n{web_results}\n\nResponse:"
        )
        
        # Invoke the LLM to process the results
        response = self.llm.invoke(llm_prompt)
        
        return response
