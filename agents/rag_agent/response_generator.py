import logging
from typing import List, Dict, Any, Optional, Union

class ResponseGenerator:
    """
    根据检索到的上下文和用户查询生成响应。
    """
    def __init__(self, config):
        """
            初始化响应生成器。

            参数:
                config：配置对象
                llm：用于响应生成的大型语言模型
        """
        self.logger = logging.getLogger(__name__)
        self.response_generator_model = config.rag.response_generator_model
        self.include_sources = getattr(config.rag, "include_sources", True)

    def _build_prompt(
            self,
            query: str, 
            context: str,
            chat_history: Optional[List[Dict[str, str]]] = None
        ) -> str:
        """
            构建语言模型的提示符。

            参数:
                query：用户查询
                context：来自检索文档的格式化上下文
                chat_history：可选的聊天记录

            返回:
                完整提示字符串
        """

        table_instructions = """
        一些检索到的信息以表格格式显示。当使用表中的信息时：
        1. 使用适当的带标头的markdown表格格式显示表格数据，如下所示：
            | Column1 | Column2 | Column3 |
            |---------|---------|---------|
            | Value1  | Value2  | Value3  |
        2. 重新格式化表格结构，使其更容易阅读和理解
        3. 如果在表的重新格式化过程中引入了任何新组件，请显式地提及它
        4. 在你的回答中清楚地解释表格数据
        5. 在展示具体数据点时，请参考相关表格
        6. 如果合适，总结表中显示的趋势或模式
        7. 如果只提到了参考文献号，并且可以从上下文中获取相应的值，如研究论文标题或作者，则将参考文献号替换为实际值
        """

        response_format_instructions = """产品说明:
        1. 仅根据上下文中提供的信息回答查询。
        2. 如果上下文不包含回答查询的相关信息，则声明：“根据提供的上下文，我没有足够的信息来回答这个问题。”
        3. 不要使用上下文中没有包含的先验知识。
        5. 要简洁准确。
        6. 如果需要，根据检索到的知识，提供结构良好的标题、副标题和表格结构的响应。标题和副标题尽量小。
        7. 只提供在聊天机器人回复中有意义的部分。例如，不要明确提及引用。
        8. 如果涉及到值，请确保使用上下文中的完美值进行响应。不要编造值。
        9. 不要在回答或回答中重复问题。"""
            
        # Build the prompt
        prompt = f"""您是一名医疗助理，根据经过验证的医疗来源提供准确的信息。

        以下是我们谈话的最后几句话：
        
        {chat_history}

        用户问了以下问题：
        {query}

        我检索了以下信息来帮助回答这个问题：

        {context}

        {table_instructions}

        {response_format_instructions}

        根据所提供的信息，请详尽而简洁地回答用户的问题。
        如果信息中不包含答案，请承认可用信息的局限性。
        
        不要提供上下文中没有的任何源链接。不要编造任何源链接。
        
        医务助理回应："""

        return prompt

    def generate_response(
            self,
            query: str,
            retrieved_docs: List[Dict[str, Any]],
            picture_paths: List[str],
            chat_history: Optional[List[Dict[str, str]]] = None,
        ) -> Dict[str, Any]:
        """
            根据检索到的文档生成响应。

            参数:
                query：用户查询
                retrieved_docs：检索的文档字典列表
                chat_history：可选的聊天记录

            返回:
                包含响应文本和源信息的字典
        """
        try:
           
            # 从文档中提取上下文内容
            doc_texts = [doc["content"] for doc in retrieved_docs]
            
            # 将检索到的文档组合到单个上下文中
            context = "\n\n===DOCUMENT SECTION===\n\n".join(doc_texts)
            
            # 构建提示符
            prompt = self._build_prompt(query, context, chat_history)
            
            # 生成响应
            response = self.response_generator_model.invoke(prompt)
            
            # 摘录引文来源
            sources = self._extract_sources(retrieved_docs) if hasattr(self, 'include_sources') and self.include_sources else []
            
            # 计算的置信度
            confidence = self._calculate_confidence(retrieved_docs)

            # 向响应添加源
            if hasattr(self, 'include_sources') and self.include_sources:
                response_with_source = response.content + "\n\n##### Source documents:"
                for current_source in sources:
                    source_path = current_source['path']
                    source_title = current_source['title']
                    response_with_source += f"\n- [{source_title}]({source_path})"
            else:
                response_with_source = response.content
            
            # Add picture paths to response
            response_with_source_and_picture_paths = response_with_source + "\n\n##### Reference images:"
            for picture_path in picture_paths:
                response_with_source_and_picture_paths += f"\n- [{picture_path.split('/')[-1]}]({picture_path})"
            
            # Format final response
            result = {
                "response": response_with_source_and_picture_paths,
                "sources": sources,
                "confidence": confidence
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return {
                "response": "I apologize, but I encountered an error while generating a response. Please try rephrasing your question.",
                "sources": [],
                "confidence": 0.0
            }

    def _extract_sources(self, documents: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        从检索到的文档中提取源信息以供引用。

        参数:
            documents：检索的文档字典列表

        返回:
            源信息字典列表
        """
        sources = []
        seen_sources = set()  # 跟踪唯一的来源以避免重复
        
        for doc in documents:
            # 提取source和source_path
            source = doc.get("source")
            source_path = doc.get("source_path")
            
            # 如果没有可用的源信息，则跳过
            if not source:
                continue
                
            # 为此源创建唯一标识符
            source_id = f"{source}|{source_path}"
            
            # 如果我们已经包含了这个源代码，请跳过
            if source_id in seen_sources:
                continue
                
            # Add to our sources list
            source_info = {
                "title": source,
                "path": source_path,
                "score": doc.get("combined_score", doc.get("rerank_score", doc.get("score", 0.0)))
            }
            
            sources.append(source_info)
            seen_sources.add(source_id)
        
        # 按分数从高到低排序来源
        sources.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        # 格式化最终源列表，删除仅用于排序的分数
        formatted_sources = []
        for source in sources:
            formatted_source = {
                "title": source["title"],
                "path": source["path"]
            }
            formatted_sources.append(formatted_source)
            
        return formatted_sources

    def _calculate_confidence(self, documents: List[Dict[str, Any]]) -> float:
        """
        根据检索到的文档计算置信度分数。

        参数:
        documents：检索到的文档

        返回:
            置信度评分在0到1之间
        """
        if not documents:
            return 0.0
            
        # 如果可以使用组合分数（重新排序和余弦相似度），否则使用原始分数
        if "combined_score" in documents[0]:
            scores = [doc.get("combined_score", 0) for doc in documents[:3]]
        elif "rerank_score" in documents[0]:
            scores = [doc.get("rerank_score", 0) for doc in documents[:3]]
        else:
            scores = [doc.get("score", 0) for doc in documents[:3]]
            
        # 前3名文件得分的平均值，如果低于3分则更少
        return sum(scores) / len(scores) if scores else 0.0