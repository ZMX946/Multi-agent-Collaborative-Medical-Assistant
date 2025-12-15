import re
import logging
from typing import List, Dict, Any, Optional, Tuple

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI

class ContentProcessor:
    """
    处理已解析的内容——概括图像，创建基于LLM的语义块
    """
    def __init__(self, config):
        """
        初始化响应生成器。

        参数：
            llm：用于图像摘要的大型语言模型
        """
        self.logger = logging.getLogger(__name__)
        self.summarizer_model = config.rag.summarizer_model     # temperature 0.5
        self.chunker_model = config.rag.chunker_model     # temperature 0.0
    
    def summarize_images(self, images: List[str]) -> List[str]:
        """
        使用提供的模型对图像进行摘要，并处理错误。

        参数：
            images：图像路径列表
        返回值：
            图像摘要列表，其中包含用于表示处理失败图像的占位符。
        """
        
        prompt_template = """详细描述图像，同时保持简洁和重点。
                            就上下文而言，该图像是医学研究论文或研究论文的一部分
                            演示人工智能技术的使用，比如
                            机器学习和深度学习在诊断疾病或医疗报告中的应用。
                            要明确图形，如条形图，如果它们出现在图像中。
                            只总结图像中存在的内容，不添加任何额外的细节或评论。
                            仅当图像与上下文相关时才对其进行总结，明确返回“非信息性”
                            如果图像是一些与上下文无关的内容。"""

        messages = [
            (
                "user",
                [
                    {"type": "text", "text": prompt_template},
                    {
                        "type": "image_url",
                        "image_url": {"url": "{image}"},
                    },
                ],
            )
        ]

        prompt = ChatPromptTemplate.from_messages(messages)        # Create the prompt
        summary_chain = prompt | self.summarizer_model | StrOutputParser()      # Create the chain
        
        results = []
        for image in images:
            try:
                summary = summary_chain.invoke({"image": image})
                results.append(summary)
            except Exception as e:
                # Log the error if needed
                print(f"Error processing image: {str(e)}")
                # Add placeholder for the failed image
                results.append("no image summary")
        
        return results
    
    def format_document_with_images(self, parsed_document: Any, image_summaries: List[str]) -> str:
        """
        通过用图像摘要替换图像占位符来格式化已解析的文档。

        参数:
            parsed_document：来自doc_parser的解析文档
            image_summaries：图像摘要列表

        返回:
            带图像摘要的格式化文档文本
        """
        IMAGE_PLACEHOLDER = "<!-- image_placeholder -->"
        PAGE_BREAK_PLACEHOLDER = "<!-- page_break -->"
        
        formatted_parsed_document = parsed_document.export_to_markdown(
            page_break_placeholder=PAGE_BREAK_PLACEHOLDER, 
            image_placeholder=IMAGE_PLACEHOLDER
        )                                                                          # 格式化已解析的文档
        
        formatted_document = self._replace_occurrences(
            formatted_parsed_document, 
            IMAGE_PLACEHOLDER, 
            image_summaries
        )                                                                           # 用图像摘要替换图像占位符
        
        return formatted_document
    
    def _replace_occurrences(self, text: str, target: str, replacements: List[str]) -> str:
        """
        用相应的替换替换出现的目标占位符。

        参数:
            text：包含占位符的文本
        目标：要替换的占位符
            replacement：每次出现的替换列表

        返回:
            带替换的文本
        """
        result = text
        for counter, replacement in enumerate(replacements):
            if target in result:
                if replacement.lower() != 'non-informative':
                    result = result.replace(
                        target, 
                        f'picture_counter_{counter}' + ' ' + replacement, 
                        1
                    )                                                                    # 替换
                else:
                    result = result.replace(target, '', 1)
            else:
                # Instead of raising an error, just break the loop when no more occurrences are found
                break
        
        return result

    def chunk_document(self, formatted_document: str) -> List[str]:
        """
        将文档拆分为语义块。

        参数:
            formatted_document：格式化的文档文本
            模型：AzureChatOpenAI模型实例（如果没有提供，将创建一个）

        返回:
            文档块列表
        """
        
        # Split by section boundaries
        SPLIT_PATTERN = "\n#"
        chunks = formatted_document.split(SPLIT_PATTERN)
        
        chunked_text = ""
        for i, chunk in enumerate(chunks):
            if chunk.startswith("#"):
                chunk = f"#{chunk}"  # add the # back to the chunk
            chunked_text += f"<|start_chunk_{i}|>\n{chunk}\n<|end_chunk_{i}|>\n"
        
        # 基于llm的语义分块
        CHUNKING_PROMPT = """
        你是专门将文本分割成语义一致的部分的助手。
        
        文件全文如下：
        <document>
            {document_text}
        </document>
        
        <instructions>
            产品说明:
            1. 文本被分成几个块，每个块都标有<|start_chunk_X|>和<|end_chunk_X|>标签，其中X是块号。
            2. 确定应该分割的点，这样类似主题的连续块就可以保持在一起。
            3. 每个块必须在256到512个单词之间。
            4. 如果第1和第2块在一起，但第3块开始了一个新的主题，建议在第2块之后分开。
            5. 数据块必须按升序排列。
            6. 以“split_after: 3,5”的形式提供您的回答。
        </instructions>
        
        只响应您认为应该发生分割的块的id。
        你必须用至少一个分裂来回应。
        """.strip()
        
        formatted_chunking_prompt = CHUNKING_PROMPT.format(document_text=chunked_text)                           # 格式化分块提示
        chunking_response = self.chunker_model.invoke(formatted_chunking_prompt).content                     # 调用分块模型
        
        return self._split_text_by_llm_suggestions(chunked_text, chunking_response)
    
    def _split_text_by_llm_suggestions(self, chunked_text: str, llm_response: str) -> List[str]:
        """
        根据LLM建议的分割点分割文本。

        参数:
            chunked_text：带有块标记的文本
            llm_response：带有拆分建议的LLM响应

        返回:
            文档块列表
        """
        # 从LLM响应中提取分裂点
        split_after = [] 
        if "split_after:" in llm_response:
            split_points = llm_response.split("split_after:")[1].strip()
            split_after = [int(x.strip()) for x in split_points.replace(',', ' ').split()] 

        # 如果没有建议分割，则返回整个文本作为一个部分
        if not split_after:
            return [chunked_text]

        # 找出文本中所有的组块标记
        chunk_pattern = r"<\|start_chunk_(\d+)\|>(.*?)<\|end_chunk_\1\|>"
        chunks = re.findall(chunk_pattern, chunked_text, re.DOTALL)

        # 根据分裂点分组块
        sections = []
        current_section = [] 

        for chunk_id, chunk_text in chunks:
            current_section.append(chunk_text)
            if int(chunk_id) in split_after:
                sections.append("".join(current_section).strip())
                current_section = [] 
        
        # 如果最后一节不是空的，添加它
        if current_section:
            sections.append("".join(current_section).strip())

        return sections