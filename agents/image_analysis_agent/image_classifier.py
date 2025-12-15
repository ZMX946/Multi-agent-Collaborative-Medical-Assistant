import os
import json
import base64
from mimetypes import guess_type

from typing import TypedDict
from langchain_core.output_parsers import JsonOutputParser

class ClassificationDecision(TypedDict):
    """决策代理的输出结构."""
    image_type: str    # 图像类型
    reasoning: str     # 推理过程
    confidence: float  # 置信度

class ImageClassifier:
    """使用GPT视觉模型分析图像并确定其类型."""
    
    def __init__(self, vision_model):
        self.vision_model = vision_model     # LLM
        self.json_parser = JsonOutputParser(pydantic_object=ClassificationDecision)    # JSON解析器
        
    def local_image_to_data_url(self, image_path: str) -> str:
        """
        获取本地图片的URL
        """
        mime_type, _ = guess_type(image_path)           # 获取图片的MIME类型

        if mime_type is None:                           # 如果无法确定MIME类型
            mime_type = "application/octet-stream"      # 默认为二进制流

        with open(image_path, "rb") as image_file:      # 打开图片文件
            base64_encoded_data = base64.b64encode(image_file.read()).decode("utf-8")        # 将图片数据编码为Base64

        return f"data:{mime_type};base64,{base64_encoded_data}"
    
    def classify_image(self, image_path: str) -> str:
        """分析图像以将其分类为医学图像并确定其类型."""
        print(f"[ImageAnalyzer] Analyzing image: {image_path}")

        vision_prompt = [
            {"role": "system", "content": "您是医学影像领域的专家。请分析上传的图像。"},        # 系统提示
            {"role": "user", "content": [
                {"type": "text", "text": (
                    """
                    判断此图像是否为医学影像。若为医学影像，请将其分类为：
                    ‘脑部MRI扫描’、'胸部X光片'、‘皮肤病变'或'其他’。若非医学影像，则返回'非医学影像'。
                    您的回答必须采用以下结构的JSON格式：
                    {{
                    “image_type”: “图像类型”,
                    “reasoning”: “选择该代理的逐步推理过程”,
                    “confidence”: 0.95  // 0.0至1.0之间的数值，表示对本次分类任务的置信度
                    }}
                    """
                )},
                {"type": "image_url", "image_url": {"url": self.local_image_to_data_url(image_path)}}  # 图片URL
            ]}
        ]
        
        # 调用大型语言模型对图像进行分类
        response = self.vision_model.invoke(vision_prompt)

        try:
            # 确保响应被解析为 JSON
            response_json = self.json_parser.parse(response.content)
            return response_json  # 返回字典而非字符串
        except json.JSONDecodeError:
            print("[ImageAnalyzer] Warning: Response was not valid JSON.")
            return {"image_type": "unknown", "reasoning": "Invalid JSON response", "confidence": 0.0}

        # return response.content.strip().lower()
