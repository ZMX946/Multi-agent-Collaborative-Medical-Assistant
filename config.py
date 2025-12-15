"""
多代理医疗聊天机器人的配置文件
"""

import os
from dotenv import load_dotenv
import openai

# 加载环境变量
load_dotenv()

# 配置全局 openai（阿里云 AI）
openai.api_key = os.getenv("ALIYUN_API_KEY")
openai.api_base = os.getenv("ALIYUN_API_ENDPOINT")
# openai.api_version = "2023-07-01-preview"  # 如果阿里云文档要求可打开


class AliLLM:
    def __init__(self, model_name="", temperature=0.3):
        self.model_name = model_name
        self.temperature = temperature

    def invoke(self, prompt: str):
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature
        )
        return response['choices'][0]['message']['content']


class AliEmbedding:
    def __init__(self, model_name="text-embedding-3-large"):
        self.model_name = model_name

    def embed(self, text: str):
        response = openai.Embedding.create(
            model=self.model_name,
            input=text
        )
        return response['data'][0]['embedding']

# ------------------------ 各类配置 ------------------------

class AgentDecisionConfig:
    def __init__(self):
        self.llm = AliLLM(model_name="", temperature=0.1)

class ConversationConfig:
    def __init__(self):
        self.llm = AliLLM(model_name="", temperature=0.7)

class WebSearchConfig:
    def __init__(self):
        self.llm = AliLLM(model_name="", temperature=0.3)
        self.context_limit = 20

class RAGConfig:
    def __init__(self):
        self.vector_db_type = ""
        self.embedding_dim = 1536
        self.distance_metric = "Cosine"
        self.use_local = True
        self.vector_local_path = ""
        self.doc_local_path = ""
        self.parsed_content_dir = ""
        self.collection_name = "medical_assistance_rag"
        self.chunk_size = 512
        self.chunk_overlap = 50
        self.embedding_model = AliEmbedding(model_name="")
        self.llm = AliLLM(model_name="", temperature=0.3)
        self.summarizer_model = AliLLM(model_name="", temperature=0.5)
        self.chunker_model = AliLLM(model_name="", temperature=0.0)
        self.response_generator_model = AliLLM(model_name="", temperature=0.3)
        self.top_k = 5
        self.vector_search_type = 'similarity'
        self.reranker_model = ""
        self.reranker_top_k = 3
        self.max_context_length = 8192
        self.include_sources = True
        self.min_retrieval_confidence = 0.40
        self.context_limit = 20

class MedicalCVConfig:
    def __init__(self):
        self.brain_tumor_model_path = ""
        self.chest_xray_model_path = ""
        self.skin_lesion_model_path = ""
        self.skin_lesion_segmentation_output_path = ""
        self.llm = AliLLM(model_name="", temperature=0.1)

class SpeechConfig:
    def __init__(self):
        self.eleven_labs_api_key = os.getenv("ELEVEN_LABS_API_KEY")
        self.eleven_labs_voice_id = "21m00Tcm4TlvDq8ikWAM"

class ValidationConfig:
    def __init__(self):
        self.require_validation = {
            "CONVERSATION_AGENT": False,
            "RAG_AGENT": False,
            "WEB_SEARCH_AGENT": False,
            "BRAIN_TUMOR_AGENT": True,
            "CHEST_XRAY_AGENT": True,
            "SKIN_LESION_AGENT": True
        }
        self.validation_timeout = 300
        self.default_action = "reject"

class APIConfig:
    def __init__(self):
        self.host = "0.0.0.0"
        self.port = 8000
        self.debug = True
        self.rate_limit = 10
        self.max_image_upload_size = 5

class UIConfig:
    def __init__(self):
        self.theme = "light"
        self.enable_speech = True
        self.enable_image_upload = True

class Config:
    def __init__(self):
        self.agent_decision = AgentDecisionConfig()
        self.conversation = ConversationConfig()
        self.rag = RAGConfig()
        self.medical_cv = MedicalCVConfig()
        self.web_search = WebSearchConfig()
        self.api = APIConfig()
        self.speech = SpeechConfig()
        self.validation = ValidationConfig()
        self.ui = UIConfig()
        self.eleven_labs_api_key = os.getenv("ELEVEN_LABS_API_KEY")
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        self.max_conversation_history = 20
