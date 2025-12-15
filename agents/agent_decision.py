"""
多智能体医疗聊天机器人的智能体决策系统。
该模块使用LangGraph处理不同代理的编排。
它根据内容和上下文动态地将用户查询路由到适当的代理。
"""

import json
from typing import Dict, List, Optional, Any, Literal, TypedDict, Union, Annotated
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import MessagesState, StateGraph, END
import os, getpass
from dotenv import load_dotenv
from agents.rag_agent import MedicalRAG
from agents.web_search_processor_agent import WebSearchProcessorAgent
from agents.image_analysis_agent import ImageAnalysisAgent
from agents.guardrails.local_guardrails import LocalGuardrails

from langgraph.checkpoint.memory import MemorySaver

import cv2
import numpy as np

from config import Config

load_dotenv()

# Load configuration
config = Config()

# Initialize memory
memory = MemorySaver()

# Specify a thread
thread_config = {"configurable": {"thread_id": "1"}}


# 代理，它决定进一步路由请求以纠正特定于任务的代理
class AgentConfig:
    """这个类用于存储代理决策系统的配置。"""
    
    # Decision model
    DECISION_MODEL = "gpt-4o"  # or whichever model you prefer
    
    # Vision model for image analysis
    VISION_MODEL = "gpt-4o"
    
    # Confidence threshold for responses
    CONFIDENCE_THRESHOLD = 0.85           #决策阈值为 0.85
    
    # System instructions for the decision agent
    DECISION_SYSTEM_PROMPT = """您是一个智能医疗分诊系统，将用户查询路由到
    适当的专门代理。您的工作是分析用户的请求并确定哪个代理
    最适合基于查询内容、图像的存在和对话上下文来处理它。

    Available agents:
    1. CONVERSATION_AGENT -用于一般聊天，问候和非医疗问题。
    2. RAG_AGENT -针对可以从已建立的医学文献中回答的特定医学知识问题。目前摄入的医学知识包括“脑肿瘤概论”、“诊断和检测脑肿瘤的深度学习技术”、“从胸部x光片诊断和检测covid-19的深度学习技术”。
    3. WEB_SEARCH_PROCESSOR_AGENT—关于最近的医疗发展、当前的疫情或时间敏感的医疗信息的问题。
    4. BRAIN_TUMOR_AGENT -用于分析脑MRI图像以检测和分割肿瘤。
    5. CHEST_XRAY_AGENT -用于分析胸部x射线图像以检测异常。
    6. SKIN_LESION_AGENT -用于分析皮肤病变图像，将其分类为良性或恶性。

    根据以下准则做出决定：
    —如果用户没有上传任何图片，总是路由到对话代理。
    —如果用户上传了医学图像，则根据图像类型和用户的查询来决定哪个医学视觉代理是合适的。如果上传图像时没有查询，则始终根据图像类型路由到正确的医疗视觉代理。
    -如果用户询问最近的医疗发展或当前的健康状况，请使用web搜索处理器代理。
    -如果用户询问特定的医学知识问题，请使用RAG代理。
    —对于一般会话、问候语或非医疗问题，请使用会话代理。但如果上传图像，总是先去找医疗视觉代理。
    您必须以JSON格式提供答案，结构如下：
    {{
    "agent": "AGENT_NAME",
    "reasoning": "Your step-by-step reasoning for selecting this agent",
    "confidence": 0.95  // Value between 0.0 and 1.0 indicating your confidence in this decision
    }}
    """

    image_analyzer = ImageAnalysisAgent(config=config)         # 创建一个ImageAnalysisAgent实例 初始化一个图像分析代理


#用于在多个 agent 工作流之间共享和维护状态。就像“全局状态容器”。
class AgentState(MessagesState):
    """跨工作流维护的状态。"""
    # messages: List[BaseMessage]  # Conversation history
    agent_name: Optional[str]  # 存储当前正在处理用户输入的代理名称
    current_input: Optional[Union[str, Dict]]  # 本次需要处理的输入内容
    has_image: bool  # 当前输入是否包含图像
    image_type: Optional[str]  # 医学图像类型（如果存在）
    output: Optional[str]  # 表示最终要返回给用户的回答内容
    needs_human_validation: bool  # 是否需要人工验证
    retrieval_confidence: float  # 检索置信度（对于RAG代理）
    bypass_routing: bool  # 绕过护栏代理路由的标志
    """
    当 True 时：
        跳过所有决策逻辑
        让系统进入安全 guardrails 模式（例如人工审核）
    常用于：
        输出被拒绝后重新执行验证
        客户端发送验证结果（validate）时
    """
    insufficient_info: bool  # RAG响应信息不足标志
    """
        当 RAG 发现资料库中没有足够信息时：
            此值设为 True
            多代理系统将自动切换到 Web Search Agent
    """


class AgentDecision(TypedDict):
    """决策代理的输出结构。"""
    agent: str
    reasoning: str
    confidence: float


def create_agent_graph():
    """创建并配置用于代理编排的LangGraph。"""

    #使用与其他地方相同的LLM初始化护栏
    guardrails = LocalGuardrails(config.rag.llm)     #Guardrails 用来检查模型输出是否安全，例如避免有害内容、错误医疗建议等

    # LLM
    decision_model = config.agent_decision.llm       #从配置中加载用于“代理选择（routing）”的 LLM
    
    # 初始化输出解析器
    json_parser = JsonOutputParser(pydantic_object=AgentDecision)       #创建一个 JSON 解析器，它会把 LLM 输出的 JSON 字符串解析成 AgentDecision TypedDict 格式。
    
    #创建决策提示  创建一个聊天提示模板
    decision_prompt = ChatPromptTemplate.from_messages([
        ("system", AgentConfig.DECISION_SYSTEM_PROMPT),
        ("human", "{input}")
    ])
    
    # Create the decision chain
    decision_chain = decision_prompt | decision_model | json_parser
    
    # Define graph state transformations
    def analyze_input(state: AgentState) -> AgentState:
        """分析输入以检测图像并确定输入类型。"""
        current_input = state["current_input"]         #从 state 中取出当前用户输入
        has_image = False
        image_type = None
        
        # Get the text from the input
        input_text = ""
        if isinstance(current_input, str):       #如果输入是一个字符串，则直接使用它
            input_text = current_input            #获取输入的文本
        elif isinstance(current_input, dict):      #如果输入是一个字典包含图像或文本，则从字典中获取文本
            input_text = current_input.get("text", "")    #获取text字段内容
        
        # 如果有文本，要先通过安全护栏检查。
        if input_text:
            is_allowed, message = guardrails.check_input(input_text)      #用 Guardrails 分析文本内容是否安全
            if not is_allowed:        #如果文本不安全则终止
                # If input is blocked, return early with guardrail message
                print(f"Selected agent: INPUT GUARDRAILS, Message: ", message)
                return {
                    **state,
                    "messages": message,
                    "agent_name": "INPUT_GUARDRAILS",
                    "has_image": False,
                    "image_type": None,
                    "bypass_routing": True  # flag to end flow
                }
        
        # Original image processing code
        #如果输入是字典，并且包含 "image" 字段，则说明用户上传了医学图像。
        if isinstance(current_input, dict) and "image" in current_input:
            has_image = True
            image_path = current_input.get("image", None)      #获取上传的医学图像路径
            image_type_response = AgentConfig.image_analyzer.analyze_image(image_path)      #调用图像分析代理
            image_type = image_type_response['image_type']         #获取图像类型
            print("ANALYZED IMAGE TYPE: ", image_type)
        
        return {
            **state,
            "has_image": has_image,
            "image_type": image_type,
            "bypass_routing": False  # Explicitly set to False for normal flow
        }


    #检查当前流程是否需要跳过代理路由（通常是因为触发了 Guardrails）
    def check_if_bypassing(state: AgentState) -> str:
        """检查一下我们是否应该绕过正常路线，因为有guardrails."""
        if state.get("bypass_routing", False):          #当 Guardrails 检测到危险输入时，bypass_routing 会被设置为 True 此处用来判断是否应该跳过正常流程
            return "apply_guardrails"       #跳到apply_guardrails节点
        return "route_to_agent"         #否则就正常处理跳到route_to_agent节点
    
    def route_to_agent(state: AgentState) -> Dict:
        """决定哪个代理应该处理查询。"""
        messages = state["messages"]                        #从 state 中取出对话历史
        current_input = state["current_input"]              #从 state 中取出当前用户输入
        has_image = state["has_image"]                      #从 state 中取出图像信息
        image_type = state["image_type"]                    #从 state 中取出图像类型
        
        # 为决策模型准备输入
        input_text = ""
        if isinstance(current_input, str):                #如果输入是一个字符串，则直接使用它
            input_text = current_input                     #获取输入的文本
        elif isinstance(current_input, dict):                 #如果输入是一个字典包含图像或文本，则从字典中获取文本
            input_text = current_input.get("text", "")         #获取text字段内容

        # 根据最近的对话历史（最近3条消息）创建上下文
        recent_context = ""
        for msg in messages[-6:]:  # 获取最近3次交换（6条消息）#未从配置中提供控制
            if isinstance(msg, HumanMessage):                  #如果是用户消息
                recent_context += f"User: {msg.content}\n"        #添加用户消息
            elif isinstance(msg, AIMessage):                   #如果是助手消息
                recent_context += f"Assistant: {msg.content}\n"   #添加助手消息
        
        # 将所有内容结合起来作为决策输入
        decision_input = f"""
        User query: {input_text}

        Recent conversation context:
        {recent_context}

        Has image: {has_image}
        Image type: {image_type if has_image else 'None'}

        Based on this information, which agent should handle this query?
        """
        
        # 做出决定
        decision = decision_chain.invoke({"input": decision_input})

        # 决定代理
        print(f"Decision: {decision['agent']}")
        
        # 用决策更新状态
        updated_state = {
            **state,
            "agent_name": decision["agent"],
        }
        
        # 基于agent名称和置信度的路由
        if decision["confidence"] < AgentConfig.CONFIDENCE_THRESHOLD:            #如果置信度低于阈值，则需要验证
            return {"agent_state": updated_state, "next": "needs_validation"}        #跳到needs_validation节点
        
        return {"agent_state": updated_state, "next": decision["agent"]}            #跳到指定代理节点

    # 定义代理执行函数（这些将在各自的模块中实现）
    def run_conversation_agent(state: AgentState) -> AgentState:
        """处理一般的谈话"""

        print(f"Selected agent: CONVERSATION_AGENT")

        messages = state["messages"]                   #从 state 中取出对话历史
        current_input = state["current_input"]            #从 state 中取出当前用户输入
        
        # 为决策模型准备输入
        input_text = ""
        if isinstance(current_input, str):          #如果输入是一个字符串，则直接使用它
            input_text = current_input              #获取输入的文本
        elif isinstance(current_input, dict):          #如果输入是一个字典包含图像或文本，则从字典中获取文本
            input_text = current_input.get("text", "")     #获取text字段内容
        
        # 从最近的谈话历史中创建上下文
        recent_context = ""
        for msg in messages:#[-20:]:  # 获取最近10次交换（20条消息）#目前正在考虑完整的历史记录-从配置中限制控制
            if isinstance(msg, HumanMessage):            #如果是用户消息
                # print("######### DEBUG 1:", msg)
                recent_context += f"User: {msg.content}\n"        #添加用户消息
            elif isinstance(msg, AIMessage):                   #如果是助手消息
                # print("######### DEBUG 2:", msg)
                recent_context += f"Assistant: {msg.content}\n"          #添加助手消息
        
        #将所有内容结合起来作为决策输入
        conversation_prompt = f"""用户查询: {input_text}

        最近的对话内容： {recent_context}

        你是一个人工智能医疗对话助手。您的目标是促进与用户的顺畅和信息丰富的对话，处理随意和医疗相关的查询。你必须在保证医疗准确性和清晰度的同时自然地做出反应。
        ### 角色与能力
        -参与**一般谈话**，同时保持专业精神。
        -用已验证的知识回答**医学问题**。
        -路由**复杂的查询**到RAG（检索增强生成）或网络搜索，如果需要的话。
        -处理**后续问题**，同时保持对话上下文的跟踪。
        -将**医学图像**重定向到适当的AI分析代理。

        ###回应指南：
        1.**一般对话:**
        -如果用户进行随意的交谈（例如，问候，闲聊），以友好，吸引人的方式回应。
        -除非需要详细的回答，否则要保持回答的简洁和吸引人。

        2.**医学问题:**
        -如果你对回答有**高的信心**，提供一个医学上准确的回答。
        -确保回答清晰、简洁、真实。

        3.**跟进及澄清：**
        -保存对话记录，以便更好地回应。
        —如果有不清楚的问题，在回答之前问一些后续问题。

        4.**处理医学图像分析：**
        **不要**尝试自己分析图像。
        -如果用户谈到从任何图像中分析或处理或检测或分割或分类任何疾病，请用户上传图像，以便在下一步将其路由到适当的医疗视觉代理。
        -如果上传了图像，它会被路由到医疗计算机视觉代理。阅读历史记录以了解诊断结果，如果用户询问有关诊断的任何问题，则继续对话。
        -处理后，**帮助用户解释结果**。

        5.**不确定性和伦理考虑：**
        -如果不确定，**永远不要假设**医学事实。
        -对于严重的医疗问题，建议咨询**有执照的医疗保健专业人员**。
        -避免提供**医疗诊断**或**处方** -坚持常识。

        ###响应格式：
        -保持专业的语气。
        -在需要的时候使用项目符号或编号列表。
        -如果是从外部来源（RAG/Web Search）提取信息，请注明信息来源（例如，“根据梅奥诊所…”）。
        -如果用户要求诊断，提醒他们**寻求医疗咨询**。
        
        示例用户查询和响应：
        **用户：“嘿，你今天过得怎么样？”
        你：“我在这里，随时准备帮助你！”今天我能为您效劳吗？”
        
        **用户：**“我头痛发烧。我该怎么办呢？”
        你：“我不是医生，但头痛和发烧可能有多种原因，从感染到脱水。如果症状持续，你应该去看专业医生。”
        
        会话式LLM回应："""

        # print("Conversation Prompt:", conversation_prompt)

        response = config.conversation.llm.invoke(conversation_prompt)             #调用 LLM

        # print("Conversation respone:", response)

        # response = AIMessage(content="This would be handled by the conversation agent.")

        return {
            **state,
            "output": response,
            "agent_name": "CONVERSATION_AGENT"
        }

    #运行rag
    def run_rag_agent(state: AgentState) -> AgentState:
        """使用RAG处理医学知识查询。"""
        # Initialize the RAG agent

        print(f"Selected agent: RAG_AGENT")

        rag_agent = MedicalRAG(config)            #创建RAG代理
        
        messages = state["messages"]          #从 state 中取出对话历史
        query = state["current_input"]         #从 state 中取出当前用户输入
        rag_context_limit = config.rag.context_limit            #从配置中获取上下文限制

        recent_context = ""
        for msg in messages[-rag_context_limit:]:# 限制由配置控制
            if isinstance(msg, HumanMessage):              #如果是用户消息
                # print("######### DEBUG 1:", msg)
                recent_context += f"User: {msg.content}\n"            #添加用户消息
            elif isinstance(msg, AIMessage):                  #如果是助手消息
                # print("######### DEBUG 2:", msg)
                recent_context += f"Assistant: {msg.content}\n"         #添加助手消息

        response = rag_agent.process_query(query, chat_history=recent_context)          #运行 RAG 检索并生成答案
        retrieval_confidence = response.get("confidence", 0.0)  # 取出检索置信度，如果没有则默认 0.0

        print(f"Retrieval Confidence: {retrieval_confidence}")
        print(f"Sources: {len(response['sources'])}")

        # 检查响应是否提示信息不足
        insufficient_info = False
        response_content = response["response"]
        
        # 根据类型适当地提取内容
        if isinstance(response_content, dict) and hasattr(response_content, 'content'):         # 检查是否为 AIMessage 或类似对象
            # If it's an AIMessage or similar object with a content attribute
            response_text = response_content.content           # 获取内容
        else:
            # If it's already a string
            response_text = response_content            # 直接使用内容
            
        print(f"Response text type: {type(response_text)}")
        print(f"Response text preview: {response_text[:100]}...")
        
        if isinstance(response_text, str) and (
            "I don't have enough information to answer this question based on the provided context" in response_text or 
            "I don't have enough information" in response_text or 
            "don't have enough information" in response_text.lower() or
            "not enough information" in response_text.lower() or
            "insufficient information" in response_text.lower() or
            "cannot answer" in response_text.lower() or
            "unable to answer" in response_text.lower()
            ):                     # 检查响应是否提示信息不足
            
            print("RAG response indicates insufficient information")
            print(f"Response text that triggered insufficient_info: {response_text[:100]}...")
            insufficient_info = True

        print(f"Insufficient info flag set to: {insufficient_info}")

        # Store RAG output ONLY if confidence is high
        if retrieval_confidence >= config.rag.min_retrieval_confidence:         # 如果置信度满足要求
            # response_output = response["response"]
            response_output = AIMessage(content=response_text)                  # 创建助手消息
        else:
            response_output = AIMessage(content="")                   # 创建空助手消息
        
        return {
            **state,
            "output": response_output,
            "needs_human_validation": False,  # Assuming no validation needed for RAG responses
            "retrieval_confidence": retrieval_confidence,
            "agent_name": "RAG_AGENT",
            "insufficient_info": insufficient_info
        }

    # Web搜索处理器节点
    def run_web_search_processor_agent(state: AgentState) -> AgentState:
        """处理网络搜索结果，用LLM处理它们，并生成精炼的响应。"""

        print(f"Selected agent: WEB_SEARCH_PROCESSOR_AGENT")
        print("[WEB_SEARCH_PROCESSOR_AGENT] Processing Web Search Results...")
        
        messages = state["messages"]             # 从 state 中取出对话历史
        web_search_context_limit = config.web_search.context_limit             # 从配置中获取上下文限制

        recent_context = ""
        for msg in messages[-web_search_context_limit:]: # 限制由配置控制
            if isinstance(msg, HumanMessage):          # 如果是用户消息
                # print("######### DEBUG 1:", msg)
                recent_context += f"User: {msg.content}\n"       #$添加用户消息
            elif isinstance(msg, AIMessage):                # 如果是助手消息
                # print("######### DEBUG 2:", msg)
                recent_context += f"Assistant: {msg.content}\n"        #添加助手消息

        web_search_processor = WebSearchProcessorAgent(config)          #创建 Web 搜索处理器实例

        processed_response = web_search_processor.process_web_search_results(query=state["current_input"], chat_history=recent_context)   #处理当前查询与上下文

        # print("######### DEBUG WEB SEARCH:", processed_response)
        
        if state['agent_name'] != None:                  # 如果之前有代理，则添加当前代理
            involved_agents = f"{state['agent_name']}, WEB_SEARCH_PROCESSOR_AGENT"          # 添加当前代理
        else:
            involved_agents = "WEB_SEARCH_PROCESSOR_AGENT"            # 如果没有，则添加当前代理

        # Overwrite any previous output with the processed Web Search response
        return {
            **state,
            # "output": "This would be handled by the web search agent, finding the latest information.",
            "output": processed_response,
            "agent_name": involved_agents
        }

    # 定义路由逻辑
    def confidence_based_routing(state: AgentState) -> Dict[str, str]:
        """基于RAG置信度评分和回复内容的路由。"""
        # Debug prints
        print(f"Routing check - Retrieval confidence: {state.get('retrieval_confidence', 0.0)}")
        print(f"Routing check - Insufficient info flag: {state.get('insufficient_info', False)}")
        
        # 如果置信度低或响应表明信息不足，则重定向
        if (state.get("retrieval_confidence", 0.0) < config.rag.min_retrieval_confidence or 
            state.get("insufficient_info", False)):              # 如果置信度低或响应表明信息不足
            print("Re-routed to Web Search Agent due to low confidence or insufficient information...")
            return "WEB_SEARCH_PROCESSOR_AGENT"  # Correct format      #返回 Web Search Processor 节点
        return "check_validation"  # 如果信心很高且信息充足，则不需要过渡
    
    def run_brain_tumor_agent(state: AgentState) -> AgentState:
        """处理脑MRI图像分析。"""

        print(f"Selected agent: BRAIN_TUMOR_AGENT")

        response = AIMessage(content="This would be handled by the brain tumor agent, analyzing the MRI image.")   #创建了一个 AIMessage 对象（假设是一个消息封装类，用于在 agent 系统里传递信息）

        return {
            **state,
            "output": response,
            "needs_human_validation": True,  # Medical diagnosis always needs validation
            "agent_name": "BRAIN_TUMOR_AGENT"
        }
    
    def run_chest_xray_agent(state: AgentState) -> AgentState:
        """处理胸部x光图像分析."""

        current_input = state["current_input"]           # 从 state 中取出当前输入
        image_path = current_input.get("image", None)       # 从输入中获取图像路径

        print(f"Selected agent: CHEST_XRAY_AGENT")

        # 将胸部x线分为新冠或正常
        predicted_class = AgentConfig.image_analyzer.classify_chest_xray(image_path)      # 调用图像分析方法

        if predicted_class == "covid19":                 # 如果预测结果为阳性
            response = AIMessage(content="The analysis of the uploaded chest X-ray image indicates a **POSITIVE** result for **COVID-19**.")
        elif predicted_class == "normal":       # 如果预测结果为阴性
            response = AIMessage(content="The analysis of the uploaded chest X-ray image indicates a **NEGATIVE** result for **COVID-19**, i.e., **NORMAL**.")
        else:
            response = AIMessage(content="The uploaded image is not clear enough to make a diagnosis / the image is not a medical image.")

        # response = AIMessage(content="This would be handled by the chest X-ray agent, analyzing the image.")

        return {
            **state,
            "output": response,
            "needs_human_validation": True,  # Medical diagnosis always needs validation
            "agent_name": "CHEST_XRAY_AGENT"
        }
    
    def run_skin_lesion_agent(state: AgentState) -> AgentState:
        """处理皮肤病变图像分析。"""

        current_input = state["current_input"]
        image_path = current_input.get("image", None)

        print(f"Selected agent: SKIN_LESION_AGENT")

        # classify chest x-ray into covid or normal
        predicted_mask = AgentConfig.image_analyzer.segment_skin_lesion(image_path)    # 调用图像分析方法

        if predicted_mask:       # 如果预测结果为真
            response = AIMessage(content="Following is the analyzed **segmented** output of the uploaded skin lesion image:")
        else:        # 如果预测结果为假
            response = AIMessage(content="The uploaded image is not clear enough to make a diagnosis / the image is not a medical image.")

        # response = AIMessage(content="This would be handled by the skin lesion agent, analyzing the skin image.")

        return {
            **state,
            "output": response,
            "needs_human_validation": True,  # Medical diagnosis always needs validation
            "agent_name": "SKIN_LESION_AGENT"
        }
    
    def handle_human_validation(state: AgentState) -> Dict:
        """如果需要，准备人工验证。"""
        if state.get("needs_human_validation", False):           # 如果需要人工验证
            return {"agent_state": state, "next": "human_validation", "agent": "HUMAN_VALIDATION"}
        return {"agent_state": state, "next": END}               # 返回结束
    
    # 处理人工验证
    def perform_human_validation(state: AgentState) -> AgentState:
        """处理人工验证过程。"""
        print(f"Selected agent: HUMAN_VALIDATION")

        # 添加验证请求到存在的输出
        """
        构建人工验证提示信息：
            使用之前 agent 的输出 state['output'].content。
        添加验证说明：
            医疗专业人员：可选择 Yes/No 并给出意见。
            患者：可简单确认 Yes。
        """
        validation_prompt = f"{state['output'].content}\n\n**Human Validation Required:**\n- If you're a healthcare professional: Please validate the output. Select **Yes** or **No**. If No, provide comments.\n- If you're a patient: Simply click Yes to confirm."

        #创建带有验证提示的AI消息
        validation_message = AIMessage(content=validation_prompt)

        return {
            **state,
            "output": validation_message,
            "agent_name": f"{state['agent_name']}, HUMAN_VALIDATION"
        }

    # 通过guardrails检查输出
    def apply_output_guardrails(state: AgentState) -> AgentState:
        """将输出guardrails应用于生成的响应。"""
        output = state["output"]
        current_input = state["current_input"]

        # 检查输出是否有效
        # 如果输出为空或无效，则返回原始状态
        if not output or not isinstance(output, (str, AIMessage)):
            return state

        # 获取输出文本 将 AIMessage 对象转换为文本
        output_text = output if isinstance(output, str) else output.content
        
        # 如果最后一条消息是人工验证消息
        if "Human Validation Required" in output_text:   # 检查输出是否包含人工验证提示
            # 检查当前输入是否是人工验证响应
            validation_input = ""
            if isinstance(current_input, str):      # 检查输入是否是文本
                validation_input = current_input      # 将输入转换为文本
            elif isinstance(current_input, dict):     # 检查输入是否是字典包含文本和图片
                validation_input = current_input.get("text", "")   # 从字段text获取内容
            
            # 如果验证输入存在
            if validation_input.lower().startswith(('yes', 'no')):         # 检查验证输入是否以Yes或No开头
                # 添加验证结果到对话历史
                validation_response = HumanMessage(content=f"Validation Result: {validation_input}")       # 创建一个HumanMessage对象
                
                # If validation is 'No', modify the output
                if validation_input.lower().startswith('no'):         # 如果验证输入是'No'
                    #生成 fallback_message，提示需要进一步审核
                    fallback_message = AIMessage(content="The previous medical analysis requires further review. A healthcare professional has flagged potential inaccuracies.")
                    return {
                        **state,
                        "messages": [validation_response, fallback_message],
                        "output": fallback_message
                    }
                
                return {
                    **state,
                    "messages": validation_response
                }
        
        # Get the original input text
        input_text = ""
        if isinstance(current_input, str):
            input_text = current_input
        elif isinstance(current_input, dict):
            input_text = current_input.get("text", "")
        
        # 应用输出处理
        sanitized_output = guardrails.check_output(output_text, input_text)
        # sanitized_output = output_text
        
        # For non-validation cases, add the sanitized output to messages
        sanitized_message = AIMessage(content=sanitized_output) if isinstance(output, AIMessage) else sanitized_output
        
        return {
            **state,
            "messages": sanitized_message,
            "output": sanitized_message
        }

    
    # Create the workflow graph
    workflow = StateGraph(AgentState)
    
    # Add nodes for each step
    workflow.add_node("analyze_input", analyze_input)
    workflow.add_node("route_to_agent", route_to_agent)
    workflow.add_node("CONVERSATION_AGENT", run_conversation_agent)
    workflow.add_node("RAG_AGENT", run_rag_agent)
    workflow.add_node("WEB_SEARCH_PROCESSOR_AGENT", run_web_search_processor_agent)
    workflow.add_node("BRAIN_TUMOR_AGENT", run_brain_tumor_agent)
    workflow.add_node("CHEST_XRAY_AGENT", run_chest_xray_agent)
    workflow.add_node("SKIN_LESION_AGENT", run_skin_lesion_agent)
    workflow.add_node("check_validation", handle_human_validation)
    workflow.add_node("human_validation", perform_human_validation)
    workflow.add_node("apply_guardrails", apply_output_guardrails)
    
    # Define the edges (workflow connections)
    workflow.set_entry_point("analyze_input")                               #设置工作流的入口点为"analyze_input"
    # workflow.add_edge("analyze_input", "route_to_agent")
    # Add conditional routing for guardrails bypass
    """
    根据 check_if_bypassing(state) 判断是否 直接跳过 agent 执行，走 guardrails。
    True → "apply_guardrails"
    False → "route_to_agent"（进入正常 agent 流程
    """
    workflow.add_conditional_edges(
        "analyze_input",
        check_if_bypassing,
        {
            "apply_guardrails": "apply_guardrails",
            "route_to_agent": "route_to_agent"
        }
    )
    
    # 将决策路由器连接到代理
    #根据 "next" 字段的值，将任务分发到对应 agent
    workflow.add_conditional_edges(
        "route_to_agent",
        lambda x: x["next"],
        {
            "CONVERSATION_AGENT": "CONVERSATION_AGENT",
            "RAG_AGENT": "RAG_AGENT",
            "WEB_SEARCH_PROCESSOR_AGENT": "WEB_SEARCH_PROCESSOR_AGENT",
            "BRAIN_TUMOR_AGENT": "BRAIN_TUMOR_AGENT",
            "CHEST_XRAY_AGENT": "CHEST_XRAY_AGENT",
            "SKIN_LESION_AGENT": "SKIN_LESION_AGENT",
            "needs_validation": "RAG_AGENT"  # 如果置信度低，默认为RAG
        }
    )
    
    # 将代理输出连接到验证检查
    workflow.add_edge("CONVERSATION_AGENT", "check_validation")
    # workflow.add_edge("RAG_AGENT", "check_validation")
    workflow.add_edge("WEB_SEARCH_PROCESSOR_AGENT", "check_validation")
    workflow.add_conditional_edges("RAG_AGENT", confidence_based_routing)
    workflow.add_edge("BRAIN_TUMOR_AGENT", "check_validation")
    workflow.add_edge("CHEST_XRAY_AGENT", "check_validation")
    workflow.add_edge("SKIN_LESION_AGENT", "check_validation")

    workflow.add_edge("human_validation", "apply_guardrails")
    workflow.add_edge("apply_guardrails", END)

    """
    根据 state["next"] 决定是否进入人工验证。
    否则直接走 guardrails。
    """
    workflow.add_conditional_edges(
        "check_validation",
        lambda x: x["next"],
        {
            "human_validation": "human_validation",
            END: "apply_guardrails"  # 通往guardrails的路线，而不是终点
        }
    )
    
    # workflow.add_edge("human_validation", END)
    
    # Compile the graph
    return workflow.compile(checkpointer=memory)


def init_agent_state() -> AgentState:
    """使用默认值初始化代理状态。"""
    """
    messages: 保存对话或消息历史，每个 agent 输出和人工验证结果都可以追加到这里。
    agent_name: 当前执行的 agent 名称。
    current_input: 当前处理的用户输入（文本或图像路径/对象）。
    has_image: 标记输入是否包含图像。
    image_type: 如果有图像，记录类型，例如 "brain_mri", "chest_xray", "skin_lesion"。
    output:   agent 的输出，通常是 AIMessage 或字符串。
    needs_human_validation:   是否需要人工验证（human-in-the-loop 标志）。
    retrieval_confidence:    检索或模型预测置信度，用于路由或人工验证决策。
    bypass_routing:   是否跳过 agent 直接进入 guardrails。
    insufficient_info:   如果输入信息不足，可用于触发 fallback 或提示用户补充信息。
    """
    return {
        "messages": [],
        "agent_name": None,
        "current_input": None,
        "has_image": False,
        "image_type": None,
        "output": None,
        "needs_human_validation": False,
        "retrieval_confidence": 0.0,
        "bypass_routing": False,
        "insufficient_info": False
    }


def process_query(query: Union[str, Dict], conversation_history: List[BaseMessage] = None) -> str:
    """
    通过代理决策系统处理用户查询。

    参数:
        query：用户输入（文本字符串或带有文本和图像的字典）
        conversation_history：以前消息的可选列表，不再需要，因为状态现在保存会话历史记录

    返回:
        来自适当代理的响应
    """
    # Initialize the graph
    graph = create_agent_graph()    # 创建代理图

    # # Save Graph Flowchart
    # image_bytes = graph.get_graph().draw_mermaid_png()
    # decoded = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)
    # cv2.imwrite("./assets/graph.png", decoded)
    # print("Graph flowchart saved in assets.")
    
    # 初始化 state
    state = init_agent_state()          # 初始化代理状态
    # if conversation_history:
    #     state["messages"] = conversation_history
    
    # 添加当前查询
    state["current_input"] = query         # 添加当前查询

    # 处理图片上传的情况
    # 如果query是字典且包含图像，给文本添加说明 "user uploaded an image for diagnosis."。
    if isinstance(query, dict):             #
        query = query.get("text", "") + ", user uploaded an image for diagnosis."
    
    state["messages"] = [HumanMessage(content=query)]            # 添加当前查询

    # result = graph.invoke(state, thread_config)
    result = graph.invoke(state, thread_config)           # 调用代理图
    # print("######### DEBUG 4:", result)
    # state["messages"] = [result["messages"][-1].content]

    # Keep history to reasonable size (ANOTHER OPTION: summarize and store before truncating history)
    #控制会话历史长度
    if len(result["messages"]) > config.max_conversation_history:  # 保持上次配置。max_conversation_history消息
        result["messages"] = result["messages"][-config.max_conversation_history:]

    # 在控制台中可视化会话历史
    for m in result["messages"]:
        m.pretty_print()
    
    # 将响应添加到会话历史记录中
    return result