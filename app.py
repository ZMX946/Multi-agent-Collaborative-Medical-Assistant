import os
import uuid
import tempfile
from typing import Dict, Union, Optional, List
import glob
import threading
import time
from io import BytesIO

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Request, Response, Cookie
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

import uvicorn
import requests
from werkzeug.utils import secure_filename
from pydub import AudioSegment
from elevenlabs.client import ElevenLabs

from config import Config
from agents.agent_decision import process_query

# Load configuration
config = Config()

# 初始化 FastAPI app
app = FastAPI(title="Multi-Agent Medical Chatbot", version="2.0")

# 设置目录
UPLOAD_FOLDER = "uploads/backend"                     #定义后端上传文件的目录路径。
FRONTEND_UPLOAD_FOLDER = "uploads/frontend"           #定义前端上传文件保存目录。
SKIN_LESION_OUTPUT = "uploads/skin_lesion_output"     #定义皮肤病灶分析的输出图片保存路径。
SPEECH_DIR = "uploads/speech"                         #定义语音合成的输出目录。

# 如果目录不存在，创建目录
for directory in [UPLOAD_FOLDER, FRONTEND_UPLOAD_FOLDER, SKIN_LESION_OUTPUT, SPEECH_DIR]:
    os.makedirs(directory, exist_ok=True)

# 挂载静态文件目录
app.mount("/data", StaticFiles(directory="data"), name="data")              #将本地 data/ 目录映射到 /data 路径，使其可以被浏览器访问。
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")     #将 uploads/ 目录挂载到 /uploads，用于返回上传的文件（如图像、音频等）

# 设置模板
templates = Jinja2Templates(directory="templates")               #创建一个 Jinja2 模板环境，用于渲染 HTML 页面。模板文件放在 templates/ 目录下。

# 初始化ElevenLabs客户端
client = ElevenLabs(
    api_key=config.speech.eleven_labs_api_key,
)                                                         #创建 ElevenLabs TTS（语音合成）客户端，用配置文件中的 API Key 初始化。

# 定义允许的文件扩展名
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


#判断文件名是否包含. 取最后一个 . 后的扩展名转小写后判断是否在允许列表中
def allowed_file(filename):
    """检查文件是否有允许的扩展名"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#定义一个后台任务，用来循环清理旧音频
def cleanup_old_audio():
    """每5分钟删除uploads/speech文件夹中所有的.mp3文件。"""
    while True:
        try:
            files = glob.glob(f"{SPEECH_DIR}/*.mp3")      #找到语音目录中所有 .mp3 文件
            for file in files:
                os.remove(file)
            print("Cleaned up old speech files.")
        except Exception as e:
            print(f"Error during cleanup: {e}")
        time.sleep(300)  # Runs every 5 minutes

# 启动后台清理线程
#创建一个后台线程来执行 cleanup_old_audio()。 daemon=True 表示这个线程会随着主程序退出而自动终止。
cleanup_thread = threading.Thread(target=cleanup_old_audio, daemon=True)
cleanup_thread.start()

#定义一个数据模型 QueryRequest，用来验证 /chat 接口接收的 JSON 数据。
class QueryRequest(BaseModel):
    query: str
    conversation_history: List = []

#定义另一个数据模型 SpeechRequest，用于语音合成接口（TTS）
class SpeechRequest(BaseModel):
    text: str
    voice_id: str = "EXAMPLE_VOICE_ID"  # Default voice ID


#当前端访问网站根路径 / 时，返回 HTML 页面
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """提供主HTML页面"""
    return templates.TemplateResponse("index.html", {"request": request})


#定义一个 GET 端点 /health，用于检查服务是否正常。
@app.get("/health")
def health_check():
    """用于Docker健康检查的健康检查端点"""
    return {"status": "healthy"}


#定义一个 POST 接口 /chat，用于处理聊天请求。
@app.post("/chat")
def chat(
    request: QueryRequest,              #用户输入的数据（被 QueryRequest 校验）
    response: Response,                  #用于设置 cookie 的 Response 对象
    session_id: Optional[str] = Cookie(None)     #从 cookie 里取 session_id（如果没有则为 None）
):
    """通过多代理系统处理用户文本查询。"""
    # 如果cookie不存在，则生成会话ID
    if not session_id:
        session_id = str(uuid.uuid4())

    try:
        response_data = process_query(request.query)            #通过代理系统处理用户查询
        response_text = response_data['messages'][-1].content       #取 AI 最后一句回复（messages 列表中最后一条）。

        # 设置会话 cookie
        response.set_cookie(key="session_id", value=session_id)        #在返回响应时设置 cookie session_id，用于保持对话连续性。

        # 检查agent是否为皮肤病变分割，并找到图像路径
        result = {
            "status": "success",
            "response": response_text,
            "agent": response_data["agent_name"]
        }                                                          #将响应数据封装成 JSON 格式返回给前端。

        # 如果是皮肤病变分割代理，检查输出图像
        if response_data["agent_name"] == "SKIN_LESION_AGENT, HUMAN_VALIDATION":      #如果 AI 选择了皮肤病变分割代理（影像处理 agent），进入下面逻辑
            segmentation_path = os.path.join(SKIN_LESION_OUTPUT, "segmentation_plot.png")              #构造分割图像输出的路径。
            if os.path.exists(segmentation_path):                                                       #检查分割图像输出路径是否存在
                result["result_image"] = f"/uploads/skin_lesion_output/segmentation_plot.png"
            else:
                print("Skin Lesion Output path does not exist.")

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

#声明一个异步函数，用于处理上传请求。
@app.post("/upload")
async def upload_image(
    response: Response,                    #接收 FastAPI 的 Response 对象，用来设置 cookie。
    image: UploadFile = File(...),         #声明一个必填字段 image，类型是上传的文件（例如 PNG / JPG）
    text: str = Form(""),                  #接收可选表单字段 text，默认空字符串，用于“图像 + 文本”混合输入。
    session_id: Optional[str] = Cookie(None)   #从 cookie 里取 session_id（如果没有则为 None）
):
    """处理医学图像上传与可选的文本输入。"""
    # Validate file type
    if not allowed_file(image.filename):                  #调用 allowed_file() 函数，检查上传文件扩展名是否合法。
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "agent": "System",
                "response": "Unsupported file type. Allowed formats: PNG, JPG, JPEG"
            }
        )

    # 保存前检查文件大小
    file_content = await image.read()                      #异步读取上传文件的字节内容。
    if len(file_content) > config.api.max_image_upload_size * 1024 * 1024:  # Convert MB to bytes
        return JSONResponse(
            status_code=413,
            content={
                "status": "error",
                "agent": "System",
                "response": f"File too large. Maximum size allowed: {config.api.max_image_upload_size}MB"
            }
        )

    # 如果cookie不存在，则生成会话ID
    if not session_id:
        session_id = str(uuid.uuid4())

    # 安全保存文件
    filename = secure_filename(f"{uuid.uuid4()}_{image.filename}")   #使用 secure_filename 防止用户上传恶意文件名，并使用 UUID 避免重名。
    file_path = os.path.join(UPLOAD_FOLDER, filename)             #构造保存路径。
    with open(file_path, "wb") as f:
        f.write(file_content)

    try:
        query = {"text": text, "image": file_path}       #构造一个字典，将文本与图像传给 AI 系统。
        response_data = process_query(query)              #通过代理系统处理用户查询。
        response_text = response_data['messages'][-1].content      #$取 AI 最后一句回复（messages 列表中最后一条）。

        # 设置会话cookie
        response.set_cookie(key="session_id", value=session_id)      #在返回响应时设置 cookie session_id，用于保持对话连续性。

        # 检查agent是否为皮肤病变分割，并找到图像路径
        result = {
            "status": "success",
            "response": response_text,
            "agent": response_data["agent_name"]
        }

        # 如果是皮肤病变分割代理，检查输出图像
        if response_data["agent_name"] == "SKIN_LESION_AGENT, HUMAN_VALIDATION":        #检查当前处理的代理是否是皮肤病变处理。
            segmentation_path = os.path.join(SKIN_LESION_OUTPUT, "segmentation_plot.png")   #构造分割图像输出路径。
            if os.path.exists(segmentation_path):              #检查分割图像输出路径是否存在
                result["result_image"] = f"/uploads/skin_lesion_output/segmentation_plot.png"
            else:
                print("Skin Lesion Output path does not exist.")

        # 发送后删除临时文件
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Failed to remove temporary file: {str(e)}")

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


#定义一个 POST 接口 /validate，用于处理人工审核（Human Validation）的结果。
@app.post("/validate")
def validate_medical_output(
    response: Response,
    validation_result: str = Form(...),
    comments: Optional[str] = Form(None),
    session_id: Optional[str] = Cookie(None)
):
    """处理医疗人工智能输出的人工验证。"""
    # 如果cookie不存在，则生成会话ID
    if not session_id:
        session_id = str(uuid.uuid4())

    try:
        # Set session cookie
        response.set_cookie(key="session_id", value=session_id)

        # 使用验证输入重新运行代理决策系统
        validation_query = f"Validation result: {validation_result}"
        if comments:
            validation_query += f" Comments: {comments}"

        response_data = process_query(validation_query)

        if validation_result.lower() == 'yes':
            return {
                "status": "validated",
                "message": "**Output confirmed by human validator:**",
                "response": response_data['messages'][-1].content
            }
        else:
            return {
                "status": "rejected",
                "comments": comments,
                "message": "**Output requires further review:**",
                "response": response_data['messages'][-1].content
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



#定义一个 POST 接口 /transcribe，用于上传音频并进行语音转文字（Speech-to-Text）。
@app.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    """端点转录语音使用ElevenLabs API"""
    if not audio.filename:
        return JSONResponse(
            status_code=400,
            content={"error": "No audio file selected"}
        )

    try:
        # 暂时保存音频文件
        os.makedirs(SPEECH_DIR, exist_ok=True)
        temp_audio = f"./{SPEECH_DIR}/speech_{uuid.uuid4()}.webm"

        # 读取并保存文件
        audio_content = await audio.read()   #异步读取上传的音频文件内容（字节）
        with open(temp_audio, "wb") as f:
            f.write(audio_content)

        # 调试：打印文件大小来检查它是否为空
        file_size = os.path.getsize(temp_audio)
        print(f"Received audio file size: {file_size} bytes")

        if file_size == 0:
            return JSONResponse(
                status_code=400,
                content={"error": "Received empty audio file"}
            )

        # Convert to MP3
        mp3_path = f"./{SPEECH_DIR}/speech_{uuid.uuid4()}.mp3"

        try:
            # 使用pydub进行格式检测
            audio = AudioSegment.from_file(temp_audio)    #使用 pydub 自动识别输入文件格式（webm）并读取音频。
            audio.export(mp3_path, format="mp3")

            # Debug: Print MP3 file size
            mp3_size = os.path.getsize(mp3_path)
            print(f"Converted MP3 file size: {mp3_size} bytes")

            with open(mp3_path, "rb") as mp3_file:
                audio_data = mp3_file.read()
            print(f"Converted audio file into byte array successfully!")

            transcription = client.speech_to_text.convert(
                file=audio_data,
                model_id="scribe_v1",
                tag_audio_events=True,
                language_code="eng",
                diarize=True,
            )

            # 清理临时文件
            try:
                os.remove(temp_audio)
                os.remove(mp3_path)
                print(f"Deleted temp files: {temp_audio}, {mp3_path}")
            except Exception as e:
                print(f"Could not delete file: {e}")

            if transcription.text:
                return {"transcript": transcription.text}
            else:
                return JSONResponse(
                    status_code=500,
                    content={"error": f"API error: {transcription}", "details": transcription.text}
                )

        except Exception as e:
            print(f"Error processing audio: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"error": f"Error processing audio: {str(e)}"}
            )

    except Exception as e:
        print(f"Transcription error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


#创建一个 POST 接口 /generate-speech，用于生成语音（Text-to-Speech）。
@app.post("/generate-speech")
async def generate_speech(request: SpeechRequest):
    """端点使用ElevenLabs API生成语音"""
    try:
        text = request.text
        selected_voice_id = request.voice_id          #从请求体中取出 voice_id，用于选择 ElevenLabs 的语音角色。

        if not text:
            return JSONResponse(
                status_code=400,
                content={"error": "Text is required"}
            )

        # 定义对ElevenLabs的API请求
        elevenlabs_url = f"https://api.elevenlabs.io/v1/text-to-speech/{selected_voice_id}/stream"
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": config.speech.eleven_labs_api_key
        }
        payload = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5
            }
        }

        # 发送请求到ElevenLabs API
        response = requests.post(elevenlabs_url, headers=headers, json=payload)

        if response.status_code != 200:
            return JSONResponse(
                status_code=500,
                content={"error": f"Failed to generate speech, status: {response.status_code}", "details": response.text}
            )

        # 暂时保存音频文件
        os.makedirs(SPEECH_DIR, exist_ok=True)
        temp_audio_path = f"./{SPEECH_DIR}/{uuid.uuid4()}.mp3"
        with open(temp_audio_path, "wb") as f:
            f.write(response.content)

        # 返回生成的音频文件
        return FileResponse(
            path=temp_audio_path,
            media_type="audio/mpeg",
            filename="generated_speech.mp3"
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

# 为过大的请求实体添加异常处理程序
@app.exception_handler(413)
async def request_entity_too_large(request, exc):
    return JSONResponse(
        status_code=413,
        content={
            "status": "error",
            "agent": "System",
            "response": f"File too large. Maximum size allowed: {config.api.max_image_upload_size}MB"
        }
    )

if __name__ == "__main__":
    uvicorn.run(app, host=config.api.host, port=config.api.port)