import sys
import json
import logging
from pathlib import Path

import warnings
warnings.filterwarnings('ignore')     #允许过滤、忽略一些运行时警告。

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# 如果需要，将项目根添加到path
sys.path.append(str(Path(__file__).parent.parent))   #__file__ → 当前文件的路径    .parent.parent → 上两级目录


# Import your components
from agents.rag_agent import MedicalRAG
from config import Config

import argparse

# 初始化 parser
parser = argparse.ArgumentParser(description="Process some command-line arguments.")   #创建一个解析器对象，用于接收命令行参数

# Add arguments
parser.add_argument("--file", type=str, required=False, help="Enter file path to ingest")    #让用户传入一个文件路径。
parser.add_argument("--dir", type=str, required=False, help="Enter directory path of files to ingest")   #用于传入一个目录路径，用来批量处理目录内所有文件。

# Parse arguments
args = parser.parse_args()

# Load configuration
config = Config()

rag = MedicalRAG(config)

# document ingestion
#用于执行文件或文件夹的摄取流程
def data_ingestion():

    if args.file:      # 如果用户传入了文件路径，则执行文件处理和摄取流程。
        # Define path to file
        file_path = args.file
        # Process and ingest the file
        result = rag.ingest_file(file_path)     #把文件处理、切片、嵌入，并写入向量数据库。
    elif args.dir:                   # 如果用户传入了目录路径，则执行目录处理和摄取流程。
        # Define path to dir
        dir_path = args.dir
        # Process and ingest the files
        result = rag.ingest_directory(dir_path)           #把目录内所有文件处理、切片、嵌入，并写入向量数据库。

    print("Ingestion result:", json.dumps(result, indent=2))   #将处理结果格式化打印出来，indent=2 表示 JSON 结构缩进 2 格，便于阅读。

    return result["success"]

# Run tests
if __name__ == "__main__":
   
    print("\nIngesting document(s)...")

    ingestion_success = data_ingestion()
    
    if ingestion_success:
        print("\nSuccessfully ingested the documents.")