import os
import time
import logging
from typing import List, Optional, Dict, Any

from .doc_parser import MedicalDocParser
from .content_processor import ContentProcessor
from .vectorstore_qdrant import VectorStore
from .reranker import Reranker
from .query_expander import QueryExpander
from .response_generator import ResponseGenerator

class MedicalRAG:
    """
    医疗检索增强生成系统，整合所有组件.
    """
    def __init__(self, config):
        """
        初始化RAG代理。

        参数：
            config：包含RAG设置的配置对象
        """
        # Set up logging
        self.logger = logging.getLogger(f"{self.__module__}")
        self.logger.info("Initializing Medical RAG system")
        self.config = config
        self.doc_parser = MedicalDocParser()         # Document parser
        self.content_processor = ContentProcessor(config)        # Content processor
        self.vector_store = VectorStore(config)              # Vector store
        self.reranker = Reranker(config)                 # Reranker
        self.query_expander = QueryExpander(config)
        self.response_generator = ResponseGenerator(config)
        self.parsed_content_dir = self.config.rag.parsed_content_dir
    
    def ingest_directory(self, directory_path: str) -> Dict[str, Any]:
        """
        将目录中的所有文件摄取到RAG系统中。

        参数:
            directory_path：包含要摄取文件的目录的路径

        返回:
            字典与摄入结果
        """
        start_time = time.time()
        self.logger.info(f"从目录中摄取文件: {directory_path}")
        
        try:
            # Check if directory exists
            if not os.path.isdir(directory_path):
                raise ValueError(f"没有找到文件: {directory_path}")
            
            # 获取目录中的所有文件
            files = [os.path.join(directory_path + '/', f) for f in os.listdir(directory_path) 
                     if os.path.isfile(os.path.join(directory_path, f))]
            
            if not files:
                self.logger.warning(f"目录中没有找到文件: {directory_path}")
                return {
                    "success": True,
                    "documents_ingested": 0,
                    "chunks_processed": 0,
                    "processing_time": time.time() - start_time
                }
            
            # Track statistics
            total_chunks_processed = 0
            successful_ingestions = 0
            failed_ingestions = 0
            failed_files = []
            
            # Process each file
            for file_path in files:
                self.logger.info(f"处理文件 {successful_ingestions + failed_ingestions + 1}/{len(files)}: {file_path}")
                
                try:
                    result = self.ingest_file(file_path)
                    if result["success"]:
                        successful_ingestions += 1
                        total_chunks_processed += result.get("chunks_processed", 0)
                    else:
                        failed_ingestions += 1
                        failed_files.append({"file": file_path, "error": result.get("error", "Unknown error")})
                except Exception as e:
                    self.logger.error(f"错误处理文件 {file_path}: {e}")
                    failed_ingestions += 1
                    failed_files.append({"file": file_path, "error": str(e)})
            
            return {
                "success": True,
                "documents_ingested": successful_ingestions,
                "failed_documents": failed_ingestions,
                "failed_files": failed_files,
                "chunks_processed": total_chunks_processed,
                "processing_time": time.time() - start_time
            }
            
        except Exception as e:
            self.logger.error(f"Error ingesting directory: {e}")
            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    def ingest_file(self, document_path: str) -> Dict[str, Any]:
        """
        将单个文件摄取到RAG系统中。

        参数:
            document_path：要摄取的文件的路径

        返回:
            字典与摄入结果
        """
        start_time = time.time()
        self.logger.info(f"Ingesting file: {document_path}")

        try:
            # Step 1: 解析文档
            self.logger.info("1. Parsing document and extracting images...")
            parsed_document, images = self.doc_parser.parse_document(document_path, self.parsed_content_dir)
            self.logger.info(f"   Parsed document and extracted {len(images)} images")

            # Step 2: 生成图像总结
            self.logger.info("2. Summarizing images...")
            image_summaries = self.content_processor.summarize_images(images)
            self.logger.info(f"   Generated {len(image_summaries)} image summaries")

            # Step 3: 用图像摘要格式化文档
            self.logger.info("3. Formatting document with image summaries...")
            formatted_document = self.content_processor.format_document_with_images(parsed_document, image_summaries)

            # Step 4: 将文档块分成语义部分
            self.logger.info("4. Chunking document into semantic sections...")
            document_chunks = self.content_processor.chunk_document(formatted_document)
            self.logger.info(f"   Document split into {len(document_chunks)} chunks")

            # Step 5: 创建向量存储和文档存储
            self.logger.info("5. Creating vector store knowledge base...")
            self.vector_store.create_vectorstore(
                document_chunks=document_chunks, 
                document_path=document_path
                )
            
            return {
                "success": True,
                "documents_ingested": 1,
                "chunks_processed": len(document_chunks),
                "processing_time": time.time() - start_time
            }
        
        except Exception as e:
            self.logger.error(f"Error ingesting file: {e}")
            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time
            }
        
    def process_query(self, query: str, chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        用RAG系统处理查询。

        参数:
            query：查询字符串
            chat_history：上下文的可选聊天记录

        返回:
            响应字典
        """
        start_time = time.time()
        self.logger.info(f"RAG Agent processing query: {query}")
        
        # 处理查询并返回结果，传递chat_history
        try:
            # Step 1: 扩展查询  用相关的医学术语展开原始查询
            self.logger.info(f"1. Expanding query: '{query}'")
            expansion_result = self.query_expander.expand_query(query)            #用相关的医学术语展开原始查询
            expanded_query = expansion_result["expanded_query"]
            self.logger.info(f"   Original: '{query}'")
            self.logger.info(f"   Expanded: '{expanded_query}'")
            query = expanded_query

            # Step 2: 向量数据库检索
            self.logger.info(f"2. Retrieving relevant documents for the query: '{query}'")
            vectorstore, docstore = self.vector_store.load_vectorstore()               # 加载向量存储
            retrieved_documents = self.vector_store.retrieve_relevant_chunks(
                query=query,
                vectorstore=vectorstore,
                docstore=docstore,
                )

            self.logger.info(f"   Retrieved {len(retrieved_documents)} relevant document chunks")

            # Step 3: 如果我们有一个重新排序器并且有足够的文档，则重新排序检索到的文档
            self.logger.info(f"3. Reranking the retrieved documents")
            if self.reranker and len(retrieved_documents) > 1:
                reranked_documents, reranked_top_k_picture_paths = self.reranker.rerank(query, retrieved_documents, self.parsed_content_dir)
                self.logger.info(f"   Reranked retrieved documents and chose top {len(reranked_documents)}")
                self.logger.info(f"   Found {len(reranked_top_k_picture_paths)} referenced images")
            else:
                self.logger.info(f"   Could not rerank the retrieved documents, falling back to original scores")
                reranked_documents = retrieved_documents
                reranked_top_k_picture_paths = []

            # Step 4: 生成响应 根据检索到的上下文和用户查询生成响应
            self.logger.info("4. Generating response...")
            response = self.response_generator.generate_response(
                query=query,
                retrieved_docs=reranked_documents,
                picture_paths=reranked_top_k_picture_paths,
                chat_history=chat_history
                )
            
            # Add timing information
            processing_time = time.time() - start_time
            response["processing_time"] = processing_time
            
            return response
        
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            # Return error response
            return {
                "response": f"I encountered an error while processing your query: {str(e)}",
                "sources": [],
                "confidence": 0.0,
                "processing_time": time.time() - start_time
            }
