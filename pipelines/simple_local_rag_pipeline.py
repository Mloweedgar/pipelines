"""
title: Simple Local Files RAG Pipeline
author: Edgar Mlowe
date: 2024-05-30
version: 1.0
license: MIT
description: A simple RAG pipeline that indexes local text files for quick testing.
requirements: llama-index, llama-index-llms-ollama, llama-index-embeddings-ollama
"""

from typing import List, Union, Generator, Iterator, Dict, Any
from pydantic import BaseModel, Field
import os
import logging
import tempfile

class Pipeline:
    class Valves(BaseModel):
        OLLAMA_BASE_URL: str = Field(
            default="http://localhost:11434", 
            description="Base URL for Ollama API"
        )
        EMBEDDING_MODEL: str = Field(
            default="nomic-embed-text", 
            description="Embedding model to use"
        )
        LLM_MODEL: str = Field(
            default="llama3", 
            description="LLM model to use for query processing"
        )
        SAMPLE_TEXT: str = Field(
            default="This is a sample text that will be used for the RAG pipeline. You can add more text here to test different queries.",
            description="Sample text to index for testing"
        )

    def __init__(self):
        self.index = None
        self.valves = self.Valves()
        self.logger = logging.getLogger("simple_rag_pipeline")
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger.setLevel(logging.INFO)

    async def on_startup(self):
        try:
            from llama_index.embeddings.ollama import OllamaEmbedding
            from llama_index.llms.ollama import Ollama
            from llama_index.core import VectorStoreIndex, Settings, Document
            
            self.logger.info("Starting simple RAG pipeline initialization")
            
            # Configure LlamaIndex settings
            Settings.embed_model = OllamaEmbedding(
                model_name=self.valves.EMBEDDING_MODEL,
                base_url=self.valves.OLLAMA_BASE_URL,
            )
            Settings.llm = Ollama(
                model=self.valves.LLM_MODEL,
                base_url=self.valves.OLLAMA_BASE_URL,
            )
            
            self.logger.info("LlamaIndex settings configured")
            
            # Create a simple document from the sample text
            sample_text = self.valves.SAMPLE_TEXT
            
            # Add some more sample text for better testing
            sample_text += """
            
            Open WebUI is a user interface for interacting with various AI models and services.
            
            Features of Open WebUI include:
            1. Chat interface for conversational AI
            2. Support for multiple models
            3. Custom pipelines for specialized tasks
            4. Retrieval Augmented Generation (RAG) capabilities
            5. Integration with various AI services
            
            RAG (Retrieval Augmented Generation) is a technique that enhances language models by retrieving relevant information from a knowledge base before generating a response. This helps provide more accurate and contextually relevant answers.
            
            LlamaIndex is a data framework for LLM applications to ingest, structure, and access private or domain-specific data. It provides tools for building RAG pipelines.
            
            Ollama is a tool for running large language models locally. It provides an API for accessing these models, which can be used by Open WebUI.
            """
            
            # Create a document
            document = Document(text=sample_text)
            
            # Create a vector index
            self.logger.info("Creating vector index from sample text")
            self.index = VectorStoreIndex.from_documents([document])
            
            self.logger.info("Simple RAG pipeline initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error during pipeline initialization: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())

    async def on_shutdown(self):
        # Clean up resources if needed
        self.index = None
        self.logger.info("Pipeline shutdown complete")

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        """
        Process the user query using the simple RAG pipeline
        """
        try:
            self.logger.info(f"Processing query: {user_message}")
            
            if not self.index:
                return "The RAG pipeline is not initialized. Please check the logs."
            
            # Create a query engine
            query_engine = self.index.as_query_engine(streaming=True)
            
            # Query the index
            self.logger.info("Querying the index")
            response = query_engine.query(user_message)
            
            self.logger.info("Query processed successfully")
            return response.response_gen
            
        except Exception as e:
            self.logger.error(f"Error in pipe method: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return f"An error occurred while processing your query: {str(e)}"