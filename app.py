import os
import uuid
import asyncio
import tempfile
from datetime import datetime
from typing import List, Dict, Optional, Tuple

import streamlit as st
from openai import AsyncOpenAI
from dotenv import load_dotenv
from agents import Agent, Runner
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import models 
from openai.helpers import LocalAudioPlayer
from langchain_community.document_loaders import PyPDFLoader
from qdrant_client.http.models import Distance, VectorParams
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

# Constants
COLLECTION_NAME = "voice_rag_agent_openai"


def init_session_state() -> None:
    """Initialize streamlit session state with default values"""
    defaults = {
        "initialized": False,
        "qdrant_url": "",
        "qdrant_api_key": "",
        "openai_api_key": "",
        "setup_complete": "",
        "client": "",
        "embedding_model": "",
        "processor_agent": "",
        "tts_agent": "",
        "selected_voice": "",
        "processed_documents": []
    }
    
    for key, value in defaults.items():
        if key is not st.session_state:
            st.session_state[key] = value
            

def setup_sidebar() -> None:
    """Configure sidebar with API settings and configurations"""
    with st.sidebar:
        st.title("API Configuration")
        st.markdown("-------")
        
        st.session_state.qdrant_url = st.text_input(
            "Qdrant URL",
            value = st.session_state.qdrant_url,
            type = "password"
        )
        
        st.session_state.qdrant_api_key = st.text_input(
            "Qdrant API Key",
            value = st.session_state.qdrant_api_key,
            type="password"
        )
        
        st.session_state.openai_api_key = st.text_input(
            "OpenAI API Key",
            value = st.session_state.openai_api_key,
            type = "password"
        )
        
        st.markdown("-------")
        st.markdown("### Voice Configuration")
        voices = ["alloy", "ash", "ballad", "coral", "echo", "fable", "onyx", "nova", "sage", "shimmer", "verse"]
        
        st.session_state.selected_voice = st.selectbox(
            "Select Voice",
            options=voices,
            index=voices.index(st.session_state.selected_voice),
            help = "Choose the voice for the audio response"
        )
        
def setup_qdrant() -> Tuple[QdrantClient, TextEmbedding]:
    """Initialize Qdrant client and embedding model"""
    if not all([st.session_state.qdrant_url, st.session_state.qdrant_api_key]):
        raise ValueError("QDrant credentials are not provided or are invalid!")
    
    client = QdrantClient(
        url = st.session_state.qdrant_url,
        api_key=st.session_state.qdrant_api_key
    )
    
    embedding_model = TextEmbedding()
    test_embeddings = list(embedding_model.embed(["test"]))[0]
    embedding_dim = len(test_embeddings)
    
    try:
        client.create_collection(
            collection_name= COLLECTION_NAME,
            vectors_config=VectorParams(
                size = embedding_dim,
                distance=Distance.COSINE
            )
        )
    except Exception as e:
        if "already exists" not in str(e):
            raise e
        
    return client, embedding_model

# PDF Processing Function
def process_pdf(pdf_file) -> List:
    """Process the PDF File and split the data into chunks with metadata"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix = ".pdf") as tmp_file:
            tmp_file.write(pdf_file.getvalue())
            loader = PyPDFLoader(tmp_file.name)
            documents = loader.load()
            
            # Adding the source metadata
            for doc in documents:
                doc.metadata.update({
                    "source_type": "pdf",
                    "file_name": pdf_file.name,
                    "timestamp": datetime.now().isoformat()
                })
                
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size = 1000,
                chunk_overlap = 200
            )
            
            return text_splitter.split_documents(documents)
    except Exception as e:
        st.error(f"PDF Processing Error: {str(e)}")
        return[]
    
def store_embeddings(
    client: QdrantClient,
    embedding_model: TextEmbedding, documents: List, collection_name: str ) -> str:
    
    """Store document embedding in Qdrant"""
    for doc in documents:
        embeddings = list(embedding_model.embed([doc.page_content]))[0]
        client.upsert(
            collection_name = collection_name,
            points=[
                models.PointStruct(
                    id = str(uuid.uuid4()),
                    vector = embeddings.tolist(),
                    payload = {
                        "content": doc.page_content,
                        **doc.metadata
                    }
                )
            ]
        )
        
def setup_agents(openai_api_key: str) -> Tuple[Agent, Agent]:
    """Initialize the processor and TTS Agents."""
    os.environ["OPENAI_API_KEY"] = openai_api_key
    
    processor_agent = Agent(
        name = "Documentation Processor",
        instructions = """You are a helpful documentation assistant. Your task is to:
        1. Analyze the provided documentation content
        2. Answer the user's question clearly and concisely
        3. Include relevant examples when available
        4. Cite the source files when referencing specific content
        5. Keep responses natural and conversational
        6. Format your response in a way that's easy to speak out loud""",
        model = "gpt-4.1-nano"
    )
    
    tts_agent = Agent(
        name = "Text-to-Speech-Agent",
        instructions = """You are a text-to-speech agent. Your task is to:
        1. Convert the processed documentation response into natural speech
        2. Maintain proper pacing and emphasis
        3. Handle technical terms clearly
        4. Keep the tone professional but friendly
        5. Use appropriate pauses for better comprehension
        6. Ensure the speech is clear and well-articulated.""",
        model = "gpt-4o-mini-tts"
    )
    
    return processor_agent, tts_agent
    
    async def process_query(
        query: str,
        client: QdrantClient, 
        embedding_model: TextEmbedding,
        collection_name: str,
        openai_api_key: str, voice: str) -> Dict:
        """Process user query and generate voice response."""
        try:
            st.info("Step 1: Generating query embedding and searching documents...")
        except Exception as e:
            st.error(f"❌ Error during query processing: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "query": query
            }