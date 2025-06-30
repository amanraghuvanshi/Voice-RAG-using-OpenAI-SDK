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
    
    embedding = TextEmbedding()
    