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