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
        st.info("🔄️ Step 1: Generating query embedding and searching documents...")
        # Getting query embedding and search
        query_embeddings = list(embedding_model.embed([query]))[0]
        st.write(f"Generated Embedding of size: {len(query_embeddings)}")
        
        search_resp = client.query_points(
            collection_name = collection_name,
            query = query_embeddings.tolist(),
            limit = 3,
            with_payload=True
        )
        
        search_results = search_resp.points if hasattr(search_resp, "points") else []
        st.write(f"Found {len(search_results)} relevant documents")
        
        if not search_results:
            raise Exception("No relevant documents found in the vector database")
        
        if not search_results:
            raise Exception("No relevant documents found in the vector database")
        
        st.info("🔄️ Step 2: Preparing context from search results...")
        # Prepare context from search results
        context = "Based on the following documentation: \n\n"
        for i, result in enumerate(search_results, 1):
            payload = result.payload
            if not payload:
                continue
            content = payload.get("content", "")
            source = payload.get("file_name", "Unknown source")
            context += f"From {source}: \n{content}\n\n"
            st.write(f"Document {i} from {source}")
        context += f"\nUser Question: {query}\n\n"
        context += "Please provide a clear, concise answer that can be easily spoken out loud."
        
        st.info("🔄️ Step 3: Setting up agents..")
        # Setting up agent if not initiated already
        if not st.session_state.processor_agent or not st.session_state.tts_agent:
            processor_agent, tts_agent = setup_agents(openai_api_key)
            st.session_state.processor_agent = st.session_state.processor_agent = processor_agent
            st.session_state.tts_agent = st.session_state.tts_agent = tts_agent
            st.write("Initialized new processor and TTS agents")
        else:
            st.write("Using Existing Agents")
        
        st.info("🔄️ Step 4: Generating text responses")
        # Generate text repsonse using processor agent
        processor_results = await Runner.run(st.session_state.processor_agent, context)
        text_response = processor_results.final_output
        st.write(f"Generated Text Response of length: {len(text_response)}")
        
        st.info("🔄️ Step 5: Generating voice instructions...")
        # Generating voice instructions using TTS Agent
        tts_results = await Runner.run(st.session_state.tts_agent, text_response)
        voice_instructions = tts_results.final_output
        st.write(f"Generated Voice Instruction of length: {len(voice_instructions)}")
        
        st.info("🔄️ Step 6: Preparing context from search results...")
        # Generate and plat the audio
        async_openai = AsyncOpenAI(api_key=openai_api_key)
        
        # First Create streaming response
        async with async_openai.audio.speech.with_streaming_response.create(
            model = "gpt-4o-mini-tts",
            voice = voice,
            input = text_response,
            instructions = voice_instructions,
            response_format = "pcm",) as stream_resp:
            st.write("Starting Playback:")
            await LocalAudioPlayer().play(stream_resp)
            st.write("Audio Playback Complete")
            
            st.write("Generating downloading MP3 version...")
            # Also save as MP3 for download
            audio_resp = await async_openai.audio.speech.create(
                model = "gpt-4o-mini-tts",
                voice = voice,
                input = text_response,
                instructions = voice_instructions,
                response_format="mp3"
            )
            
            temp_dir = tempfile.gettempdir()
            audio_path = os.path.join(temp_dir, f"response_{uuid.uuid4()}.mp3")
            
            with open(audio_path, "wb") as f:
                f.write(audio_resp.content)
            st.write(f"Saved MP3 file to: {audio_path}")
            
            st.success("✅ Query Processing Complete")
            return {
                "status": "success",
                "text_response": text_response,
                "voice_instructions": voice_instructions,
                "audio_path": audio_path,
                "sources": [r.payload.get("file_name", "Unknown Source") for r in search_results if r.payload]
            }
            
    except Exception as e:
        st.error(f"❌ Error during query processing: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "query": query
        }

def main() -> None:
    """Main Application Function"""
    st.set_page_config(
        page_title = "Voice RAG Agent",
        page_icon = "🎙️",
        layout = "wide"
    )
    
    init_session_state()
    setup_sidebar()
    
    # File upload section
    uploaded_file = st.file_uploader("Upload PDF", type = ["pdf"])
    
    if uploaded_file:
        file_name = uploaded_file.name
        
        if file_name not in st.session_state.processed_documents:
            with st.spinner("Processing PDF..."):
                try:
                    # Setup QDrant if not already done
                    if not st.session_state.client:
                        client, embedding_model = setup_qdrant()
                        st.session_state.client = client
                        st.session_state.embedding_model = embedding_model
                        
                    # Process and Store Documents
                    docs = process_pdf(uploaded_file)
                    if docs:
                        store_embeddings(
                            st.session_state.client, 
                            st.session_state.embedding_model, 
                            docs, 
                            COLLECTION_NAME
                        )
                        
                        st.session_state.processed_documents.append(file_name)
                        st.success(f"Added PDF: {file_name}")
                        st.session_state.setup_complete = True
                except Exception as e:
                    st.error(f"Error Processing Documents: {str(e)}")
    
    if st.session_state.processed_documents:
        st.sidebar.header("Processed Documents")                
        for doc in st.session_state.processed_documents:
            st.sidebar.text(f"📄 {doc}")
    
    # Query Interface
    query = st.text_input(
        "What would you like to ask from the documentation?",
        placeholder = "e.g How do I authenticate API requests?",
        disabled = not st.session_state.setup_complete
    )
    
    if query and st.session_state.setup_complete:
        with st.status("Processing your query...", expanded = True) as status:
            try:
                result = asyncio.run(process_query(
                    query,
                    st.session_state.client, 
                    st.session_state.embedding_model, 
                    COLLECTION_NAME, 
                    st.session_state.openai_api_key,
                    st.session_state.selected_voices
                ))

                if result["status"] == "success":
                    status.update(label = "✅ Query Processed!", status = "complete")
                    
                    st.markdown("### Markdown")
                    st.write(result["text_response"])
                    
                    if "audio_path" in result:
                        st.markdown(f"### 🔊 Audio Response (Voice: {st.session_state.selected_voice})")
                        st.audio(result["audio_path"], format="audio/mp3", start_time=0)
                        
                        with open(result["audio_path"], "rb") as audio_file:
                            audio_bytes = audio_file.read()
                            st.download_button(
                                label="📥 Download Audio Response",
                                data=audio_bytes,
                                file_name=f"voice_response_{st.session_state.selected_voice}.mp3",
                                mime="audio/mp3"
                            )
                    
                    st.markdown("### Sources:")
                    for source in result["sources"]:
                        st.markdown(f"- {source}")
                else:
                    status.update(label="❌ Error processing query", state="error")
                    st.error(f"Error: {result.get('error', 'Unknown error occurred')}")
            
            except Exception as e:
                status.update(label="❌ Error processing query", state="error")
                st.error(f"Error processing query: {str(e)}")
    
    elif not st.session_state.setup_complete:
        st.info("👈 Please configure the system and upload documents first!")

if __name__ == "__main__":
    main()