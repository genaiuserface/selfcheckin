"""
chatbot_app.py - Production-ready ChatGPT-style Streamlit Chatbot
"""

import streamlit as st
import requests
import uuid
from datetime import datetime
import json

# ===================================================================
# PAGE CONFIGURATION
# ===================================================================

st.set_page_config(
    page_title="AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for ChatGPT-like appearance
st.markdown("""
<style>
    /* Main chat container */
    .stApp {
        background-color: #f7f7f8;
    }
    
    /* Message bubbles */
    div.stMarkdown {
        background-color: white;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
    }
    
    /* Chat input fixed at bottom */
    .stChatInput {
        position: fixed;
        bottom: 0;
        background-color: white;
        padding: 20px;
        border-top: 1px solid #ddd;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ===================================================================
# SESSION STATE INITIALIZATION
# ===================================================================

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "user_id" not in st.session_state:
    st.session_state.user_id = f"user_{uuid.uuid4().hex[:8]}"

# ===================================================================
# CONFIGURATION (Change this to your backend URL later)
# ===================================================================

# TODO: Replace with your actual backend URL
BACKEND_URL = "http://localhost:8000/process"
USE_MOCK = True  # Set to False when backend is ready

# ===================================================================
# BACKEND COMMUNICATION
# ===================================================================

def call_backend(query: str, session_id: str, user_id: str) -> dict:
    """Call backend API or return mock response"""
    
    if USE_MOCK:
        # Mock responses for testing
        import time
        import random
        
        time.sleep(1)  # Simulate processing time
        
        mock_responses = [
            "I understand your question. Here's what I can tell you about that topic...",
            "That's an interesting question! Let me explain...",
            "Based on my analysis, here's what I found...",
            "Great question! Here's a comprehensive answer...",
        ]
        
        return {
            "response": random.choice(mock_responses),
            "session_id": session_id,
            "agent_path": ["Agent-A"],
            "processing_time": random.uniform(0.5, 2.0),
            "metadata": {"mock": True}
        }
    
    else:
        # Actual backend call
        try:
            response = requests.post(
                BACKEND_URL,
                json={
                    "query": query,
                    "session_id": session_id,
                    "user_id": user_id,
                    "context": {
                        "timestamp": datetime.now().isoformat(),
                        "message_count": len(st.session_state.messages)
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "response": f"Error: Backend returned status {response.status_code}",
                    "error": True
                }
                
        except requests.exceptions.ConnectionError:
            return {
                "response": "⚠️ Cannot connect to backend. Please ensure the service is running.",
                "error": True
            }
        except Exception as e:
            return {
                "response": f"Error: {str(e)}",
                "error": True
            }

# ===================================================================
# SIDEBAR (Optional - for settings/info)
# ===================================================================

with st.sidebar:
    st.title("ℹ️ Session Info")
    st.text(f"Session ID: {st.session_state.session_id[:8]}...")
    st.text(f"User ID: {st.session_state.user_id}")
    st.text(f"Messages: {len(st.session_state.messages)}")
    
    st.divider()
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()
    
    st.divider()
    
    # Show backend status
    if USE_MOCK:
        st.warning("Using mock responses")
    else:
        st.success(f"Connected to: {BACKEND_URL}")

# ===================================================================
# MAIN CHAT INTERFACE
# ===================================================================

# Title
st.title("🤖 AI Assistant")

# Chat messages container
chat_container = st.container()

# Display chat messages (newest at bottom, like ChatGPT)
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show metadata if available
            if "metadata" in message and message["role"] == "assistant":
                with st.expander("Details", expanded=False):
                    if "agent_path" in message["metadata"]:
                        st.text(f"Agents: {' → '.join(message['metadata']['agent_path'])}")
                    if "processing_time" in message["metadata"]:
                        st.text(f"Time: {message['metadata']['processing_time']:.2f}s")

# Chat input at the bottom - moved to right side
col1, col2 = st.columns([1, 2])
with col2:
    if prompt := st.chat_input("Type your message here...", key="chat_input"):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message immediately
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)
        
        # Show typing indicator
        with chat_container:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Call backend
                    response_data = call_backend(
                        query=prompt,
                        session_id=st.session_state.session_id,
                        user_id=st.session_state.user_id
                    )
                    
                    # Extract response
                    response_text = response_data.get("response", "No response received")
                    
                    # Add assistant message to chat
                    assistant_message = {
                        "role": "assistant",
                        "content": response_text,
                        "metadata": {
                            "agent_path": response_data.get("agent_path", []),
                            "processing_time": response_data.get("processing_time", 0),
                            "session_id": response_data.get("session_id", ""),
                        }
                    }
                    
                    st.session_state.messages.append(assistant_message)
                    
                    # Display response
                    st.markdown(response_text)
                    
                    # Show metadata
                    if not response_data.get("error"):
                        with st.expander("Details", expanded=False):
                            if "agent_path" in response_data:
                                st.text(f"Agents: {' → '.join(response_data['agent_path'])}")
                            if "processing_time" in response_data:
                                st.text(f"Time: {response_data['processing_time']:.2f}s")
        
        # Rerun to update chat display
        st.rerun()

# ===================================================================
# FOOTER INFO
# ===================================================================

# Show connection status at bottom
status_container = st.container()
with status_container:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if not USE_MOCK:
            st.caption(f"🟢 Connected to backend")

# ===================================================================
# RUN INSTRUCTIONS
# ===================================================================

# Run with: streamlit run chatbot_app.py
# To connect to your backend:
# 1. Set USE_MOCK = False
# 2. Update BACKEND_URL = "http://your-backend-url/process"
