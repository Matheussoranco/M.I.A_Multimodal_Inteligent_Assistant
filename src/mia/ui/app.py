import streamlit as st
import sys
import os
import logging
from pathlib import Path

# Add src to path
current_dir = Path(__file__).parent.parent.parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from mia.main import process_with_llm, detect_and_execute_agent_commands
from mia.providers import provider_registry
from mia.config_manager import ConfigManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="M.I.A - Multimodal Intelligent Assistant",
    page_icon="�",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Redirect notice
st.markdown("""
<div style="background: linear-gradient(135deg, #1f6feb, #a371f7); padding: 20px; border-radius: 12px; margin-bottom: 20px; text-align: center;">
    <h2 style="color: white; margin: 0;">🚀 New Ollama-Style Web UI Available!</h2>
    <p style="color: rgba(255,255,255,0.9); margin: 10px 0;">
        Run <code style="background: rgba(0,0,0,0.2); padding: 4px 8px; border-radius: 4px;">mia --web</code> or 
        <code style="background: rgba(0,0,0,0.2); padding: 4px 8px; border-radius: 4px;">python -m mia.web.webui</code> 
        for the new AGI-focused interface!
    </p>
</div>
""", unsafe_allow_html=True)

# Custom CSS for styling
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .stChatMessage[data-testid="stChatMessageUser"] {
        background-color: #2b313e;
    }
    .stChatMessage[data-testid="stChatMessageAssistant"] {
        background-color: #1e2329;
        border: 1px solid #3e4c5e;
    }
    .stTextInput input {
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def init_mia_components():
    """Initialize M.I.A components once."""
    components = {}
    
    # Config
    try:
        config_manager = ConfigManager()
        config_manager.load_config()
        components["config_manager"] = config_manager
    except Exception as e:
        logger.error(f"Config load error: {e}")
        components["config_manager"] = None

    # LLM
    try:
        components["llm"] = provider_registry.create("llm", config_manager=components["config_manager"])
    except Exception as e:
        logger.error(f"LLM init error: {e}")
        components["llm"] = None

    # Action Executor
    try:
        components["action_executor"] = provider_registry.create(
            "actions",
            config_manager=components["config_manager"]
        )
    except Exception as e:
        logger.error(f"Action Executor init error: {e}")
        components["action_executor"] = None
        
    # Web Agent (Implicitly used by Action Executor but good to have reference if needed)
    
    return components

def main():
    st.title("🤖 M.I.A")
    st.caption("Multimodal Intelligent Assistant")

    # Initialize components
    if "components" not in st.session_state:
        with st.spinner("Initializing M.I.A core systems..."):
            st.session_state.components = init_mia_components()
            st.success("System Online")

    # Sidebar
    with st.sidebar:
        st.header("System Status")
        components = st.session_state.components
        
        llm_status = "🟢 Online" if components.get("llm") else "🔴 Offline"
        st.metric("LLM Engine", llm_status)
        
        action_status = "🟢 Ready" if components.get("action_executor") else "🔴 Unavailable"
        st.metric("Action Executor", action_status)
        
        st.divider()
        st.markdown("### Capabilities")
        st.markdown("- 🧠 **Reasoning**")
        st.markdown("- 🖥️ **Desktop Control**")
        st.markdown("- 🌐 **Web Automation**")
        st.markdown("- 📁 **File Management**")

    # Chat History
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Input
    if prompt := st.chat_input("How can I help you today?"):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Process with M.I.A
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Thinking...")
            
            try:
                components = st.session_state.components
                
                # Use the existing process_with_llm logic from main.py
                # We need to mock 'inputs' as it's not used heavily in text mode
                inputs = {} 
                
                # We can reuse the logic from main.py or call components directly
                # Calling process_with_llm is better to keep consistency
                
                result = process_with_llm(prompt, inputs, components)
                
                response_text = ""
                if result.get("error"):
                    response_text = f"⚠️ Error: {result['error']}"
                elif result.get("response"):
                    response_text = result["response"]
                else:
                    response_text = "I couldn't generate a response."

                message_placeholder.markdown(response_text)
                
                # Add assistant response to history
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                
            except Exception as e:
                error_msg = f"An error occurred: {str(e)}"
                message_placeholder.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    main()
