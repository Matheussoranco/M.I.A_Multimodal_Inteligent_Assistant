#!/usr/bin/env python3
"""
M.I.A Web UI - Lightweight Standalone Version
This is a self-contained web UI that connects to Ollama for AI capabilities.
All ML is done by Ollama - this is just the chat interface.
"""

import os
import sys
import json
import time
import uuid
import asyncio
import threading
import webbrowser
from datetime import datetime
from typing import Optional, List, Dict, Any, AsyncGenerator

# Set environment to prevent interactive prompts
os.environ['TESTING'] = 'true'

def ensure_dependencies():
    """Check and report missing dependencies."""
    missing = []
    try:
        import fastapi
    except ImportError:
        missing.append('fastapi')
    try:
        import uvicorn
    except ImportError:
        missing.append('uvicorn')
    try:
        import requests
    except ImportError:
        missing.append('requests')
    
    if missing:
        print(f"\n❌ Missing required packages: {', '.join(missing)}")
        print("\nPlease install them with:")
        print(f"  pip install {' '.join(missing)}")
        input("\nPress Enter to exit...")
        sys.exit(1)

ensure_dependencies()

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel
import requests
import uvicorn

# ============================================================
# Configuration
# ============================================================
OLLAMA_BASE_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = "qwen2.5:3b-instruct-q4_K_M"
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8080

# ============================================================
# Pydantic Models
# ============================================================
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: bool = True

class ModelSwitchRequest(BaseModel):
    model: str

# ============================================================
# Ollama API Client
# ============================================================
class OllamaClient:
    """Simple client for Ollama API."""
    
    def __init__(self, base_url: str = OLLAMA_BASE_URL):
        self.base_url = base_url.rstrip('/')
        self.current_model = DEFAULT_MODEL
    
    def list_models(self) -> List[Dict[str, Any]]:
        """Get list of available Ollama models."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return data.get("models", [])
        except Exception as e:
            print(f"Warning: Could not connect to Ollama: {e}")
        return []
    
    def is_available(self) -> bool:
        """Check if Ollama is running."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    async def chat_stream(self, model: str, messages: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
        """Stream chat completion from Ollama."""
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={"model": model, "messages": messages, "stream": True},
                stream=True,
                timeout=300
            )
            
            if response.status_code != 200:
                yield f"data: {json.dumps({'error': f'Ollama error: {response.status_code}'})}\n\n"
                return
            
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        if "message" in data and "content" in data["message"]:
                            content = data["message"]["content"]
                            yield f"data: {json.dumps({'content': content})}\n\n"
                        if data.get("done", False):
                            yield f"data: {json.dumps({'done': True})}\n\n"
                            break
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

# ============================================================
# FastAPI Application
# ============================================================
app = FastAPI(title="M.I.A - Multimodal Intelligent Assistant")
ollama_client = OllamaClient()

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the Ollama-style web UI."""
    return get_html_ui()

@app.get("/api/models")
async def get_models():
    """Get available Ollama models."""
    models = ollama_client.list_models()
    return {
        "models": models,
        "current_model": ollama_client.current_model,
        "ollama_available": ollama_client.is_available()
    }

@app.post("/api/model")
async def switch_model(request: ModelSwitchRequest):
    """Switch the current model."""
    ollama_client.current_model = request.model
    return {"success": True, "model": request.model}

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Handle chat requests with streaming."""
    if not ollama_client.is_available():
        raise HTTPException(status_code=503, detail="Ollama is not available. Please ensure Ollama is running.")
    
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    
    if request.stream:
        return StreamingResponse(
            ollama_client.chat_stream(request.model, messages),
            media_type="text/event-stream"
        )
    else:
        # Non-streaming response
        response = requests.post(
            f"{ollama_client.base_url}/api/chat",
            json={"model": request.model, "messages": messages, "stream": False},
            timeout=300
        )
        return response.json()

@app.get("/api/status")
async def status():
    """Get server status."""
    return {
        "status": "online",
        "ollama_available": ollama_client.is_available(),
        "current_model": ollama_client.current_model,
        "timestamp": datetime.now().isoformat()
    }

# ============================================================
# HTML UI (Embedded)
# ============================================================
def get_html_ui() -> str:
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>M.I.A - Multimodal Intelligent Assistant</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a1a;
            color: #e0e0e0;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }
        .header {
            background: #2a2a2a;
            padding: 1rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #333;
        }
        .logo { font-size: 1.5rem; font-weight: bold; color: #4ade80; }
        .model-selector {
            background: #333;
            border: 1px solid #444;
            color: #e0e0e0;
            padding: 0.5rem 1rem;
            border-radius: 8px;
            cursor: pointer;
        }
        .model-selector:hover { border-color: #4ade80; }
        .main-container {
            flex: 1;
            display: flex;
            flex-direction: column;
            max-width: 900px;
            margin: 0 auto;
            width: 100%;
            padding: 2rem;
        }
        .chat-container {
            flex: 1;
            overflow-y: auto;
            padding: 1rem 0;
        }
        .message {
            margin-bottom: 1.5rem;
            animation: fadeIn 0.3s ease;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .message-role {
            font-size: 0.85rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
            color: #888;
        }
        .message-role.assistant { color: #4ade80; }
        .message-role.user { color: #60a5fa; }
        .message-content {
            background: #2a2a2a;
            padding: 1rem 1.25rem;
            border-radius: 12px;
            line-height: 1.6;
            white-space: pre-wrap;
        }
        .message-content code {
            background: #1a1a1a;
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
            font-family: 'Fira Code', monospace;
        }
        .message-content pre {
            background: #1a1a1a;
            padding: 1rem;
            border-radius: 8px;
            overflow-x: auto;
            margin: 0.5rem 0;
        }
        .input-area {
            background: #2a2a2a;
            border-radius: 16px;
            padding: 1rem;
            margin-top: 1rem;
        }
        .input-row {
            display: flex;
            gap: 0.75rem;
        }
        #user-input {
            flex: 1;
            background: #333;
            border: 1px solid #444;
            color: #e0e0e0;
            padding: 0.75rem 1rem;
            border-radius: 12px;
            font-size: 1rem;
            resize: none;
            min-height: 50px;
            max-height: 200px;
        }
        #user-input:focus {
            outline: none;
            border-color: #4ade80;
        }
        #send-btn {
            background: #4ade80;
            color: #1a1a1a;
            border: none;
            border-radius: 12px;
            padding: 0 1.5rem;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }
        #send-btn:hover { background: #22c55e; }
        #send-btn:disabled { background: #333; color: #666; cursor: not-allowed; }
        .status-bar {
            text-align: center;
            padding: 0.5rem;
            font-size: 0.85rem;
            color: #666;
        }
        .status-bar.connected { color: #4ade80; }
        .status-bar.error { color: #ef4444; }
        .welcome {
            text-align: center;
            padding: 3rem;
        }
        .welcome h1 { font-size: 2.5rem; margin-bottom: 1rem; }
        .welcome p { color: #888; margin-bottom: 2rem; }
        .typing-indicator {
            display: flex;
            gap: 4px;
            padding: 0.5rem;
        }
        .typing-indicator span {
            width: 8px;
            height: 8px;
            background: #4ade80;
            border-radius: 50%;
            animation: bounce 1.4s infinite;
        }
        .typing-indicator span:nth-child(2) { animation-delay: 0.2s; }
        .typing-indicator span:nth-child(3) { animation-delay: 0.4s; }
        @keyframes bounce {
            0%, 60%, 100% { transform: translateY(0); }
            30% { transform: translateY(-8px); }
        }
    </style>
</head>
<body>
    <header class="header">
        <div class="logo">🧠 M.I.A</div>
        <select class="model-selector" id="model-select">
            <option value="">Loading models...</option>
        </select>
    </header>
    
    <main class="main-container">
        <div class="chat-container" id="chat-container">
            <div class="welcome">
                <h1>Welcome to M.I.A</h1>
                <p>Your Multimodal Intelligent Assistant</p>
                <p>Select a model above and start chatting!</p>
            </div>
        </div>
        
        <div class="input-area">
            <div class="input-row">
                <textarea id="user-input" placeholder="Type your message..." rows="1"></textarea>
                <button id="send-btn">Send</button>
            </div>
        </div>
        
        <div class="status-bar" id="status-bar">Connecting to Ollama...</div>
    </main>

    <script>
        const chatContainer = document.getElementById('chat-container');
        const userInput = document.getElementById('user-input');
        const sendBtn = document.getElementById('send-btn');
        const modelSelect = document.getElementById('model-select');
        const statusBar = document.getElementById('status-bar');
        
        let messages = [];
        let isGenerating = false;
        let currentModel = '';
        
        // Auto-resize textarea
        userInput.addEventListener('input', () => {
            userInput.style.height = 'auto';
            userInput.style.height = Math.min(userInput.scrollHeight, 200) + 'px';
        });
        
        // Load models on startup
        async function loadModels() {
            try {
                const response = await fetch('/api/models');
                const data = await response.json();
                
                modelSelect.innerHTML = '';
                
                if (!data.ollama_available) {
                    statusBar.textContent = '⚠️ Ollama not running - please start Ollama';
                    statusBar.className = 'status-bar error';
                    modelSelect.innerHTML = '<option value="">Ollama not available</option>';
                    return;
                }
                
                if (data.models.length === 0) {
                    modelSelect.innerHTML = '<option value="">No models found</option>';
                    statusBar.textContent = '⚠️ No models installed - run: ollama pull llama3.2';
                    return;
                }
                
                data.models.forEach(model => {
                    const option = document.createElement('option');
                    option.value = model.name;
                    option.textContent = model.name;
                    if (model.name === data.current_model) option.selected = true;
                    modelSelect.appendChild(option);
                });
                
                currentModel = modelSelect.value || data.models[0].name;
                statusBar.textContent = '✓ Connected to Ollama';
                statusBar.className = 'status-bar connected';
                
            } catch (error) {
                statusBar.textContent = '❌ Cannot connect to server';
                statusBar.className = 'status-bar error';
            }
        }
        
        // Handle model change
        modelSelect.addEventListener('change', async () => {
            currentModel = modelSelect.value;
            await fetch('/api/model', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model: currentModel })
            });
        });
        
        // Add message to chat
        function addMessage(role, content) {
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message';
            messageDiv.innerHTML = `
                <div class="message-role ${role}">${role === 'assistant' ? '🤖 Assistant' : '👤 You'}</div>
                <div class="message-content">${escapeHtml(content)}</div>
            `;
            
            // Remove welcome message if exists
            const welcome = chatContainer.querySelector('.welcome');
            if (welcome) welcome.remove();
            
            chatContainer.appendChild(messageDiv);
            chatContainer.scrollTop = chatContainer.scrollHeight;
            return messageDiv;
        }
        
        // Show typing indicator
        function showTyping() {
            const typingDiv = document.createElement('div');
            typingDiv.className = 'message';
            typingDiv.id = 'typing-indicator';
            typingDiv.innerHTML = `
                <div class="message-role assistant">🤖 Assistant</div>
                <div class="message-content">
                    <div class="typing-indicator">
                        <span></span><span></span><span></span>
                    </div>
                </div>
            `;
            chatContainer.appendChild(typingDiv);
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
        
        function removeTyping() {
            const typing = document.getElementById('typing-indicator');
            if (typing) typing.remove();
        }
        
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
        
        // Send message
        async function sendMessage() {
            const content = userInput.value.trim();
            if (!content || isGenerating || !currentModel) return;
            
            isGenerating = true;
            sendBtn.disabled = true;
            userInput.value = '';
            userInput.style.height = 'auto';
            
            // Add user message
            messages.push({ role: 'user', content });
            addMessage('user', content);
            
            // Show typing indicator
            showTyping();
            
            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        model: currentModel,
                        messages: messages,
                        stream: true
                    })
                });
                
                removeTyping();
                
                if (!response.ok) {
                    throw new Error(`Error: ${response.status}`);
                }
                
                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let assistantContent = '';
                let messageDiv = addMessage('assistant', '');
                const contentDiv = messageDiv.querySelector('.message-content');
                
                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    
                    const chunk = decoder.decode(value);
                    const lines = chunk.split('\\n');
                    
                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            try {
                                const data = JSON.parse(line.slice(6));
                                if (data.content) {
                                    assistantContent += data.content;
                                    contentDiv.textContent = assistantContent;
                                    chatContainer.scrollTop = chatContainer.scrollHeight;
                                }
                                if (data.error) {
                                    contentDiv.textContent = `Error: ${data.error}`;
                                }
                            } catch (e) {}
                        }
                    }
                }
                
                messages.push({ role: 'assistant', content: assistantContent });
                
            } catch (error) {
                removeTyping();
                addMessage('assistant', `Error: ${error.message}`);
            }
            
            isGenerating = false;
            sendBtn.disabled = false;
            userInput.focus();
        }
        
        // Event listeners
        sendBtn.addEventListener('click', sendMessage);
        userInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
        
        // Initialize
        loadModels();
        setInterval(loadModels, 30000); // Refresh models every 30s
    </script>
</body>
</html>'''

# ============================================================
# Main Entry Point
# ============================================================
def open_browser(port: int):
    """Open browser after a short delay."""
    time.sleep(1.5)
    webbrowser.open(f'http://{SERVER_HOST}:{port}')

def main():
    """Main entry point."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║     🧠  M.I.A - Multimodal Intelligent Assistant                             ║
║         Lightweight Web Interface                                            ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Check Ollama
    print("    📡 Checking Ollama connection...")
    if ollama_client.is_available():
        models = ollama_client.list_models()
        print(f"    ✓ Ollama connected - {len(models)} model(s) available")
        if models:
            ollama_client.current_model = models[0].get('name', DEFAULT_MODEL)
            print(f"    ✓ Default model: {ollama_client.current_model}")
    else:
        print("    ⚠ Ollama not running - please start Ollama for AI features")
    
    print()
    print(f"    🌐 Starting server at http://{SERVER_HOST}:{SERVER_PORT}")
    print(f"    📝 Press Ctrl+C to stop")
    print()
    
    # Open browser in background
    browser_thread = threading.Thread(target=open_browser, args=(SERVER_PORT,), daemon=True)
    browser_thread.start()
    
    # Start server
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT, log_level="warning")

if __name__ == "__main__":
    main()
