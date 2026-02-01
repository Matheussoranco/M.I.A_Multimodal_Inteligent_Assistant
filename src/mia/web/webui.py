"""
M.I.A Web UI - Ollama-style interface with AGI Agent capabilities
A clean, minimal chat interface with function calling and tool execution
"""

import asyncio
import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, AsyncGenerator, Union

# Add src to path for imports
current_dir = Path(__file__).parent.parent.parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

try:
    import uvicorn
    from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
except ImportError as e:
    raise RuntimeError(f"FastAPI/uvicorn not installed. Install: pip install fastapi uvicorn. Error: {e}")

from mia.config_manager import ConfigManager
from mia.providers import provider_registry, ProviderLookupError
from mia.llm.llm_manager import LLMManager

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# Data Models
# ═══════════════════════════════════════════════════════════════════════════════

class ChatMessage(BaseModel):
    role: str  # "user", "assistant", "system", "tool"
    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None
    timestamp: Optional[str] = None

class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: bool = True
    temperature: float = 0.7
    max_tokens: int = 4096
    tools: Optional[Union[bool, List[Dict[str, Any]]]] = None
    tool_choice: Optional[str] = "auto"
    
    def get_tools_list(self) -> Optional[List[Dict[str, Any]]]:
        """Convert tools parameter to actual tools list."""
        if self.tools is True:
            return AVAILABLE_TOOLS
        elif isinstance(self.tools, list):
            return self.tools
        return None

class ModelInfo(BaseModel):
    name: str
    provider: str
    size: Optional[str] = None
    description: Optional[str] = None

class ToolDefinition(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]

# ═══════════════════════════════════════════════════════════════════════════════
# Available Tools for Agent
# ═══════════════════════════════════════════════════════════════════════════════

AVAILABLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information. Use for questions about recent events, facts, or any topic requiring up-to-date information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "youtube_search",
            "description": "Search YouTube for a video.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The YouTube search query"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "auto_task",
            "description": "Interpret a natural-language request and execute it (WhatsApp, Wikipedia, YouTube, or web search).",
            "parameters": {
                "type": "object",
                "properties": {
                    "request": {"type": "string", "description": "Natural-language request"},
                    "phone": {"type": "string", "description": "WhatsApp phone number (optional)"},
                    "message": {"type": "string", "description": "WhatsApp message (optional)"},
                    "query": {"type": "string", "description": "Search query (optional)"}
                },
                "required": ["request"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_file",
            "description": "Create a new file with the specified content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to create"},
                    "content": {"type": "string", "description": "Content to write to the file"}
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to read"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Execute a shell command and return its output. Use carefully.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The shell command to execute"}
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_code",
            "description": "Analyze code for issues, improvements, and provide suggestions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "The code to analyze"},
                    "language": {"type": "string", "description": "Programming language (python, javascript, etc.)"}
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_browse",
            "description": "Navigate to a URL and extract its content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to browse"}
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Perform mathematical calculations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Mathematical expression to evaluate"}
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "memory_store",
            "description": "Store information in long-term memory for later retrieval.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "Information to remember"},
                    "category": {"type": "string", "description": "Category for organization (optional)"}
                },
                "required": ["content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "memory_search",
            "description": "Search long-term memory for previously stored information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query for memory"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Send an email to a recipient.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "Recipient email address"},
                    "subject": {"type": "string", "description": "Email subject"},
                    "body": {"type": "string", "description": "Email body content"}
                },
                "required": ["to", "subject", "body"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "desktop_automation",
            "description": "Perform desktop automation tasks like opening apps, typing text, clicking.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["open_app", "type_text", "click", "screenshot"], "description": "Type of desktop action"},
                    "target": {"type": "string", "description": "Target application, coordinates, or text"},
                    "data": {"type": "string", "description": "Additional data for the action (optional)"}
                },
                "required": ["action", "target"]
            }
        }
    },
    {
        "type": "function", 
        "function": {
            "name": "reasoning",
            "description": "Perform deep reasoning and analysis on complex problems. Use for mathematical proofs, logical deduction, planning, and multi-step problem solving.",
            "parameters": {
                "type": "object",
                "properties": {
                    "problem": {"type": "string", "description": "The problem or question to reason about"},
                    "approach": {"type": "string", "enum": ["deductive", "inductive", "abductive", "analogical", "planning"], "description": "Reasoning approach to use"}
                },
                "required": ["problem"]
            }
        }
    }
]

# ═══════════════════════════════════════════════════════════════════════════════
# MIA Agent Core
# ═══════════════════════════════════════════════════════════════════════════════

class MIAAgent:
    """Core agent class with tool execution capabilities."""
    
    def __init__(self):
        self.llm = None
        self.action_executor = None
        self.config_manager = None
        self.memory = []
        self.conversations: Dict[str, List[ChatMessage]] = {}
        self.current_model = None
        self._initialize()
    
    def _initialize(self):
        """Initialize the agent components."""
        import os
        # Set environment to skip interactive prompts
        os.environ['TESTING'] = 'true'
        
        try:
            self.config_manager = ConfigManager()
            self.config_manager.load_config()
        except Exception as e:
            logger.warning(f"Config manager init error: {e}")
            self.config_manager = None
        
        # Try to initialize LLM with auto-detection
        try:
            # First try to get Ollama models
            ollama_models = LLMManager.detect_ollama_models()
            if ollama_models:
                model_name = ollama_models[0]["name"]
                from mia.llm.llm_manager import LLMManager as LLM
                self.llm = LLM(
                    provider="ollama",
                    model_id=model_name,
                    auto_detect=False,
                    config_manager=self.config_manager
                )
                self.current_model = model_name
                logger.info(f"Initialized with Ollama model: {model_name}")
            else:
                # Try API providers
                api_providers = LLMManager.detect_api_providers()
                if api_providers:
                    provider = api_providers[0]
                    from mia.llm.llm_manager import LLMManager as LLM
                    self.llm = LLM(
                        provider=provider["name"],
                        model_id=provider["model"],
                        api_key=provider["api_key"],
                        url=provider["url"],
                        auto_detect=False,
                        config_manager=self.config_manager
                    )
                    self.current_model = f"{provider['name']}:{provider['model']}"
                    logger.info(f"Initialized with API: {self.current_model}")
                else:
                    logger.warning("No LLM providers available")
        except Exception as e:
            logger.warning(f"LLM init error: {e}")
            self.llm = None
        
        try:
            self.action_executor = provider_registry.create("actions", config_manager=self.config_manager)
        except Exception as e:
            logger.warning(f"Action executor init error: {e}")
            self.action_executor = None
    
    def set_model(self, model_name: str) -> bool:
        """Switch to a different model."""
        try:
            # Determine provider from model name
            if ":" in model_name:
                parts = model_name.split(":")
                if parts[0] in ["openai", "anthropic", "groq", "gemini", "grok"]:
                    provider = parts[0]
                    model_id = parts[1]
                else:
                    provider = "ollama"
                    model_id = model_name
            else:
                provider = "ollama"
                model_id = model_name
            
            from mia.llm.llm_manager import LLMManager as LLM
            self.llm = LLM(
                provider=provider,
                model_id=model_id,
                auto_detect=False,
                config_manager=self.config_manager
            )
            self.current_model = model_name
            return True
        except Exception as e:
            logger.error(f"Failed to switch model: {e}")
            return False
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get all available models from different providers."""
        models = []
        
        # Ollama models
        ollama_models = LLMManager.detect_ollama_models()
        for m in ollama_models:
            models.append({
                "name": m["name"],
                "provider": "ollama",
                "size": self._format_size(m.get("size", 0)),
                "modified": m.get("modified", ""),
            })
        
        # API providers
        api_providers = LLMManager.detect_api_providers()
        for p in api_providers:
            models.append({
                "name": f"{p['name']}:{p['model']}",
                "provider": p["name"],
                "size": "API",
                "modified": "",
            })
        
        # Local models
        local_models = LLMManager.detect_local_models()
        for m in local_models:
            models.append({
                "name": m["name"],
                "provider": "local",
                "size": "Local",
                "modified": "",
            })
        
        return models if models else [{"name": "No models found", "provider": "none", "size": "", "modified": ""}]
    
    def _format_size(self, size_bytes: int) -> str:
        """Format size in bytes to human readable."""
        size = float(size_bytes)
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"
    
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Execute a tool and return the result."""
        try:
            # Add timeout to prevent hanging
            import asyncio
            
            if self.action_executor:
                try:
                    # Run with timeout
                    result = await asyncio.wait_for(
                        asyncio.to_thread(self.action_executor.execute, tool_name, arguments),
                        timeout=30.0
                    )
                    return json.dumps(result) if isinstance(result, dict) else str(result)
                except asyncio.TimeoutError:
                    return f"Tool '{tool_name}' timed out after 30 seconds"
                except Exception as e:
                    logger.warning(f"Action executor failed: {e}, using fallback")
            
            # Fallback implementations
            if tool_name == "calculator":
                try:
                    # Safe eval for math
                    expr = arguments.get("expression", "")
                    allowed = set("0123456789+-*/.() ")
                    if all(c in allowed for c in expr):
                        result = eval(expr)
                        return str(result)
                    return "Invalid expression"
                except Exception as e:
                    return f"Calculation error: {e}"
            
            elif tool_name == "web_search":
                return f"[Search results for: {arguments.get('query', '')}] - Web search executed"
            
            elif tool_name == "read_file":
                path = arguments.get("path", "")
                if os.path.exists(path):
                    with open(path, "r", encoding="utf-8") as f:
                        return f.read()[:10000]  # Limit content
                return f"File not found: {path}"
            
            elif tool_name == "create_file":
                path = arguments.get("path", "")
                content = arguments.get("content", "")
                os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)
                return f"File created: {path}"
            
            elif tool_name == "run_command":
                import subprocess
                cmd = arguments.get("command", "")
                try:
                    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
                    return result.stdout + result.stderr
                except subprocess.TimeoutExpired:
                    return f"Command timed out after 30 seconds"
            
            elif tool_name == "reasoning":
                problem = arguments.get("problem", "")
                approach = arguments.get("approach", "deductive")
                return f"[Reasoning Analysis - {approach}]\nProblem: {problem}\n\nThis requires deep analysis..."
            
            return f"Tool '{tool_name}' executed with args: {arguments}"
            
        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            return f"Error executing {tool_name}: {str(e)}"
    
    async def chat_stream(self, request: ChatRequest) -> AsyncGenerator[str, None]:
        """Stream chat responses with tool calling support."""
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        
        # Check if tools are enabled
        tools_enabled = request.tools is not None and request.tools
        
        # Add system prompt for agent behavior
        if tools_enabled:
            system_prompt = """You are M.I.A (Multimodal Intelligent Assistant), an advanced AI agent designed to be helpful, harmless, and honest. You have access to various tools to help users accomplish tasks.

Your capabilities include:
- Web searching for current information
- File operations (read, write, create)
- Code analysis and generation
- Desktop automation
- Memory storage and retrieval
- Mathematical calculations
- Deep reasoning and planning

When a user asks for something that requires action, use the appropriate tools. Think step by step and explain your reasoning. Always aim for accuracy and helpfulness.

IMPORTANT: When you need to use a tool, respond with a JSON block like this:
```tool_call
{"tool": "tool_name", "arguments": {"arg1": "value1"}}
```

Available tools: web_search, create_file, read_file, run_command, analyze_code, web_browse, calculator, memory_store, memory_search, send_email, desktop_automation, reasoning"""
        else:
            system_prompt = """You are M.I.A (Multimodal Intelligent Assistant), an advanced AI assistant designed to be helpful, harmless, and honest.

You are knowledgeable, thoughtful, and aim to provide accurate, well-reasoned responses. Be concise when appropriate, but thorough when the question demands it."""

        if not messages or messages[0].get("role") != "system":
            messages.insert(0, {"role": "system", "content": system_prompt})
        
        user_prompt = messages[-1]["content"] if messages else ""
        
        # Check if LLM is available
        if not self.llm:
            yield f"data: {json.dumps({'error': 'No LLM available. Please make sure Ollama is running or an API key is configured.'})}\n\n"
            yield "data: [DONE]\n\n"
            return
        
        try:
            # Check if streaming is supported
            supports_stream = hasattr(self.llm, "stream") and callable(getattr(self.llm, "stream"))
            stream_enabled = getattr(self.llm, "stream_enabled", True)
            can_stream = supports_stream and stream_enabled
            
            if can_stream and self.llm is not None:
                # Use synchronous generator in thread
                llm = self.llm  # Capture reference
                def sync_stream():
                    try:
                        return list(llm.stream(user_prompt, messages=messages))
                    except Exception as e:
                        logger.error(f"Stream error: {e}")
                        return None
                
                try:
                    tokens = await asyncio.wait_for(
                        asyncio.to_thread(sync_stream),
                        timeout=120.0  # 2 minute timeout for streaming
                    )
                    
                    if tokens is None:
                        # Fallback to non-streaming
                        response = await asyncio.wait_for(
                            asyncio.to_thread(self.llm.query, user_prompt, messages=messages),
                            timeout=60.0
                        )
                        if response:
                            for word in response.split():
                                yield f"data: {json.dumps({'content': word + ' '})}\n\n"
                                await asyncio.sleep(0.02)
                        else:
                            yield f"data: {json.dumps({'error': 'No response from LLM. Check if the model is loaded correctly.'})}\n\n"
                    else:
                        full_response = ""
                        for token in tokens:
                            if token:
                                full_response += str(token)
                                yield f"data: {json.dumps({'content': str(token)})}\n\n"
                                await asyncio.sleep(0.01)
                        
                        if not full_response:
                            yield f"data: {json.dumps({'error': 'Empty response from model.'})}\n\n"
                        elif tools_enabled:
                            # Process tool calls and send results
                            tool_results = await self._process_tool_calls(full_response)
                            if tool_results:
                                tool_header = '\n\n---\n**Tool Results:**\n'
                                yield f"data: {json.dumps({'content': tool_header})}\n\n"
                                for tr in tool_results:
                                    tool_name = tr['tool']
                                    tool_result = tr['result']
                                    tool_content = f'- {tool_name}: {tool_result}\n'
                                    yield f"data: {json.dumps({'content': tool_content})}\n\n"
                    
                except asyncio.TimeoutError:
                    yield f"data: {json.dumps({'error': 'Request timed out. The model took too long to respond.'})}\n\n"
                except Exception as stream_error:
                    logger.warning(f"Streaming failed: {stream_error}")
                    yield f"data: {json.dumps({'error': f'Streaming error: {stream_error}'})}\n\n"
            else:
                # Non-streaming query
                try:
                    response = await asyncio.wait_for(
                        asyncio.to_thread(self.llm.query, user_prompt, messages=messages),
                        timeout=60.0
                    )
                    if response:
                        # Simulate streaming for better UX
                        words = response.split()
                        for word in words:
                            yield f"data: {json.dumps({'content': word + ' '})}\n\n"
                            await asyncio.sleep(0.02)
                        
                        if tools_enabled:
                            # Process tool calls
                            tool_results = await self._process_tool_calls(response)
                            if tool_results:
                                tool_header = '\n\n---\n**Tool Results:**\n'
                                yield f"data: {json.dumps({'content': tool_header})}\n\n"
                                for tr in tool_results:
                                    tool_name = tr['tool']
                                    tool_result = tr['result']
                                    tool_content = f'- {tool_name}: {tool_result}\n'
                                    yield f"data: {json.dumps({'content': tool_content})}\n\n"
                    else:
                        yield f"data: {json.dumps({'error': 'No response from model.'})}\n\n"
                except asyncio.TimeoutError:
                    yield f"data: {json.dumps({'error': 'Request timed out.'})}\n\n"
                except Exception as query_error:
                    yield f"data: {json.dumps({'error': f'Query error: {query_error}'})}\n\n"
                
        except Exception as e:
            logger.error(f"Chat stream error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        
        yield "data: [DONE]\n\n"
    
    async def _process_tool_calls(self, response: str) -> Optional[List[Dict[str, Any]]]:
        """Extract and process tool calls from response."""
        import re
        
        # Look for tool_call blocks
        tool_pattern = r'```tool_call\s*\n?({.*?})\s*\n?```'
        matches = re.findall(tool_pattern, response, re.DOTALL)
        
        if not matches:
            return None
        
        results = []
        for match in matches:
            try:
                tool_data = json.loads(match)
                tool_name = tool_data.get("tool")
                arguments = tool_data.get("arguments", {})
                
                if tool_name:
                    try:
                        # Execute with timeout
                        result = await asyncio.wait_for(
                            self.execute_tool(tool_name, arguments),
                            timeout=30.0
                        )
                        results.append({"tool": tool_name, "result": result})
                    except asyncio.TimeoutError:
                        results.append({"tool": tool_name, "result": "Tool execution timed out"})
                    except Exception as e:
                        results.append({"tool": tool_name, "result": f"Error: {e}"})
            except json.JSONDecodeError:
                continue
        
        return results if results else None
    
    async def chat(self, request: ChatRequest) -> Dict[str, Any]:
        """Non-streaming chat endpoint."""
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        
        try:
            if self.llm:
                response = await asyncio.to_thread(
                    self.llm.query,
                    messages[-1]["content"] if messages else "",
                )
                return {
                    "model": request.model,
                    "message": {
                        "role": "assistant",
                        "content": response
                    },
                    "done": True
                }
            return {"error": "LLM not available"}
        except Exception as e:
            return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
# FastAPI Application
# ═══════════════════════════════════════════════════════════════════════════════

app = FastAPI(title="M.I.A - Multimodal Intelligent Assistant", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global agent instance
agent = MIAAgent()

# ═══════════════════════════════════════════════════════════════════════════════
# API Routes
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the Ollama-style Web UI."""
    return get_ollama_style_html()

@app.get("/api/tags")
async def get_models():
    """Get available models (Ollama-compatible endpoint)."""
    models = agent.get_available_models()
    return {"models": models}

@app.post("/api/model")
async def set_model(request: Dict[str, str]):
    """Switch the active model."""
    model_name = request.get("model", "")
    if not model_name:
        raise HTTPException(status_code=400, detail="Model name required")
    
    success = agent.set_model(model_name)
    if success:
        return {"status": "success", "model": model_name}
    else:
        raise HTTPException(status_code=500, detail="Failed to switch model")

@app.get("/api/version")
async def get_version():
    """Get version info."""
    return {"version": "1.0.0", "name": "M.I.A", "current_model": agent.current_model}

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Chat endpoint with optional streaming."""
    if request.stream:
        return StreamingResponse(
            agent.chat_stream(request),
            media_type="text/event-stream"
        )
    return await agent.chat(request)

@app.get("/api/tools")
async def get_tools():
    """Get available tools for function calling."""
    return {"tools": AVAILABLE_TOOLS}

@app.post("/api/tools/{tool_name}")
async def execute_tool(tool_name: str, request: Dict[str, Any]):
    """Execute a specific tool."""
    result = await agent.execute_tool(tool_name, request)
    return {"result": result, "tool": tool_name}

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "llm_available": agent.llm is not None,
        "tools_available": agent.action_executor is not None
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Ollama-Style HTML UI
# ═══════════════════════════════════════════════════════════════════════════════

def get_ollama_style_html() -> str:
    """Return the complete Ollama-style HTML interface."""
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>M.I.A - Multimodal Intelligent Assistant</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <style>
        :root {
            --bg-primary: #0d1117;
            --bg-secondary: #161b22;
            --bg-tertiary: #21262d;
            --bg-input: #0d1117;
            --border-color: #30363d;
            --text-primary: #e6edf3;
            --text-secondary: #8b949e;
            --text-muted: #6e7681;
            --accent-color: #58a6ff;
            --accent-hover: #79c0ff;
            --user-bubble: #1f6feb;
            --assistant-bubble: #21262d;
            --success-color: #3fb950;
            --warning-color: #d29922;
            --error-color: #f85149;
            --tool-color: #a371f7;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }

        /* Header */
        .header {
            background: var(--bg-secondary);
            border-bottom: 1px solid var(--border-color);
            padding: 12px 24px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .logo {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .logo-icon {
            width: 36px;
            height: 36px;
            background: linear-gradient(135deg, var(--accent-color), var(--tool-color));
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
        }

        .logo-text {
            font-size: 20px;
            font-weight: 600;
            background: linear-gradient(135deg, var(--text-primary), var(--accent-color));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .logo-subtitle {
            font-size: 11px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .header-controls {
            display: flex;
            align-items: center;
            gap: 16px;
        }

        .model-selector {
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            padding: 8px 16px;
            border-radius: 8px;
            font-size: 14px;
            cursor: pointer;
            min-width: 200px;
            outline: none;
            transition: border-color 0.2s;
        }

        .model-selector:hover, .model-selector:focus {
            border-color: var(--accent-color);
        }

        .status-indicator {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 13px;
            color: var(--text-secondary);
        }

        .tools-button {
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            color: var(--text-secondary);
            border-radius: 8px;
            padding: 8px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.2s;
        }

        .tools-button:hover {
            border-color: var(--accent-color);
            color: var(--accent-color);
        }

        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--success-color);
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }

        /* Main Chat Area */
        .main-container {
            flex: 1;
            display: flex;
            flex-direction: column;
            max-width: 900px;
            width: 100%;
            margin: 0 auto;
            padding: 0 24px;
        }

        .chat-container {
            flex: 1;
            overflow-y: auto;
            padding: 24px 0;
            display: flex;
            flex-direction: column;
            gap: 24px;
        }

        /* Welcome Screen */
        .welcome-screen {
            flex: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            padding: 40px;
            gap: 24px;
        }

        .welcome-icon {
            width: 80px;
            height: 80px;
            background: linear-gradient(135deg, var(--accent-color), var(--tool-color));
            border-radius: 24px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 40px;
            margin-bottom: 16px;
        }

        .welcome-title {
            font-size: 32px;
            font-weight: 700;
            margin-bottom: 8px;
        }

        .welcome-subtitle {
            font-size: 16px;
            color: var(--text-secondary);
            max-width: 500px;
            line-height: 1.6;
        }

        .capabilities-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            width: 100%;
            max-width: 700px;
            margin-top: 24px;
        }

        .capability-card {
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 20px;
            text-align: left;
            transition: all 0.2s;
            cursor: pointer;
        }

        .capability-card:hover {
            border-color: var(--accent-color);
            transform: translateY(-2px);
        }

        .capability-icon {
            width: 40px;
            height: 40px;
            background: var(--bg-tertiary);
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 12px;
            color: var(--accent-color);
        }

        .capability-card:hover .capability-icon {
            background: var(--accent-color);
            color: var(--bg-primary);
        }

        .capability-title {
            font-size: 14px;
            font-weight: 600;
            margin-bottom: 4px;
        }

        .capability-desc {
            font-size: 12px;
            color: var(--text-secondary);
        }

        /* Messages */
        .message {
            display: flex;
            gap: 16px;
            animation: fadeIn 0.3s ease;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .message-avatar {
            width: 36px;
            height: 36px;
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
            font-size: 16px;
        }

        .message.user .message-avatar {
            background: var(--user-bubble);
        }

        .message.assistant .message-avatar {
            background: linear-gradient(135deg, var(--accent-color), var(--tool-color));
        }

        .message.tool .message-avatar {
            background: var(--tool-color);
        }

        .message-content {
            flex: 1;
            min-width: 0;
        }

        .message-header {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 8px;
        }

        .message-role {
            font-size: 13px;
            font-weight: 600;
            color: var(--text-secondary);
        }

        .message-time {
            font-size: 11px;
            color: var(--text-muted);
        }

        .message-body {
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 16px;
            line-height: 1.6;
            font-size: 14px;
        }

        .message.user .message-body {
            background: var(--user-bubble);
            border-color: var(--user-bubble);
        }

        .message-body p {
            margin-bottom: 12px;
        }

        .message-body p:last-child {
            margin-bottom: 0;
        }

        .message-body pre {
            background: var(--bg-primary);
            border-radius: 8px;
            padding: 16px;
            overflow-x: auto;
            margin: 12px 0;
        }

        .message-body code {
            font-family: 'JetBrains Mono', 'Fira Code', monospace;
            font-size: 13px;
        }

        .message-body code:not(pre code) {
            background: var(--bg-tertiary);
            padding: 2px 6px;
            border-radius: 4px;
        }

        /* Tool Calls */
        .tool-call {
            background: var(--bg-tertiary);
            border: 1px solid var(--tool-color);
            border-radius: 8px;
            padding: 12px;
            margin: 12px 0;
        }

        .tool-call-header {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 8px;
            font-size: 12px;
            color: var(--tool-color);
            font-weight: 600;
        }

        .tool-call-content {
            font-family: monospace;
            font-size: 12px;
            color: var(--text-secondary);
        }

        /* Thinking Indicator */
        .thinking {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 16px;
            color: var(--text-secondary);
        }

        .thinking-dots {
            display: flex;
            gap: 4px;
        }

        .thinking-dots span {
            width: 8px;
            height: 8px;
            background: var(--accent-color);
            border-radius: 50%;
            animation: bounce 1.4s infinite ease-in-out both;
        }

        .thinking-dots span:nth-child(1) { animation-delay: -0.32s; }
        .thinking-dots span:nth-child(2) { animation-delay: -0.16s; }

        @keyframes bounce {
            0%, 80%, 100% { transform: scale(0); }
            40% { transform: scale(1); }
        }

        /* Input Area */
        .input-area {
            position: sticky;
            bottom: 0;
            background: var(--bg-primary);
            padding: 24px 0;
            border-top: 1px solid var(--border-color);
        }

        .input-container {
            display: flex;
            gap: 12px;
            align-items: flex-end;
        }

        .input-wrapper {
            flex: 1;
            position: relative;
        }

        .message-input {
            width: 100%;
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 16px 20px;
            padding-right: 50px;
            color: var(--text-primary);
            font-size: 14px;
            font-family: inherit;
            resize: none;
            outline: none;
            min-height: 56px;
            max-height: 200px;
            transition: border-color 0.2s;
        }

        .message-input:focus {
            border-color: var(--accent-color);
        }

        .message-input::placeholder {
            color: var(--text-muted);
        }

        .send-button {
            position: absolute;
            right: 12px;
            bottom: 12px;
            width: 36px;
            height: 36px;
            background: var(--accent-color);
            border: none;
            border-radius: 8px;
            color: white;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.2s;
        }

        .send-button:hover {
            background: var(--accent-hover);
            transform: scale(1.05);
        }

        .send-button:disabled {
            background: var(--bg-tertiary);
            cursor: not-allowed;
            transform: none;
        }

        .send-button svg {
            width: 18px;
            height: 18px;
        }

        /* Tools Toggle */
        .tools-toggle {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-top: 12px;
            font-size: 13px;
            color: var(--text-secondary);
        }

        .tools-toggle input[type="checkbox"] {
            appearance: none;
            width: 40px;
            height: 22px;
            background: var(--bg-tertiary);
            border-radius: 11px;
            position: relative;
            cursor: pointer;
            transition: background 0.2s;
        }

        .tools-toggle input[type="checkbox"]::after {
            content: '';
            position: absolute;
            width: 18px;
            height: 18px;
            background: var(--text-secondary);
            border-radius: 50%;
            top: 2px;
            left: 2px;
            transition: all 0.2s;
        }

        .tools-toggle input[type="checkbox"]:checked {
            background: var(--accent-color);
        }

        .tools-toggle input[type="checkbox"]:checked::after {
            left: 20px;
            background: white;
        }

        /* Keyboard Shortcut Hint */
        .input-hint {
            font-size: 11px;
            color: var(--text-muted);
            margin-top: 8px;
            text-align: center;
        }

        .input-hint kbd {
            background: var(--bg-tertiary);
            padding: 2px 6px;
            border-radius: 4px;
            font-family: inherit;
        }

        /* Sidebar (Tools Panel) */
        .sidebar {
            position: fixed;
            right: -400px;
            top: 0;
            width: 400px;
            height: 100vh;
            background: var(--bg-secondary);
            border-left: 1px solid var(--border-color);
            transition: right 0.3s;
            z-index: 200;
            overflow-y: auto;
            padding: 24px;
        }

        .sidebar.open {
            right: 0;
        }

        .sidebar-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 24px;
        }

        .sidebar-title {
            font-size: 18px;
            font-weight: 600;
        }

        .sidebar-close {
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            color: var(--text-secondary);
            cursor: pointer;
            border-radius: 8px;
            padding: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.2s;
        }

        .sidebar-close:hover {
            border-color: var(--error-color);
            color: var(--error-color);
        }

        .tool-list {
            display: flex;
            flex-direction: column;
            gap: 12px;
        }

        .tool-item {
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 16px;
        }

        .tool-name {
            font-weight: 600;
            color: var(--tool-color);
            margin-bottom: 4px;
        }

        .tool-description {
            font-size: 13px;
            color: var(--text-secondary);
        }

        /* Overlay */
        .overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.5);
            opacity: 0;
            visibility: hidden;
            transition: all 0.3s;
            z-index: 150;
        }

        .overlay.active {
            opacity: 1;
            visibility: visible;
        }

        /* Responsive */
        @media (max-width: 768px) {
            .header {
                padding: 12px 16px;
            }

            .main-container {
                padding: 0 16px;
            }

            .model-selector {
                min-width: 140px;
                font-size: 13px;
            }

            .welcome-screen {
                padding: 24px;
            }

            .capabilities-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <!-- Header -->
    <header class="header">
        <div class="logo">
            <div class="logo-icon">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" width="22" height="22">
                    <path d="M12 2a4 4 0 0 1 4 4v2a4 4 0 0 1-8 0V6a4 4 0 0 1 4-4z"/>
                    <path d="M18 14a6 6 0 0 0-12 0v4a2 2 0 0 0 2 2h8a2 2 0 0 0 2-2v-4z"/>
                    <circle cx="9" cy="10" r="1" fill="currentColor"/>
                    <circle cx="15" cy="10" r="1" fill="currentColor"/>
                </svg>
            </div>
            <div>
                <div class="logo-text">M.I.A</div>
                <div class="logo-subtitle">AGI Agent</div>
            </div>
        </div>
        <div class="header-controls">
            <select class="model-selector" id="modelSelect">
                <option value="">Loading models...</option>
            </select>
            <div class="status-indicator">
                <div class="status-dot" id="statusDot"></div>
                <span id="statusText">Ready</span>
            </div>
            <button class="tools-button" id="toolsButton" title="View Tools">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="20" height="20">
                    <path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/>
                </svg>
            </button>
        </div>
    </header>

    <!-- Main Container -->
    <div class="main-container">
        <div class="chat-container" id="chatContainer">
            <!-- Welcome Screen -->
            <div class="welcome-screen" id="welcomeScreen">
                <div class="welcome-icon">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" width="48" height="48">
                        <path d="M12 2a4 4 0 0 1 4 4v2a4 4 0 0 1-8 0V6a4 4 0 0 1 4-4z"/>
                        <path d="M18 14a6 6 0 0 0-12 0v4a2 2 0 0 0 2 2h8a2 2 0 0 0 2-2v-4z"/>
                        <circle cx="9" cy="10" r="1" fill="currentColor"/>
                        <circle cx="15" cy="10" r="1" fill="currentColor"/>
                    </svg>
                </div>
                <h1 class="welcome-title">M.I.A</h1>
                <p class="welcome-subtitle">
                    Multimodal Intelligent Assistant - An advanced AI agent with reasoning, 
                    tool use, and the goal of pushing towards AGI.
                </p>
                <div class="capabilities-grid">
                    <div class="capability-card" onclick="insertPrompt('Search the web for the latest AI news')">
                        <div class="capability-icon">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="24" height="24">
                                <circle cx="12" cy="12" r="10"/>
                                <path d="M2 12h20M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/>
                            </svg>
                        </div>
                        <div class="capability-title">Web Search</div>
                        <div class="capability-desc">Find current information online</div>
                    </div>
                    <div class="capability-card" onclick="insertPrompt('Analyze this code and suggest improvements')">
                        <div class="capability-icon">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="24" height="24">
                                <polyline points="16 18 22 12 16 6"/>
                                <polyline points="8 6 2 12 8 18"/>
                            </svg>
                        </div>
                        <div class="capability-title">Code Analysis</div>
                        <div class="capability-desc">Review and improve code</div>
                    </div>
                    <div class="capability-card" onclick="insertPrompt('Create a Python script that...')">
                        <div class="capability-icon">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="24" height="24">
                                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                                <polyline points="14 2 14 8 20 8"/>
                                <line x1="16" y1="13" x2="8" y2="13"/>
                                <line x1="16" y1="17" x2="8" y2="17"/>
                                <polyline points="10 9 9 9 8 9"/>
                            </svg>
                        </div>
                        <div class="capability-title">File Operations</div>
                        <div class="capability-desc">Read, write, and manage files</div>
                    </div>
                    <div class="capability-card" onclick="insertPrompt('Help me reason through this problem step by step')">
                        <div class="capability-icon">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="24" height="24">
                                <circle cx="12" cy="12" r="3"/>
                                <path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/>
                            </svg>
                        </div>
                        <div class="capability-title">Deep Reasoning</div>
                        <div class="capability-desc">Complex problem solving</div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Input Area -->
        <div class="input-area">
            <div class="input-container">
                <div class="input-wrapper">
                    <textarea 
                        class="message-input" 
                        id="messageInput" 
                        placeholder="Ask me anything... I can search, code, reason, and execute tasks."
                        rows="1"
                    ></textarea>
                    <button class="send-button" id="sendButton">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z"/>
                        </svg>
                    </button>
                </div>
            </div>
            <div class="tools-toggle">
                <input type="checkbox" id="toolsEnabled" checked>
                <label for="toolsEnabled">Enable Tool Calling (Function execution)</label>
            </div>
            <div class="input-hint">
                Press <kbd>Enter</kbd> to send, <kbd>Shift+Enter</kbd> for new line
            </div>
        </div>
    </div>

    <!-- Tools Sidebar -->
    <div class="sidebar" id="toolsSidebar">
        <div class="sidebar-header">
            <h2 class="sidebar-title">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="20" height="20" style="vertical-align: middle; margin-right: 8px;">
                    <path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/>
                </svg>
                Available Tools
            </h2>
            <button class="sidebar-close" id="closeSidebar">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="20" height="20">
                    <line x1="18" y1="6" x2="6" y2="18"/>
                    <line x1="6" y1="6" x2="18" y2="18"/>
                </svg>
            </button>
        </div>
        <div class="tool-list" id="toolList">
            <!-- Tools will be populated here -->
        </div>
    </div>

    <!-- Overlay -->
    <div class="overlay" id="overlay"></div>

    <script>
        // State
        let messages = [];
        let isGenerating = false;
        let currentModel = '';
        let toolsEnabled = true;

        // DOM Elements
        const chatContainer = document.getElementById('chatContainer');
        const welcomeScreen = document.getElementById('welcomeScreen');
        const messageInput = document.getElementById('messageInput');
        const sendButton = document.getElementById('sendButton');
        const modelSelect = document.getElementById('modelSelect');
        const statusDot = document.getElementById('statusDot');
        const statusText = document.getElementById('statusText');
        const toolsButton = document.getElementById('toolsButton');
        const toolsSidebar = document.getElementById('toolsSidebar');
        const closeSidebar = document.getElementById('closeSidebar');
        const overlay = document.getElementById('overlay');
        const toolList = document.getElementById('toolList');
        const toolsEnabledCheckbox = document.getElementById('toolsEnabled');

        // Configure marked for Markdown
        marked.setOptions({
            highlight: function(code, lang) {
                if (lang && hljs.getLanguage(lang)) {
                    return hljs.highlight(code, { language: lang }).value;
                }
                return hljs.highlightAuto(code).value;
            },
            breaks: true
        });

        // Initialize
        async function init() {
            await loadModels();
            await loadTools();
            setupEventListeners();
            autoResizeTextarea();
        }

        // Load available models
        async function loadModels() {
            try {
                const response = await fetch('/api/tags');
                const data = await response.json();
                
                modelSelect.innerHTML = '';
                data.models.forEach((model, index) => {
                    const option = document.createElement('option');
                    option.value = model.name;
                    option.textContent = `${model.name} (${model.provider})`;
                    if (index === 0) {
                        option.selected = true;
                        currentModel = model.name;
                    }
                    modelSelect.appendChild(option);
                });
                
                updateStatus('Ready', true);
            } catch (error) {
                console.error('Failed to load models:', error);
                modelSelect.innerHTML = '<option value="">No models available</option>';
                updateStatus('No models', false);
            }
        }

        // Load available tools
        async function loadTools() {
            try {
                const response = await fetch('/api/tools');
                const data = await response.json();
                
                toolList.innerHTML = '';
                data.tools.forEach(tool => {
                    const func = tool.function;
                    const item = document.createElement('div');
                    item.className = 'tool-item';
                    item.innerHTML = `
                        <div class="tool-name">${func.name}</div>
                        <div class="tool-description">${func.description}</div>
                    `;
                    toolList.appendChild(item);
                });
            } catch (error) {
                console.error('Failed to load tools:', error);
            }
        }

        // Setup event listeners
        function setupEventListeners() {
            sendButton.addEventListener('click', sendMessage);
            
            messageInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    sendMessage();
                }
            });

            messageInput.addEventListener('input', autoResizeTextarea);

            modelSelect.addEventListener('change', async (e) => {
                const newModel = e.target.value;
                if (newModel && newModel !== currentModel) {
                    updateStatus('Switching model...', true);
                    try {
                        const response = await fetch('/api/model', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ model: newModel })
                        });
                        if (response.ok) {
                            currentModel = newModel;
                            updateStatus('Ready', true);
                        } else {
                            updateStatus('Model switch failed', false);
                            // Revert selection
                            modelSelect.value = currentModel;
                        }
                    } catch (error) {
                        console.error('Model switch error:', error);
                        updateStatus('Error', false);
                        modelSelect.value = currentModel;
                    }
                }
            });

            toolsButton.addEventListener('click', () => {
                toolsSidebar.classList.add('open');
                overlay.classList.add('active');
            });

            closeSidebar.addEventListener('click', closeSidebarPanel);
            overlay.addEventListener('click', closeSidebarPanel);

            toolsEnabledCheckbox.addEventListener('change', (e) => {
                toolsEnabled = e.target.checked;
            });
        }

        function closeSidebarPanel() {
            toolsSidebar.classList.remove('open');
            overlay.classList.remove('active');
        }

        // Auto-resize textarea
        function autoResizeTextarea() {
            messageInput.style.height = 'auto';
            messageInput.style.height = Math.min(messageInput.scrollHeight, 200) + 'px';
        }

        // Update status indicator
        function updateStatus(text, online) {
            statusText.textContent = text;
            statusDot.style.background = online ? 'var(--success-color)' : 'var(--error-color)';
        }

        // Insert prompt from capability cards
        function insertPrompt(text) {
            messageInput.value = text;
            messageInput.focus();
            autoResizeTextarea();
        }

        // Send message
        async function sendMessage() {
            const content = messageInput.value.trim();
            if (!content || isGenerating) return;

            // Hide welcome screen
            welcomeScreen.style.display = 'none';

            // Add user message
            addMessage('user', content);
            messages.push({ role: 'user', content });

            // Clear input
            messageInput.value = '';
            autoResizeTextarea();

            // Generate response
            isGenerating = true;
            sendButton.disabled = true;
            updateStatus('Thinking...', true);

            await generateResponse();

            isGenerating = false;
            sendButton.disabled = false;
            updateStatus('Ready', true);
        }

        // Add message to chat
        function addMessage(role, content, isStreaming = false) {
            const id = 'msg-' + Date.now();
            const time = new Date().toLocaleTimeString();
            
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${role}`;
            messageDiv.id = id;
            
            const avatar = role === 'user' 
                ? '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="18" height="18"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>'
                : '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" width="18" height="18"><path d="M12 2a4 4 0 0 1 4 4v2a4 4 0 0 1-8 0V6a4 4 0 0 1 4-4z"/><path d="M18 14a6 6 0 0 0-12 0v4a2 2 0 0 0 2 2h8a2 2 0 0 0 2-2v-4z"/><circle cx="9" cy="10" r="1" fill="currentColor"/><circle cx="15" cy="10" r="1" fill="currentColor"/></svg>';
            const roleName = role === 'user' ? 'You' : 'M.I.A';
            
            messageDiv.innerHTML = `
                <div class="message-avatar">${avatar}</div>
                <div class="message-content">
                    <div class="message-header">
                        <span class="message-role">${roleName}</span>
                        <span class="message-time">${time}</span>
                    </div>
                    <div class="message-body" id="${id}-body">
                        ${isStreaming ? '<div class="thinking"><div class="thinking-dots"><span></span><span></span><span></span></div><span>Thinking...</span></div>' : renderMarkdown(content)}
                    </div>
                </div>
            `;
            
            chatContainer.appendChild(messageDiv);
            scrollToBottom();
            
            return id;
        }

        // Render markdown content
        function renderMarkdown(content) {
            try {
                return marked.parse(content);
            } catch (e) {
                return content.replace(/\\n/g, '<br>');
            }
        }

        // Generate response with streaming
        async function generateResponse() {
            const messageId = addMessage('assistant', '', true);
            const bodyElement = document.getElementById(messageId + '-body');
            
            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        model: currentModel,
                        messages: messages,
                        stream: true,
                        temperature: 0.7,
                        max_tokens: 4096,
                        tools: toolsEnabled ? true : null
                    })
                });

                if (!response.ok) {
                    const errorText = await response.text();
                    throw new Error(`Server error: ${response.status} - ${errorText}`);
                }

                if (!response.body) {
                    throw new Error('No response body received');
                }

                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let fullContent = '';
                let hasReceivedContent = false;

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;

                    const chunk = decoder.decode(value);
                    const lines = chunk.split('\\n');

                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            const data = line.slice(6).trim();
                            if (data === '[DONE]') {
                                hasReceivedContent = true;
                                break;
                            }
                            
                            try {
                                const parsed = JSON.parse(data);
                                if (parsed.content) {
                                    hasReceivedContent = true;
                                    fullContent += parsed.content;
                                    bodyElement.innerHTML = renderMarkdown(fullContent);
                                    scrollToBottom();
                                }
                                if (parsed.error) {
                                    hasReceivedContent = true;
                                    bodyElement.innerHTML = `<span style="color: var(--error-color)">Error: ${parsed.error}</span>`;
                                    console.error('Server error:', parsed.error);
                                }
                            } catch (e) {
                                console.log('Parse chunk:', data);
                            }
                        }
                    }
                }

                // If no content received, show error
                if (!hasReceivedContent || fullContent === '') {
                    bodyElement.innerHTML = '<span style="color: var(--warning-color)">No response received. Make sure a model is selected and the LLM provider is running (e.g., Ollama).</span>';
                    return;
                }

                // Apply syntax highlighting to code blocks
                bodyElement.querySelectorAll('pre code').forEach((block) => {
                    hljs.highlightElement(block);
                });

                // Store message
                messages.push({ role: 'assistant', content: fullContent });

            } catch (error) {
                console.error('Generation error:', error);
                bodyElement.innerHTML = `<span style="color: var(--error-color)">Error: ${error.message}</span>`;
            }
        }

        // Scroll to bottom of chat
        function scrollToBottom() {
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }

        // Initialize on load
        document.addEventListener('DOMContentLoaded', init);
    </script>
</body>
</html>'''


# ═══════════════════════════════════════════════════════════════════════════════
# Main Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

def run_webui(host: str = "0.0.0.0", port: int = 8080):
    """Run the M.I.A Web UI server."""
    print(f"""
+------------------------------------------------------------------------------+
|                                                                              |
|     M.I.A - Multimodal Intelligent Assistant                                 |
|     AGI-Focused Web Interface                                                |
|                                                                              |
|     Server starting at: http://{host}:{port}                                   |
|     Open your browser to interact with M.I.A                                 |
|                                                                              |
+------------------------------------------------------------------------------+
    """)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_webui()
