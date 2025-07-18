"""
LLM Manager: Unified interface for multiple LLM APIs (OpenAI, HuggingFace, Local, etc.)
"""
import os
import requests
import logging
from typing import Optional, Dict, Any, Union

# Import custom exceptions and error handling
from ..exceptions import LLMProviderError, NetworkError, ConfigurationError, InitializationError
from ..error_handler import global_error_handler, with_error_handling, safe_execute
from ..config_manager import ConfigManager

# Optional imports with error handling
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    OpenAI = None
    HAS_OPENAI = False

try:
    import warnings
    # Suppress transformers warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        from transformers.pipelines import pipeline
        from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    pipeline = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
    HAS_TRANSFORMERS = False

logger = logging.getLogger(__name__)

class LLMManager:
    """Unified LLM manager supporting multiple providers."""
    
    def __init__(self, provider=None, model_id=None, api_key=None, url=None, local_model_path=None, config_manager=None, **kwargs):
        # Initialize configuration manager
        self.config_manager = config_manager or ConfigManager()
        
        # Use configuration values if not provided (with safe access)
        config = self.config_manager.config
        if config and hasattr(config, 'llm'):
            self.provider = provider or config.llm.provider
            self.model_id = model_id or config.llm.model_id
            self.api_key = api_key or config.llm.api_key or os.getenv('OPENAI_API_KEY')
            self.url = url or config.llm.url
            self.max_tokens = config.llm.max_tokens
            self.temperature = config.llm.temperature
            self.timeout = config.llm.timeout
        else:
            # Default values if config is not available
            self.provider = provider or 'ollama'
            self.model_id = model_id or 'deepseek-r1:1.5b'
            self.api_key = api_key or os.getenv('OPENAI_API_KEY')
            self.url = url or 'http://localhost:11434'
            self.max_tokens = 2048
            self.temperature = 0.7
            self.timeout = 30
        
        self.local_model_path = local_model_path
        
        self.client: Optional[Union[Any, object]] = None
        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None
        
        self._initialize_provider()
    
    def _initialize_provider(self):
        """Initialize the specific provider with comprehensive error handling."""
        try:
            if self.provider == 'openai':
                self._initialize_openai()
            elif self.provider == 'huggingface':
                self._initialize_huggingface()
            elif self.provider == 'local':
                self._initialize_local()
            elif self.provider in ['grok', 'gemini', 'ollama', 'groq', 'anthropic']:
                self._initialize_api_provider()
            else:
                raise ConfigurationError(f"Unknown provider: {self.provider}", "UNKNOWN_PROVIDER")
                
        except Exception as e:
            error_context = {
                'provider': self.provider,
                'model_id': self.model_id,
                'has_api_key': bool(self.api_key)
            }
            
            # Convert to appropriate M.I.A exception
            if isinstance(e, (ConfigurationError, InitializationError)):
                raise e
            else:
                raise InitializationError(f"Failed to initialize provider {self.provider}: {str(e)}", 
                                        "PROVIDER_INIT_FAILED", error_context)
    
    def _initialize_openai(self):
        """Initialize OpenAI provider with specific error handling."""
        if not HAS_OPENAI:
            raise InitializationError("OpenAI package not installed. Run: pip install openai", 
                                    "MISSING_DEPENDENCY")
        if OpenAI is None:
            raise InitializationError("OpenAI class not available", "IMPORT_ERROR")
        
        api_key = self.api_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ConfigurationError("OpenAI API key not provided", "MISSING_API_KEY")
            
        try:
            self.client = OpenAI(
                base_url=self.url,
                api_key=api_key
            )
        except Exception as e:
            raise InitializationError(f"OpenAI client initialization failed: {str(e)}", 
                                    "CLIENT_INIT_FAILED")
    
    def _initialize_huggingface(self):
        """Initialize HuggingFace provider with specific error handling."""
        if not HAS_TRANSFORMERS:
            raise InitializationError("Transformers package not installed. Run: pip install transformers", 
                                    "MISSING_DEPENDENCY")
        if pipeline is None:
            raise InitializationError("Transformers pipeline not available", "IMPORT_ERROR")
        
        if not self.model_id:
            raise ConfigurationError("Model ID required for HuggingFace provider", "MISSING_MODEL_ID")
            
        try:
            self.client = pipeline('text-generation', model=self.model_id)
        except Exception as e:
            raise InitializationError(f"HuggingFace pipeline initialization failed: {str(e)}", 
                                    "PIPELINE_INIT_FAILED")
    
    def _initialize_local(self):
        """Initialize local model provider with specific error handling."""
        if not HAS_TRANSFORMERS:
            raise InitializationError("Transformers package not installed. Run: pip install transformers", 
                                    "MISSING_DEPENDENCY")
        if AutoTokenizer is None or AutoModelForCausalLM is None:
            raise InitializationError("Transformers models not available", "IMPORT_ERROR")
        
        model_path = self.local_model_path or self.model_id
        if model_path is None:
            raise ConfigurationError("Model path must be specified for local provider", 
                                   "MISSING_MODEL_PATH")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(model_path)
        except Exception as e:
            raise InitializationError(f"Local model initialization failed: {str(e)}", 
                                    "MODEL_LOAD_FAILED")
    
    def _initialize_api_provider(self):
        """Initialize API-based providers."""
        # API-based providers don't need special initialization
        # but we can validate configuration here
        if self.provider == 'ollama':
            # Validate Ollama configuration
            if not self.url and not os.getenv('OLLAMA_URL'):
                self.url = 'http://localhost:11434/api/generate'
        
        logger.info(f"Initialized API provider: {self.provider}")

    def query_model(self, prompt: str, **kwargs) -> Optional[str]:
        """Alias for compatibility with main.py and other modules."""
        return self.query(prompt, **kwargs)

    @with_error_handling(global_error_handler, fallback_value=None)
    def query(self, prompt: str, **kwargs) -> Optional[str]:
        """Query the selected LLM provider with comprehensive error handling."""
        if not prompt:
            raise ValueError("Empty prompt provided")
            
        if not prompt.strip():
            raise ValueError("Prompt contains only whitespace")
            
        try:
            if self.provider == 'openai':
                return self._query_openai(prompt, **kwargs)
            elif self.provider == 'anthropic':
                return self._query_anthropic(prompt, **kwargs)
            elif self.provider == 'gemini':
                return self._query_gemini(prompt, **kwargs)
            elif self.provider == 'ollama':
                return self._query_ollama(prompt, **kwargs)
            elif self.provider == 'groq':
                return self._query_groq(prompt, **kwargs)
            elif self.provider == 'grok':
                return self._query_grok(prompt, **kwargs)
            elif self.provider == 'huggingface':
                return self._query_huggingface(prompt, **kwargs)
            elif self.provider == 'local':
                return self._query_local(prompt, **kwargs)
            else:
                raise LLMProviderError(f"Provider {self.provider} not implemented", 
                                     "PROVIDER_NOT_IMPLEMENTED")
                
        except Exception as e:
            # Re-raise M.I.A exceptions
            if isinstance(e, (LLMProviderError, NetworkError, ConfigurationError)):
                raise e
            # Convert other exceptions to LLMProviderError
            raise LLMProviderError(f"Query failed: {str(e)}", "QUERY_FAILED", {
                'provider': self.provider,
                'prompt_length': len(prompt),
                'kwargs': kwargs
            })
    
    def _query_openai(self, prompt: str, **kwargs) -> Optional[str]:
        """Query OpenAI with specific error handling."""
        if self.client is None or not HAS_OPENAI:
            raise LLMProviderError("OpenAI client not available", "CLIENT_NOT_AVAILABLE")
            
        try:
            # Use getattr for safer attribute access
            chat_attr = getattr(self.client, 'chat', None)
            if chat_attr is None:
                raise LLMProviderError("OpenAI client missing chat attribute", "CLIENT_MALFORMED")
                
            completions_attr = getattr(chat_attr, 'completions', None)
            if completions_attr is None:
                raise LLMProviderError("OpenAI client missing completions attribute", "CLIENT_MALFORMED")
                
            response = completions_attr.create(
                model=self.model_id or 'gpt-3.5-turbo',
                messages=[{"role": "user", "content": prompt}],
                max_tokens=kwargs.get('max_tokens', 1024),
                temperature=kwargs.get('temperature', 0.7)
            )
            
            if not response.choices:
                raise LLMProviderError("OpenAI returned no choices", "EMPTY_RESPONSE")
                
            content = response.choices[0].message.content
            if content is None:
                raise LLMProviderError("OpenAI returned null content", "NULL_CONTENT")
                
            return content
            
        except Exception as e:
            if isinstance(e, LLMProviderError):
                raise e
            raise LLMProviderError(f"OpenAI API error: {str(e)}", "API_ERROR")
    
    def _query_huggingface(self, prompt: str, **kwargs) -> Optional[str]:
        """Query HuggingFace with specific error handling."""
        if self.client is None or not HAS_TRANSFORMERS:
            raise LLMProviderError("HuggingFace client not available", "CLIENT_NOT_AVAILABLE")
            
        try:
            if not callable(self.client):
                raise LLMProviderError("HuggingFace pipeline not callable", "CLIENT_NOT_CALLABLE")
                
            result = self.client(prompt, max_length=kwargs.get('max_length', 256))
            
            if not isinstance(result, list) or len(result) == 0:
                raise LLMProviderError("HuggingFace returned invalid result format", "INVALID_RESPONSE")
                
            if 'generated_text' not in result[0]:
                raise LLMProviderError("HuggingFace result missing generated_text", "MISSING_TEXT")
                
            return result[0]['generated_text']
            
        except Exception as e:
            if isinstance(e, LLMProviderError):
                raise e
            raise LLMProviderError(f"HuggingFace error: {str(e)}", "PIPELINE_ERROR")
    
    def _query_local(self, prompt: str, **kwargs) -> Optional[str]:
        """Query local model with specific error handling."""
        if self.model is None or self.tokenizer is None or not HAS_TRANSFORMERS:
            raise LLMProviderError("Local model not available", "MODEL_NOT_AVAILABLE")
            
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.model.generate(**inputs, max_length=kwargs.get('max_length', 256))
            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            if not result:
                raise LLMProviderError("Local model returned empty result", "EMPTY_RESULT")
                
            return result
            
        except Exception as e:
            if isinstance(e, LLMProviderError):
                raise e
            raise LLMProviderError(f"Local model error: {str(e)}", "MODEL_ERROR")

    def _query_anthropic(self, prompt: str, **kwargs) -> Optional[str]:
        """Query Anthropic Claude API."""
        try:
            headers = {
                'x-api-key': self.api_key or os.getenv('ANTHROPIC_API_KEY'),
                'anthropic-version': '2023-06-01',
                'content-type': 'application/json'
            }
            data = {
                'model': self.model_id or 'claude-3-opus-20240229',
                'max_tokens': kwargs.get('max_tokens', 1024),
                'messages': [{"role": "user", "content": prompt}]
            }
            url = self.url or 'https://api.anthropic.com/v1/messages'
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            
            if 'content' in result and isinstance(result['content'], list) and result['content']:
                return result['content'][0].get('text', '')
            return ''
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            return None

    def _query_gemini(self, prompt: str, **kwargs) -> Optional[str]:
        """Query Google Gemini API."""
        try:
            headers = {
                'Content-Type': 'application/json',
                'x-goog-api-key': self.api_key or os.getenv('GEMINI_API_KEY')
            }
            data = {
                'contents': [{"parts": [{"text": prompt}]}]
            }
            url = self.url or 'https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent'
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            
            candidates = result.get('candidates', [])
            if candidates and 'content' in candidates[0]:
                parts = candidates[0]['content'].get('parts', [])
                if parts and 'text' in parts[0]:
                    return parts[0]['text']
            return ''
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return None

    def _query_ollama(self, prompt: str, **kwargs) -> Optional[str]:
        """Query Ollama local API with enhanced error handling."""
        try:
            headers = {'Content-Type': 'application/json'}
            data = {
                'model': self.model_id or 'mistral:instruct',
                'prompt': prompt,
                'stream': False
            }
            
            if self.api_key and self.api_key != 'ollama':
                headers['Authorization'] = f'Bearer {self.api_key}'
                
            url = self.url or 'http://localhost:11434/api/generate'
            
            try:
                response = requests.post(url, headers=headers, json=data, timeout=30)
                response.raise_for_status()
            except requests.exceptions.Timeout:
                raise NetworkError("Ollama request timed out", "TIMEOUT")
            except requests.exceptions.ConnectionError:
                raise NetworkError("Cannot connect to Ollama. Make sure it's running: ollama serve", 
                                 "CONNECTION_ERROR")
            except requests.exceptions.HTTPError as e:
                raise NetworkError(f"Ollama HTTP error: {e}", "HTTP_ERROR")
            
            try:
                result = response.json()
            except ValueError:
                raise LLMProviderError("Ollama returned invalid JSON", "INVALID_JSON")
                
            if 'response' not in result:
                raise LLMProviderError("Ollama response missing 'response' field", "MISSING_RESPONSE")
                
            return result['response']
            
        except (NetworkError, LLMProviderError):
            raise
        except Exception as e:
            raise LLMProviderError(f"Ollama API error: {str(e)}", "API_ERROR")

    def _query_groq(self, prompt: str, **kwargs) -> Optional[str]:
        """Query Groq API."""
        try:
            headers = {
                'Authorization': f'Bearer {self.api_key or os.getenv("GROQ_API_KEY")}',
                'Content-Type': 'application/json'
            }
            data = {
                'model': self.model_id or 'llama2-70b-4096',
                'messages': [{"role": "user", "content": prompt}],
                'max_tokens': kwargs.get('max_tokens', 1024)
            }
            url = self.url or 'https://api.groq.com/openai/v1/chat/completions'
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content']
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            return None

    def _query_grok(self, prompt: str, **kwargs) -> Optional[str]:
        """Query xAI Grok API."""
        try:
            headers = {
                'Authorization': f'Bearer {self.api_key or os.getenv("GROK_API_KEY")}',
                'Content-Type': 'application/json'
            }
            data = {
                'model': self.model_id or 'grok-1',
                'messages': [{"role": "user", "content": prompt}],
                'max_tokens': kwargs.get('max_tokens', 1024)
            }
            url = self.url or 'https://api.grok.com/v1/chat/completions'
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content']
        except Exception as e:
            logger.error(f"Grok API error: {e}")
            return None

    def is_available(self) -> bool:
        """Check if the LLM provider is available and working."""
        try:
            if self.provider == 'openai' and self.client is not None and HAS_OPENAI:
                # Type-safe OpenAI client checking
                try:
                    chat_attr = getattr(self.client, 'chat', None)
                    if chat_attr is not None:
                        completions_attr = getattr(chat_attr, 'completions', None)
                        if completions_attr is not None:
                            response = completions_attr.create(
                                model=self.model_id or 'gpt-3.5-turbo',
                                messages=[{"role": "user", "content": "Hello"}],
                                max_tokens=10
                            )
                            return response.choices[0].message.content is not None
                    return False
                except Exception:
                    return False
            
            elif self.provider == 'ollama':
                # Test Ollama connection
                url = self.url or 'http://localhost:11434/api/generate'
                headers = {'Content-Type': 'application/json'}
                data = {
                    'model': self.model_id or 'mistral:instruct',
                    'prompt': 'Hello',
                    'stream': False
                }
                response = requests.post(url, headers=headers, json=data, timeout=5)
                return response.status_code == 200
            
            elif self.provider == 'huggingface' and self.client is not None and HAS_TRANSFORMERS:
                return callable(self.client)
            
            elif self.provider == 'local' and self.model is not None and self.tokenizer is not None and HAS_TRANSFORMERS:
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Error checking availability for {self.provider}: {e}")
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model configuration."""
        return {
            'provider': self.provider,
            'model_id': self.model_id,
            'available': self.is_available(),
            'url': self.url,
            'has_api_key': bool(self.api_key)
        }
