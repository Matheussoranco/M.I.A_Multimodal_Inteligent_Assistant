"""
LLM Manager: Unified interface for multiple LLM APIs (OpenAI, HuggingFace, Local, etc.)
"""
import os
import requests
import logging
from typing import Optional, Dict, Any, Union

# Import custom exceptions and error handling
from ..exceptions import LLMProviderError, NetworkError, ConfigurationError, InitializationError
from ..error_handler import global_error_handler, with_error_handling
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

# Async imports
try:
    import aiohttp
    import asyncio
    HAS_AIOHTTP = True
except ImportError:
    aiohttp = None
    asyncio = None
    HAS_AIOHTTP = False

logger = logging.getLogger(__name__)

class LLMManager:
    """Unified LLM manager supporting multiple providers."""
    
    @classmethod
    def detect_available_providers(cls, interactive: bool = True) -> Dict[str, Any]:
        """
        Detect available LLM providers based on API keys and test connectivity.
        
        Args:
            interactive: Whether to allow user selection if multiple providers are available
            
        Returns:
            Dict with 'provider', 'model_id', 'api_key', 'url' for the selected provider
        """
        available_providers = []
        
        # Define providers and their environment variables
        provider_configs = {
            'openai': {
                'env_vars': ['OPENAI_API_KEY'],
                'default_model': 'gpt-4.1',
                'url': 'https://api.openai.com/v1'
            },
            'anthropic': {
                'env_vars': ['ANTHROPIC_API_KEY'],
                'default_model': 'claude-3-haiku-20240307',
                'url': 'https://api.anthropic.com'
            },
            'gemini': {
                'env_vars': ['GOOGLE_API_KEY'],
                'default_model': 'gemini-pro',
                'url': 'https://generativelanguage.googleapis.com'
            },
            'groq': {
                'env_vars': ['GROQ_API_KEY'],
                'default_model': 'llama2-70b-4096',
                'url': 'https://api.groq.com'
            },
            'grok': {
                'env_vars': ['XAI_API_KEY'],
                'default_model': 'grok-beta',
                'url': 'https://api.x.ai'
            },
            'ollama': {
                'env_vars': [],  # Ollama doesn't need API key
                'default_model': 'deepseek-r1:1.5b',
                'url': 'http://localhost:11434/api/generate'
            }
        }
        
        print("🔍 Detecting available LLM providers...")
        
        for provider_name, config in provider_configs.items():
            try:
                # Check if required environment variables are set
                has_keys = all(os.getenv(var) for var in config['env_vars']) if config['env_vars'] else True
                
                if not has_keys:
                    print(f"❌ {provider_name}: No API key found")
                    continue
                
                # Test connectivity
                if cls._test_provider_connectivity(provider_name, config):
                    available_providers.append({
                        'name': provider_name,
                        'model': config['default_model'],
                        'url': config['url'],
                        'api_key': os.getenv(config['env_vars'][0]) if config['env_vars'] else None
                    })
                    print(f"✅ {provider_name}: Available")
                else:
                    print(f"❌ {provider_name}: Connection test failed")
                    
            except Exception as e:
                print(f"❌ {provider_name}: Error - {str(e)}")
        
        if not available_providers:
            raise ConfigurationError("No LLM providers available. Please set API keys for at least one provider.", "NO_PROVIDERS_AVAILABLE")
        
        # If only one provider, use it
        if len(available_providers) == 1:
            selected = available_providers[0]
            print(f"📌 Using {selected['name']} (only available provider)")
            return {
                'provider': selected['name'],
                'model_id': selected['model'],
                'api_key': selected['api_key'],
                'url': selected['url']
            }
        
        # Multiple providers available
        if interactive:
            print(f"\n📋 Found {len(available_providers)} available providers:")
            for i, provider in enumerate(available_providers, 1):
                print(f"  {i}. {provider['name']} ({provider['model']})")
            
            while True:
                try:
                    choice = input("\nSelect provider (1-{}): ".format(len(available_providers))).strip()
                    idx = int(choice) - 1
                    if 0 <= idx < len(available_providers):
                        selected = available_providers[idx]
                        print(f"📌 Selected: {selected['name']}")
                        return {
                            'provider': selected['name'],
                            'model_id': selected['model'],
                            'api_key': selected['api_key'],
                            'url': selected['url']
                        }
                    else:
                        print("Invalid choice. Please select a valid number.")
                except (ValueError, KeyboardInterrupt):
                    print("Invalid input. Please enter a number.")
        else:
            # Non-interactive mode: prefer OpenAI, then others
            preferred_order = ['openai', 'anthropic', 'gemini', 'groq', 'grok', 'ollama']
            for pref in preferred_order:
                for provider in available_providers:
                    if provider['name'] == pref:
                        print(f"📌 Using {provider['name']} (preferred)")
                        return {
                            'provider': provider['name'],
                            'model_id': provider['model'],
                            'api_key': provider['api_key'],
                            'url': provider['url']
                        }
            
            # Fallback to first available
            selected = available_providers[0]
            print(f"📌 Using {selected['name']} (fallback)")
            return {
                'provider': selected['name'],
                'model_id': selected['model'],
                'api_key': selected['api_key'],
                'url': selected['url']
            }
    
    @classmethod
    def _test_provider_connectivity(cls, provider_name: str, config: Dict[str, Any]) -> bool:
        """Test connectivity to a provider."""
        try:
            if provider_name == 'openai' and HAS_OPENAI and OpenAI:
                # For OpenAI, just check if API key is set and looks valid
                api_key = os.getenv('OPENAI_API_KEY')
                if api_key and api_key.startswith('sk-') and len(api_key) > 20:
                    # Optional: Try a minimal request to verify
                    try:
                        client = OpenAI(
                            api_key=api_key,
                            max_retries=1,
                            timeout=5.0
                        )
                        # Try a minimal request
                        response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[{"role": "user", "content": "test"}],
                            max_tokens=1
                        )
                        return True
                    except Exception:
                        # If API call fails, still consider it available if key looks valid
                        return True
                return False
                
            elif provider_name == 'ollama':
                # Test Ollama connectivity
                import requests
                response = requests.get('http://localhost:11434/api/tags', timeout=5)
                return response.status_code == 200
                
            elif provider_name in ['anthropic', 'gemini', 'groq', 'grok']:
                # For other providers, just check if API key is set
                env_vars = config.get('env_vars', [])
                return all(os.getenv(var) for var in env_vars)
                
            else:
                return False
                
        except Exception:
            return False
    
    def __init__(self, provider: Optional[str] = None, model_id: Optional[str] = None, api_key: Optional[str] = None, url: Optional[str] = None, local_model_path: Optional[str] = None, config_manager: Optional[Any] = None, auto_detect: bool = True, **kwargs: Any) -> None:
        # Initialize configuration manager
        self.config_manager = config_manager or ConfigManager()
        
        # Auto-detect provider if not specified and not in testing mode
        import sys
        is_testing = 'pytest' in sys.modules or os.getenv('TESTING') == 'true'
        
        if auto_detect and not provider and not is_testing:
            try:
                detected = self.detect_available_providers(interactive=True)
                provider = detected['provider']
                model_id = detected['model_id']
                api_key = detected['api_key']
                url = detected['url']
                logger.info(f"Auto-detected provider: {provider}")
            except Exception as e:
                logger.warning(f"Auto-detection failed: {e}, falling back to config")
        
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
            self.provider = provider or 'openai'
            self.model_id = model_id or 'gpt-3.5-turbo'
            self.api_key = api_key or os.getenv('OPENAI_API_KEY')
            self.url = url or 'https://api.openai.com/v1'
            self.max_tokens = 2048
            self.temperature = 0.7
            self.timeout = 30
        
        self.local_model_path = local_model_path
        
        self.client: Optional[Union[Any, object]] = None
        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None
        self._available = True
        
        try:
            self._initialize_provider()
        except Exception as e:
            logger.warning(f"Failed to initialize LLM provider {self.provider}: {e}")
            # Try fallback provider if ollama fails
            if self.provider == 'ollama':
                logger.info("Trying fallback to OpenAI provider...")
                try:
                    self.provider = 'openai'
                    self.model_id = 'gpt-3.5-turbo'
                    self.url = 'https://api.openai.com/v1'
                    self._initialize_provider()
                    logger.info("Successfully fell back to OpenAI provider")
                except Exception as fallback_e:
                    logger.warning(f"Fallback to OpenAI also failed: {fallback_e}")
                    self._available = False
            else:
                self._available = False
            # Re-raise exceptions during testing
            import sys
            if 'pytest' in sys.modules or os.getenv('TESTING') == 'true':
                raise e
    
    def _initialize_provider(self) -> None:
        """Initialize the specific provider with comprehensive error handling."""
        try:
            if self.provider == 'openai':
                self._initialize_openai()
            elif self.provider == 'huggingface':
                self._initialize_huggingface()
            elif self.provider == 'local':
                self._initialize_local()
            elif self.provider in ['grok', 'gemini', 'ollama', 'groq', 'anthropic', 'nanochat']:
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
    
    def _initialize_openai(self) -> None:
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
            # For OpenAI, use base URL without path
            base_url = self.url
            if self.url and 'api.openai.com' in self.url:
                base_url = 'https://api.openai.com/v1'
            
            self.client = OpenAI(
                base_url=base_url,
                api_key=api_key
            )
        except Exception as e:
            raise InitializationError(f"OpenAI client initialization failed: {str(e)}", 
                                    "CLIENT_INIT_FAILED")
    
    def _initialize_huggingface(self) -> None:
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
    
    def _initialize_local(self) -> None:
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
    
    def _initialize_api_provider(self) -> None:
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
            
        # Use async if available, otherwise fallback to sync
        if HAS_AIOHTTP and asyncio is not None:
            try:
                # Create event loop if needed
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # If loop is already running, we need to handle differently
                        import concurrent.futures
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(asyncio.run, self.query_async(prompt, **kwargs))
                            return future.result(timeout=30)
                    else:
                        return loop.run_until_complete(self.query_async(prompt, **kwargs))
                except RuntimeError:
                    # No event loop, create new one
                    return asyncio.run(self.query_async(prompt, **kwargs))
            except Exception as e:
                logger.warning(f"Async query failed, falling back to sync: {e}")
                return self._query_sync_fallback(prompt, **kwargs)
        else:
            return self._query_sync_fallback(prompt, **kwargs)

    @with_error_handling(global_error_handler, fallback_value=None)
    async def query_async(self, prompt: str, **kwargs) -> Optional[str]:
        """Async version of query method for better performance."""
        if not prompt:
            raise ValueError("Empty prompt provided")

        if not prompt.strip():
            raise ValueError("Prompt contains only whitespace")

        try:
            if self.provider == 'openai':
                return await self._query_openai_async(prompt, **kwargs)
            elif self.provider == 'anthropic':
                return await self._query_anthropic_async(prompt, **kwargs)
            elif self.provider == 'gemini':
                return await self._query_gemini_async(prompt, **kwargs)
            elif self.provider == 'ollama':
                return await self._query_ollama_async(prompt, **kwargs)
            elif self.provider == 'groq':
                return await self._query_groq_async(prompt, **kwargs)
            elif self.provider == 'grok':
                return await self._query_grok_async(prompt, **kwargs)
            elif self.provider == 'nanochat':
                return await self._query_nanochat_async(prompt, **kwargs)
            elif self.provider == 'huggingface':
                return self._query_huggingface(prompt, **kwargs)  # Sync for now
            elif self.provider == 'local':
                return self._query_local(prompt, **kwargs)  # Sync for now
            else:
                raise LLMProviderError(f"Provider {self.provider} not implemented",
                                     "PROVIDER_NOT_IMPLEMENTED")

        except Exception as e:
            # Re-raise M.I.A exceptions
            if isinstance(e, (LLMProviderError, NetworkError, ConfigurationError)):
                raise e
            # Convert other exceptions to LLMProviderError
            raise LLMProviderError(f"Async query failed: {str(e)}", "ASYNC_QUERY_FAILED", {
                'provider': self.provider,
                'prompt_length': len(prompt),
                'kwargs': kwargs
            })

    def _query_sync_fallback(self, prompt: str, **kwargs) -> Optional[str]:
        """Fallback synchronous query method."""
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
            elif self.provider == 'nanochat':
                return self._query_nanochat(prompt, **kwargs)
            elif self.provider == 'huggingface':
                return self._query_huggingface(prompt, **kwargs)
            elif self.provider == 'local':
                return self._query_local(prompt, **kwargs)
            else:
                raise LLMProviderError(f"Provider {self.provider} not implemented",
                                     "PROVIDER_NOT_IMPLEMENTED")
        except Exception as e:
            if isinstance(e, (LLMProviderError, NetworkError, ConfigurationError)):
                raise e
            raise LLMProviderError(f"Sync query failed: {str(e)}", "SYNC_QUERY_FAILED", {
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

    def _query_nanochat(self, prompt: str, **kwargs) -> Optional[str]:
        """Query Nanochat API."""
        try:
            headers = {'Content-Type': 'application/json'}
            data = {
                'model': self.model_id or 'nanochat-model',
                'prompt': prompt,
                'stream': False,
                'max_tokens': kwargs.get('max_tokens', self.max_tokens),
                'temperature': kwargs.get('temperature', self.temperature)
            }

            if self.api_key:
                headers['Authorization'] = f'Bearer {self.api_key}'

            url = self.url or 'http://localhost:8081/api/generate'
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()

            if 'response' not in result:
                raise LLMProviderError("Nanochat response missing 'response' field", "MISSING_RESPONSE")

            return result['response']
        except Exception as e:
            logger.error(f"Nanochat API error: {e}")
            return None

    # Async methods for better performance
    async def _query_openai_async(self, prompt: str, **kwargs) -> Optional[str]:
        """Async query OpenAI."""
        if not HAS_AIOHTTP or aiohttp is None:
            return self._query_openai(prompt, **kwargs)

        if self.client is None or not HAS_OPENAI:
            raise LLMProviderError("OpenAI client not available", "CLIENT_NOT_AVAILABLE")

        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    'Authorization': f'Bearer {self.api_key}',
                    'Content-Type': 'application/json'
                }
                data = {
                    'model': self.model_id or 'gpt-3.5-turbo',
                    'messages': [{"role": "user", "content": prompt}],
                    'max_tokens': kwargs.get('max_tokens', 1024),
                    'temperature': kwargs.get('temperature', 0.7)
                }
                url = self.url or 'https://api.openai.com/v1/chat/completions'

                async with session.post(url, headers=headers, json=data) as response:
                    response.raise_for_status()
                    result = await response.json()
                    return result['choices'][0]['message']['content']
        except Exception as e:
            logger.error(f"OpenAI async error: {e}")
            return None

    async def _query_anthropic_async(self, prompt: str, **kwargs) -> Optional[str]:
        """Async query Anthropic Claude API."""
        if not HAS_AIOHTTP or aiohttp is None:
            return self._query_anthropic(prompt, **kwargs)

        try:
            async with aiohttp.ClientSession() as session:
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

                async with session.post(url, headers=headers, json=data) as response:
                    response.raise_for_status()
                    result = await response.json()
                    if 'content' in result and isinstance(result['content'], list) and result['content']:
                        return result['content'][0].get('text', '')
                    return ''
        except Exception as e:
            logger.error(f"Anthropic async API error: {e}")
            return None

    async def _query_gemini_async(self, prompt: str, **kwargs) -> Optional[str]:
        """Async query Google Gemini API."""
        if not HAS_AIOHTTP or aiohttp is None:
            return self._query_gemini(prompt, **kwargs)

        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    'Content-Type': 'application/json',
                    'x-goog-api-key': self.api_key or os.getenv('GEMINI_API_KEY')
                }
                data = {
                    'contents': [{"parts": [{"text": prompt}]}]
                }
                url = self.url or 'https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent'

                async with session.post(url, headers=headers, json=data) as response:
                    response.raise_for_status()
                    result = await response.json()
                    candidates = result.get('candidates', [])
                    if candidates and 'content' in candidates[0]:
                        parts = candidates[0]['content'].get('parts', [])
                        if parts and 'text' in parts[0]:
                            return parts[0]['text']
                    return ''
        except Exception as e:
            logger.error(f"Gemini async API error: {e}")
            return None

    async def _query_ollama_async(self, prompt: str, **kwargs) -> Optional[str]:
        """Async query Ollama local API."""
        if not HAS_AIOHTTP or aiohttp is None:
            return self._query_ollama(prompt, **kwargs)

        try:
            async with aiohttp.ClientSession() as session:
                headers = {'Content-Type': 'application/json'}
                data = {
                    'model': self.model_id or 'mistral:instruct',
                    'prompt': prompt,
                    'stream': False
                }

                if self.api_key and self.api_key != 'ollama':
                    headers['Authorization'] = f'Bearer {self.api_key}'

                url = self.url or 'http://localhost:11434/api/generate'

                async with session.post(url, headers=headers, json=data) as response:
                    response.raise_for_status()
                    result = await response.json()

                    if 'response' not in result:
                        raise LLMProviderError("Ollama response missing 'response' field", "MISSING_RESPONSE")

                    return result['response']
        except Exception as e:
            logger.error(f"Ollama async API error: {e}")
            return None

    async def _query_groq_async(self, prompt: str, **kwargs) -> Optional[str]:
        """Async query Groq API."""
        if not HAS_AIOHTTP or aiohttp is None:
            return self._query_groq(prompt, **kwargs)

        try:
            async with aiohttp.ClientSession() as session:
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

                async with session.post(url, headers=headers, json=data) as response:
                    response.raise_for_status()
                    result = await response.json()
                    return result['choices'][0]['message']['content']
        except Exception as e:
            logger.error(f"Groq async API error: {e}")
            return None

    async def _query_grok_async(self, prompt: str, **kwargs) -> Optional[str]:
        """Async query Grok API."""
        if not HAS_AIOHTTP or aiohttp is None:
            return self._query_grok(prompt, **kwargs)

        try:
            async with aiohttp.ClientSession() as session:
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

                async with session.post(url, headers=headers, json=data) as response:
                    response.raise_for_status()
                    result = await response.json()
                    return result['choices'][0]['message']['content']
        except Exception as e:
            logger.error(f"Grok async API error: {e}")
            return None

    async def _query_nanochat_async(self, prompt: str, **kwargs) -> Optional[str]:
        """Async query Nanochat API."""
        if not HAS_AIOHTTP or aiohttp is None:
            return self._query_nanochat(prompt, **kwargs)

        try:
            async with aiohttp.ClientSession() as session:
                headers = {'Content-Type': 'application/json'}
                data = {
                    'model': self.model_id or 'nanochat-model',
                    'prompt': prompt,
                    'stream': False,
                    'max_tokens': kwargs.get('max_tokens', self.max_tokens),
                    'temperature': kwargs.get('temperature', self.temperature)
                }

                if self.api_key:
                    headers['Authorization'] = f'Bearer {self.api_key}'

                url = self.url or 'http://localhost:8081/api/generate'  # Default nanochat port

                async with session.post(url, headers=headers, json=data) as response:
                    response.raise_for_status()
                    result = await response.json()

                    if 'response' not in result:
                        raise LLMProviderError("Nanochat response missing 'response' field", "MISSING_RESPONSE")

                    return result['response']
        except Exception as e:
            logger.error(f"Nanochat async API error: {e}")
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
