"""
LLM Manager: Unified interface for multiple LLM APIs (OpenAI, HuggingFace, Local, etc.)
"""
import os
import requests
import logging
from typing import Optional, Dict, Any

# Optional imports with error handling
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    OpenAI = None
    HAS_OPENAI = False

try:
    from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    pipeline = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
    HAS_TRANSFORMERS = False

logger = logging.getLogger(__name__)

class LLMManager:
    """Unified LLM manager supporting multiple providers."""
    
    def __init__(self, provider='openai', model_id=None, api_key=None, url=None, local_model_path=None, **kwargs):
        self.provider = provider
        self.model_id = model_id
        self.api_key = api_key
        self.url = url
        self.local_model_path = local_model_path
        self.client = None
        self.model = None
        self.tokenizer = None
        
        self._initialize_provider()
    
    def _initialize_provider(self):
        """Initialize the specific provider."""
        try:
            if self.provider == 'openai':
                if not HAS_OPENAI:
                    raise ImportError("OpenAI package not installed. Run: pip install openai")
                self.client = OpenAI(
                    base_url=self.url,
                    api_key=self.api_key or os.getenv('OPENAI_API_KEY')
                )
                
            elif self.provider == 'huggingface':
                if not HAS_TRANSFORMERS:
                    raise ImportError("Transformers package not installed. Run: pip install transformers")
                self.client = pipeline('text-generation', model=self.model_id)
                
            elif self.provider == 'local':
                if not HAS_TRANSFORMERS:
                    raise ImportError("Transformers package not installed. Run: pip install transformers")
                model_path = self.local_model_path or self.model_id
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.model = AutoModelForCausalLM.from_pretrained(model_path)
                
            elif self.provider in ['grok', 'gemini', 'ollama', 'groq', 'anthropic']:
                # API-based providers - no special initialization needed
                pass
            else:
                logger.warning(f"Unknown provider: {self.provider}")
                
        except Exception as e:
            logger.error(f"Failed to initialize provider {self.provider}: {e}")
            self.client = None

    def query_model(self, prompt: str, **kwargs) -> Optional[str]:
        """Alias for compatibility with main.py and other modules."""
        return self.query(prompt, **kwargs)

    def query(self, prompt: str, **kwargs) -> Optional[str]:
        """Query the selected LLM provider with the given prompt."""
        if not prompt:
            return None
            
        try:
            if self.provider == 'openai' and self.client is not None:
                response = self.client.chat.completions.create(
                    model=self.model_id or 'gpt-3.5-turbo',
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=kwargs.get('max_tokens', 1024),
                    temperature=kwargs.get('temperature', 0.7)
                )
                return response.choices[0].message.content

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
                
            elif self.provider == 'huggingface' and self.client and callable(self.client):
                result = self.client(prompt, max_length=kwargs.get('max_length', 256))
                if isinstance(result, list) and len(result) > 0 and 'generated_text' in result[0]:
                    return result[0]['generated_text']
                return str(result)
                
            elif self.provider == 'local' and self.model and self.tokenizer:
                inputs = self.tokenizer(prompt, return_tensors="pt")
                outputs = self.model.generate(**inputs, max_length=kwargs.get('max_length', 256))
                return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
            else:
                logger.error(f"Provider {self.provider} not implemented or dependencies missing.")
                return None
                
        except Exception as e:
            logger.error(f"Error querying {self.provider}: {e}")
            return None

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
        """Query Ollama local API."""
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
            response = requests.post(url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            result = response.json()
            return result.get('response', '')
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            logger.error(f"Make sure Ollama is running: ollama serve")
            return None

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
            if self.provider == 'openai' and self.client is not None:
                # Test with a simple query
                response = self.client.chat.completions.create(
                    model=self.model_id or 'gpt-3.5-turbo',
                    messages=[{"role": "user", "content": "Hello"}],
                    max_tokens=10
                )
                return response.choices[0].message.content is not None
            
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
            
            elif self.provider == 'huggingface' and self.client:
                return callable(self.client)
            
            elif self.provider == 'local' and self.model and self.tokenizer:
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
