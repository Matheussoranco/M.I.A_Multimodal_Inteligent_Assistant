"""
LLM Manager: Unified interface for multiple LLM APIs (OpenAI, HuggingFace, Local, etc.)
"""
import os
import requests
from openai import OpenAI
# Optionally import HuggingFace, Cohere, etc. as needed
try:
    from transformers.pipelines import pipeline
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    pipeline = None
    AutoModelForCausalLM = None
    AutoTokenizer = None

class LLMManager:
    def __init__(self, provider='openai', model_id=None, api_key=None, url=None, local_model_path=None, **kwargs):
        self.provider = provider
        self.model_id = model_id
        self.api_key = api_key
        self.url = url
        self.local_model_path = local_model_path
        self.client = None
        if provider == 'openai':
            try:
                self.client = OpenAI(base_url=url, api_key=api_key)
            except Exception:
                self.client = None
        elif provider == 'huggingface' and pipeline:
            self.client = pipeline('text-generation', model=model_id)
        elif provider == 'local' and AutoModelForCausalLM and AutoTokenizer:
            self.tokenizer = AutoTokenizer.from_pretrained(local_model_path or model_id)
            self.model = AutoModelForCausalLM.from_pretrained(local_model_path or model_id)
        elif provider == 'grok':
            self.grok_url = url or 'https://api.grok.com/v1/chat/completions'
            self.grok_api_key = api_key
        elif provider == 'gemini':
            self.gemini_url = url or 'https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent'
            self.gemini_api_key = api_key
        elif provider == 'ollama':
            self.ollama_url = url or 'http://localhost:11434/api/generate'
            self.ollama_api_key = api_key
        elif provider == 'groq':
            self.groq_url = url or 'https://api.groq.com/openai/v1/chat/completions'
            self.groq_api_key = api_key
        # Add more providers as needed

    def query_model(self, *args, **kwargs):
        # Alias for compatibility with main.py and other modules
        return self.query(*args, **kwargs)

    def query(self, prompt, **kwargs):
        if self.provider == 'openai' and self.client is not None and type(self.client).__name__ == 'OpenAI':
        elif self.provider == 'gemini':
            headers = {
                'Content-Type': 'application/json',
                'x-goog-api-key': self.gemini_api_key
            }
            data = {
                'contents': [{"parts": [{"text": prompt}]}]
            }
            response = requests.post(self.gemini_url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            # Gemini returns candidates list
            candidates = result.get('candidates', [])
            if candidates and 'content' in candidates[0]:
                parts = candidates[0]['content'].get('parts', [])
                if parts and 'text' in parts[0]:
                    return parts[0]['text']
            return ''
        elif self.provider == 'ollama':
            headers = {
                'Content-Type': 'application/json',
            }
            data = {
                'model': self.model_id or 'llama2',
                'prompt': prompt
            }
            if self.ollama_api_key:
                headers['Authorization'] = f'Bearer {self.ollama_api_key}'
            response = requests.post(self.ollama_url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            return result.get('response', '')
        elif self.provider == 'groq':
            headers = {
                'Authorization': f'Bearer {self.groq_api_key}',
                'Content-Type': 'application/json'
            }
            data = {
                'model': self.model_id or 'llama2-70b-4096',
                'messages': [{"role": "user", "content": prompt}]
            }
            response = requests.post(self.groq_url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content']
            # Defensive: Only call if client is OpenAI instance
            # Ensure model_id is not None and is a string
            model_id = self.model_id if self.model_id is not None else 'gpt-3.5-turbo'
            # Avoid calling OpenAI client if it is actually a HuggingFace pipeline
            # Only call if client is OpenAI, not Pipeline
            if type(self.client).__name__ == 'OpenAI':
                response = self.client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt}],
                    **kwargs
                )
                return response.choices[0].message.content
            else:
                raise RuntimeError("OpenAI client is not properly initialized.")
        elif self.provider == 'huggingface' and self.client and callable(self.client):
            result = self.client(prompt, max_length=256, **kwargs)
            if isinstance(result, list) and 'generated_text' in result[0]:
                return result[0]['generated_text']
            return str(result)
        elif self.provider == 'local' and hasattr(self, 'model'):
            inputs = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.model.generate(**inputs, max_length=256)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        elif self.provider == 'grok':
            headers = {
                'Authorization': f'Bearer {self.grok_api_key}',
                'Content-Type': 'application/json'
            }
            data = {
                'model': self.model_id or 'grok-1',
                'messages': [{"role": "user", "content": prompt}]
            }
            response = requests.post(self.grok_url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            raise NotImplementedError(f"Provider {self.provider} not implemented or dependencies missing.")
