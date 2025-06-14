"""
LLM Manager: Unified interface for multiple LLM APIs (OpenAI, HuggingFace, Local, etc.)
"""
import os
import requests
from openai import OpenAI
# Optionally import HuggingFace, Cohere, etc. as needed
try:
    from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
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
            self.client = OpenAI(base_url=url, api_key=api_key)
        elif provider == 'huggingface' and pipeline:
            self.client = pipeline('text-generation', model=model_id)
        elif provider == 'local' and AutoModelForCausalLM and AutoTokenizer:
            self.tokenizer = AutoTokenizer.from_pretrained(local_model_path or model_id)
            self.model = AutoModelForCausalLM.from_pretrained(local_model_path or model_id)
        elif provider == 'grok':
            self.grok_url = url or 'https://api.grok.com/v1/chat/completions'
            self.grok_api_key = api_key
        # Add more providers as needed

    def query(self, prompt, **kwargs):
        if self.provider == 'openai':
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )
            return response.choices[0].message.content
        elif self.provider == 'huggingface' and self.client:
            result = self.client(prompt, max_length=256, **kwargs)
            return result[0]['generated_text']
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
