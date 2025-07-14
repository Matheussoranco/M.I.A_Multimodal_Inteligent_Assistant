#!/usr/bin/env python3
"""
Simple test script to check Ollama connection
"""
import requests
import json

def test_ollama_connection():
    url = "http://localhost:11434/api/generate"
    headers = {'Content-Type': 'application/json'}
    data = {
        'model': 'deepseek-r1:1.5b',
        'prompt': 'Hello! What is your name?',
        'stream': False
    }
    
    try:
        print("Testing Ollama connection...")
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        result = response.json()
        print(f"Response: {result.get('response', 'No response')}")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    if test_ollama_connection():
        print("Ollama connection successful!")
    else:
        print("Ollama connection failed!")
