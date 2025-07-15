# M.I.A API Reference - Technical Documentation

## 🔌 API Overview

M.I.A provides a comprehensive REST API and WebSocket interface for enterprise integration. The API is built with FastAPI, providing automatic OpenAPI documentation, request validation, and high performance.

## 🚀 Base Configuration

### API Endpoints
- **Base URL**: `http://localhost:8080/api/v1`
- **WebSocket**: `ws://localhost:8080/ws`
- **Health Check**: `GET /health`
- **Metrics**: `GET /metrics`
- **OpenAPI Docs**: `GET /docs`

### Authentication
```http
Authorization: Bearer <jwt_token>
Content-Type: application/json
```

## 🧠 Core API Endpoints

### 1. Chat Completion API

#### POST `/chat/completions`
Generate responses using the cognitive architecture.

**Request Body:**
```json
{
  "messages": [
    {
      "role": "user",
      "content": "Explain quantum computing",
      "metadata": {
        "timestamp": "2025-07-15T10:00:00Z",
        "user_id": "user_123"
      }
    }
  ],
  "model": "deepseek-r1:1.5b",
  "temperature": 0.7,
  "max_tokens": 2048,
  "stream": false,
  "reasoning_enabled": true,
  "multimodal": {
    "audio": null,
    "image": null
  }
}
```

**Response:**
```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1642608000,
  "model": "deepseek-r1:1.5b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Quantum computing is a revolutionary computational paradigm...",
        "reasoning_chain": [
          {
            "step": 1,
            "type": "logical",
            "content": "First, I need to define quantum computing...",
            "confidence": 0.95
          }
        ]
      },
      "finish_reason": "stop",
      "confidence": 0.92
    }
  ],
  "usage": {
    "prompt_tokens": 15,
    "completion_tokens": 150,
    "total_tokens": 165
  },
  "performance": {
    "processing_time": 1.2,
    "inference_time": 0.8,
    "memory_usage": 2.4
  }
}
```

### 2. Multimodal Processing API

#### POST `/multimodal/process`
Process multimodal inputs (text, audio, image) simultaneously.

**Request Body (multipart/form-data):**
```http
POST /api/v1/multimodal/process
Content-Type: multipart/form-data

text: "Analyze this image and audio"
image: <image_file>
audio: <audio_file>
config: {
  "vision_model": "clip-vit-base-patch32",
  "audio_model": "wav2vec2-base",
  "fusion_strategy": "attention",
  "output_format": "structured"
}
```

**Response:**
```json
{
  "id": "multimodal-456",
  "modalities": {
    "text": {
      "processed": true,
      "embeddings": [0.1, 0.2, ...],
      "tokens": ["analyze", "this", "image", "and", "audio"]
    },
    "image": {
      "processed": true,
      "objects": [
        {
          "label": "person",
          "confidence": 0.95,
          "bbox": [100, 200, 300, 400]
        }
      ],
      "description": "A person standing in a room",
      "embeddings": [0.3, 0.4, ...]
    },
    "audio": {
      "processed": true,
      "transcript": "Hello, can you analyze this image?",
      "sentiment": "neutral",
      "embeddings": [0.5, 0.6, ...]
    }
  },
  "fusion_result": {
    "unified_embedding": [0.2, 0.3, ...],
    "cross_modal_attention": {
      "text_to_image": 0.8,
      "text_to_audio": 0.7,
      "image_to_audio": 0.6
    },
    "interpretation": "The user is asking for analysis of an image showing a person, with supporting audio context."
  },
  "performance": {
    "processing_time": 2.5,
    "vision_time": 1.0,
    "audio_time": 0.8,
    "fusion_time": 0.7
  }
}
```

### 3. Memory Management API

#### POST `/memory/store`
Store information in the vector memory system.

**Request Body:**
```json
{
  "content": "User prefers technical explanations with examples",
  "metadata": {
    "user_id": "user_123",
    "category": "preference",
    "timestamp": "2025-07-15T10:00:00Z",
    "tags": ["technical", "examples", "preference"]
  },
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "collection": "user_preferences"
}
```

**Response:**
```json
{
  "id": "memory-789",
  "stored": true,
  "embedding_id": "embed_123",
  "collection": "user_preferences",
  "similarity_score": null,
  "metadata": {
    "storage_time": 0.05,
    "embedding_time": 0.02,
    "vector_dimension": 384
  }
}
```

#### GET `/memory/search`
Search the vector memory system.

**Query Parameters:**
- `query`: Search query string
- `k`: Number of results (default: 5)
- `threshold`: Similarity threshold (default: 0.7)
- `collection`: Collection name (optional)
- `user_id`: User ID for filtering (optional)

**Response:**
```json
{
  "results": [
    {
      "id": "embed_123",
      "content": "User prefers technical explanations with examples",
      "similarity": 0.95,
      "metadata": {
        "user_id": "user_123",
        "category": "preference",
        "timestamp": "2025-07-15T10:00:00Z",
        "tags": ["technical", "examples", "preference"]
      }
    }
  ],
  "total_results": 1,
  "query_time": 0.03,
  "collection": "user_preferences"
}
```

### 4. Model Management API

#### GET `/models`
List available models and their status.

**Response:**
```json
{
  "models": [
    {
      "id": "deepseek-r1:1.5b",
      "name": "DeepSeek R1 1.5B",
      "provider": "ollama",
      "status": "loaded",
      "capabilities": ["text", "reasoning"],
      "context_length": 128000,
      "parameters": "1.5B",
      "memory_usage": 3.2,
      "performance": {
        "tokens_per_second": 45,
        "average_latency": 320
      }
    },
    {
      "id": "clip-vit-base-patch32",
      "name": "CLIP ViT Base",
      "provider": "huggingface",
      "status": "loaded",
      "capabilities": ["vision", "text"],
      "context_length": 77,
      "parameters": "151M",
      "memory_usage": 0.6
    }
  ],
  "total_models": 2,
  "total_memory": 3.8
}
```

#### POST `/models/load`
Load a specific model.

**Request Body:**
```json
{
  "model_id": "gemma2:9b",
  "provider": "ollama",
  "config": {
    "quantization": "int8",
    "gpu_layers": 32,
    "context_length": 8192
  }
}
```

**Response:**
```json
{
  "model_id": "gemma2:9b",
  "loaded": true,
  "load_time": 15.2,
  "memory_usage": 9.1,
  "status": "ready"
}
```

### 5. Performance Monitoring API

#### GET `/performance/metrics`
Get current performance metrics.

**Response:**
```json
{
  "system": {
    "cpu_usage": 45.2,
    "memory_usage": 67.8,
    "gpu_usage": 32.1,
    "disk_usage": 15.3,
    "network_io": {
      "bytes_sent": 1048576,
      "bytes_recv": 2097152
    }
  },
  "application": {
    "active_connections": 12,
    "requests_per_second": 25.3,
    "average_response_time": 1.2,
    "error_rate": 0.02,
    "cache_hit_rate": 0.85
  },
  "models": {
    "active_models": 3,
    "total_inferences": 15423,
    "average_inference_time": 0.8,
    "tokens_processed": 1847392
  },
  "memory": {
    "vector_db_size": 1.2,
    "cache_size": 0.8,
    "conversation_history": 0.3
  }
}
```

#### GET `/performance/health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-07-15T10:00:00Z",
  "version": "2.0.0",
  "components": {
    "cognitive_architecture": "healthy",
    "llm_manager": "healthy",
    "vector_database": "healthy",
    "cache_system": "healthy",
    "security_manager": "healthy"
  },
  "metrics": {
    "uptime": 86400,
    "total_requests": 15423,
    "active_users": 25,
    "system_load": 0.65
  }
}
```

## 🔌 WebSocket API

### Connection
```javascript
const ws = new WebSocket('ws://localhost:8080/ws');
```

### Real-time Chat
```javascript
// Send message
ws.send(JSON.stringify({
  type: "chat",
  data: {
    message: "Hello, how are you?",
    stream: true,
    model: "deepseek-r1:1.5b"
  }
}));

// Receive streaming response
ws.onmessage = (event) => {
  const response = JSON.parse(event.data);
  if (response.type === "chat_stream") {
    console.log(response.data.delta);
  }
};
```

### Multimodal Streaming
```javascript
// Send multimodal data
ws.send(JSON.stringify({
  type: "multimodal",
  data: {
    text: "Analyze this",
    image_base64: "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD...",
    stream: true
  }
}));
```

## 🔒 Security API

### Authentication

#### POST `/auth/login`
User authentication.

**Request Body:**
```json
{
  "username": "user@example.com",
  "password": "secure_password",
  "mfa_code": "123456"
}
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600,
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "user": {
    "id": "user_123",
    "email": "user@example.com",
    "permissions": ["read", "write", "admin"]
  }
}
```

#### POST `/auth/refresh`
Refresh authentication token.

**Request Body:**
```json
{
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

### Content Filtering

#### POST `/security/filter`
Filter content for safety and compliance.

**Request Body:**
```json
{
  "content": "This is some user-generated content",
  "filters": ["profanity", "toxicity", "pii"],
  "strictness": "medium"
}
```

**Response:**
```json
{
  "filtered": true,
  "safe": true,
  "violations": [],
  "filtered_content": "This is some user-generated content",
  "confidence": 0.95,
  "processing_time": 0.05
}
```

## 📊 Analytics API

### Usage Analytics

#### GET `/analytics/usage`
Get usage statistics.

**Query Parameters:**
- `start_date`: Start date (ISO 8601)
- `end_date`: End date (ISO 8601)
- `granularity`: Hour, day, week, month
- `user_id`: Filter by user (optional)

**Response:**
```json
{
  "period": {
    "start": "2025-07-01T00:00:00Z",
    "end": "2025-07-15T23:59:59Z",
    "granularity": "day"
  },
  "metrics": [
    {
      "date": "2025-07-15",
      "requests": 1523,
      "users": 89,
      "tokens_processed": 184739,
      "response_time_avg": 1.2,
      "error_rate": 0.02
    }
  ],
  "totals": {
    "requests": 23845,
    "unique_users": 456,
    "tokens_processed": 2847392,
    "total_inference_time": 1024.5
  }
}
```

## 🛠️ Development API

### Configuration

#### GET `/config`
Get current configuration.

**Response:**
```json
{
  "llm": {
    "default_model": "deepseek-r1:1.5b",
    "temperature": 0.7,
    "max_tokens": 2048,
    "context_optimization": true
  },
  "multimodal": {
    "vision_model": "clip-vit-base-patch32",
    "audio_model": "wav2vec2-base",
    "fusion_strategy": "attention"
  },
  "memory": {
    "vector_db": "chromadb",
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "max_memory_size": 100000
  },
  "security": {
    "encryption_enabled": true,
    "audit_logging": true,
    "content_filtering": true
  }
}
```

#### PUT `/config`
Update configuration.

**Request Body:**
```json
{
  "llm": {
    "temperature": 0.8,
    "max_tokens": 4096
  },
  "multimodal": {
    "fusion_strategy": "concatenation"
  }
}
```

### Debugging

#### GET `/debug/logs`
Get application logs.

**Query Parameters:**
- `level`: Log level (debug, info, warning, error)
- `limit`: Number of logs (default: 100)
- `offset`: Offset for pagination

**Response:**
```json
{
  "logs": [
    {
      "timestamp": "2025-07-15T10:00:00Z",
      "level": "INFO",
      "message": "Processing multimodal request",
      "context": {
        "request_id": "req_123",
        "user_id": "user_456",
        "processing_time": 1.2
      }
    }
  ],
  "total": 1523,
  "offset": 0,
  "limit": 100
}
```

## 📚 Code Examples

### Python SDK Example
```python
import requests
import json

# Initialize client
class MIAClient:
    def __init__(self, base_url="http://localhost:8080/api/v1"):
        self.base_url = base_url
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer your_token"
        }
    
    def chat(self, message, model="deepseek-r1:1.5b"):
        payload = {
            "messages": [{"role": "user", "content": message}],
            "model": model,
            "reasoning_enabled": True
        }
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json=payload
        )
        
        return response.json()
    
    def multimodal_process(self, text, image_path=None, audio_path=None):
        files = {}
        data = {"text": text}
        
        if image_path:
            files["image"] = open(image_path, "rb")
        if audio_path:
            files["audio"] = open(audio_path, "rb")
        
        response = requests.post(
            f"{self.base_url}/multimodal/process",
            data=data,
            files=files,
            headers={"Authorization": self.headers["Authorization"]}
        )
        
        return response.json()

# Usage
client = MIAClient()
response = client.chat("Explain quantum computing")
print(response["choices"][0]["message"]["content"])
```

### JavaScript SDK Example
```javascript
class MIAClient {
    constructor(baseUrl = "http://localhost:8080/api/v1") {
        this.baseUrl = baseUrl;
        this.headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer your_token"
        };
    }
    
    async chat(message, model = "deepseek-r1:1.5b") {
        const payload = {
            messages: [{ role: "user", content: message }],
            model: model,
            reasoning_enabled: true
        };
        
        const response = await fetch(`${this.baseUrl}/chat/completions`, {
            method: "POST",
            headers: this.headers,
            body: JSON.stringify(payload)
        });
        
        return await response.json();
    }
    
    async streamChat(message, onChunk) {
        const payload = {
            messages: [{ role: "user", content: message }],
            stream: true
        };
        
        const response = await fetch(`${this.baseUrl}/chat/completions`, {
            method: "POST",
            headers: this.headers,
            body: JSON.stringify(payload)
        });
        
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            const chunk = decoder.decode(value);
            const lines = chunk.split('\n');
            
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const data = JSON.parse(line.slice(6));
                    onChunk(data);
                }
            }
        }
    }
}

// Usage
const client = new MIAClient();
client.chat("Hello, how are you?").then(response => {
    console.log(response.choices[0].message.content);
});
```

## 🔧 Error Handling

### Error Response Format
```json
{
  "error": {
    "code": "INVALID_MODEL",
    "message": "The specified model is not available",
    "details": {
      "model_id": "invalid-model:1b",
      "available_models": ["deepseek-r1:1.5b", "gemma2:9b"]
    },
    "timestamp": "2025-07-15T10:00:00Z",
    "request_id": "req_123"
  }
}
```

### Common Error Codes
- `INVALID_MODEL`: Model not found or not loaded
- `AUTHENTICATION_FAILED`: Invalid or expired token
- `RATE_LIMIT_EXCEEDED`: Too many requests
- `CONTENT_FILTERED`: Content blocked by safety filters
- `PROCESSING_ERROR`: Error during processing
- `INSUFFICIENT_RESOURCES`: Not enough system resources
- `VALIDATION_ERROR`: Invalid request format

## 📈 Rate Limiting

All API endpoints are subject to rate limiting:

- **Free Tier**: 100 requests/hour
- **Standard**: 1,000 requests/hour
- **Premium**: 10,000 requests/hour
- **Enterprise**: Custom limits

Rate limit headers:
```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1642611600
```

## 📚 Additional Resources

- **OpenAPI Specification**: Available at `/docs`
- **Postman Collection**: Import from `/api/v1/postman`
- **SDK Documentation**: Language-specific SDKs
- **Integration Examples**: Common integration patterns
- **Best Practices**: Performance and security guidelines

This API reference provides comprehensive coverage of M.I.A's enterprise-grade capabilities, enabling developers to build sophisticated applications leveraging advanced multimodal AI capabilities.
