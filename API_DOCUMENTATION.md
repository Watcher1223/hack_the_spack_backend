# API Documentation

**YC Hack2 - AI Agent with Tool Marketplace**
Version: 1.0
Last Updated: 2026-01-31

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [API Endpoints](#api-endpoints)
4. [Tool Marketplace](#tool-marketplace)
5. [Embeddings & Search](#embeddings--search)
6. [Code Examples](#code-examples)
7. [Environment Configuration](#environment-configuration)

---

## Overview

This AI agent system provides:
- **Autonomous task execution** with tool generation and reuse
- **Tool marketplace** with semantic search using Voyage AI embeddings
- **Persistent storage** in MongoDB with vector similarity search
- **RESTful API** for agent interactions and tool management

### Key Features

✅ Autonomous tool generation from API documentation
✅ Semantic tool search with Voyage AI embeddings (1024 dimensions)
✅ Tool persistence and reuse across conversations
✅ MongoDB-backed storage with vector similarity
✅ Real-time tool execution with error handling
✅ Support for multiple LLM providers (Anthropic via OpenRouter)

---

## Architecture

```
┌─────────────┐
│   UI/Chat   │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────┐
│         FastAPI Server              │
│  ┌─────────────────────────────┐   │
│  │    Agent Service (LLM)      │   │
│  └─────────────────────────────┘   │
│              │                      │
│              ▼                      │
│  ┌─────────────────────────────┐   │
│  │     Tool Service            │   │
│  │  - Generate Tools           │   │
│  │  - Execute Tools            │   │
│  │  - Search Tools             │   │
│  └─────────────────────────────┘   │
│              │                      │
│              ▼                      │
│  ┌─────────────────────────────┐   │
│  │    Database Service         │   │
│  │  - Voyage AI Embeddings     │   │
│  │  - Cosine Similarity        │   │
│  │  - MongoDB Storage          │   │
│  └─────────────────────────────┘   │
└─────────────────────────────────────┘
              │
              ▼
     ┌────────────────┐
     │    MongoDB     │
     │   - tools      │
     │   - convs      │
     └────────────────┘
```

---

## API Endpoints

### Base URL

```
http://localhost:8001
```

### Authentication

Currently no authentication required (development mode).

---

### 1. Chat with Agent

**POST** `/chat`

Send a message to the AI agent and receive a response.

#### Request Body

```json
{
  "message": "Get the current Bitcoin price",
  "conversation_id": "optional-conversation-id",
  "model": "anthropic/claude-haiku-4.5",
  "stream": false
}
```

#### Parameters

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `message` | string | Yes | - | User's message/request |
| `conversation_id` | string | No | Auto-generated | Unique conversation identifier |
| `model` | string | No | `anthropic/claude-haiku-4.5` | LLM model to use |
| `stream` | boolean | No | `false` | Enable streaming responses |

#### Response (Non-Streaming)

```json
{
  "response": "Here's the current Bitcoin price: $78,055.00 USD",
  "conversation_id": "20260131_191432",
  "model": "anthropic/claude-haiku-4.5",
  "tool_calls": [
    {
      "name": "get_crypto_price",
      "arguments": {"symbol": "BTCUSD"},
      "result": {
        "symbol": "BTCUSD",
        "price": 78055,
        "market_cap": 1558084519998.399,
        "volume_24h": 74756742542.5399,
        "change_24h": -7.001003917369599
      }
    }
  ]
}
```

#### Response (Streaming)

```
data: {"type": "text", "content": "Let me get that for you..."}
data: {"type": "tool_call", "name": "get_crypto_price", "arguments": {...}}
data: {"type": "tool_result", "result": {...}}
data: {"type": "text", "content": "The current Bitcoin price is..."}
data: {"type": "done"}
```

#### Example Usage

```bash
curl -X POST http://localhost:8001/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Get the current Bitcoin price"
  }'
```

```python
import httpx

async def chat_with_agent(message: str):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8001/chat",
            json={"message": message}
        )
        return response.json()

# Usage
result = await chat_with_agent("Get the current Bitcoin price")
print(result["response"])
```

---

### 2. List Tools

**GET** `/tools`

Retrieve all available tools in the marketplace.

#### Response

```json
{
  "success": true,
  "count": 3,
  "tools": [
    {
      "_id": "697e54b1110dd4e8f38cbc29",
      "name": "get_crypto_price",
      "description": "Get the current price of any cryptocurrency",
      "parameters": {
        "type": "object",
        "properties": {
          "symbol": {
            "type": "string",
            "description": "Cryptocurrency symbol (e.g., BTCUSD, ETHUSD)"
          }
        },
        "required": ["symbol"]
      },
      "created_at": "2026-01-31T19:14:57.195000"
    }
  ]
}
```

#### Example Usage

```bash
curl http://localhost:8001/tools
```

```python
async def list_all_tools():
    async with httpx.AsyncClient() as client:
        response = await client.get("http://localhost:8001/tools")
        return response.json()
```

---

### 3. Search Tools

**GET** `/tools/search`

Search for tools using semantic similarity.

#### Query Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `q` | string | Yes | - | Search query |
| `limit` | integer | No | `10` | Max results (1-50) |

#### Response

```json
{
  "success": true,
  "query": "cryptocurrency price",
  "count": 1,
  "tools": [
    {
      "name": "get_crypto_price",
      "description": "Get the current price of any cryptocurrency",
      "similarity_score": 0.5806206672080676
    }
  ]
}
```

#### Example Usage

```bash
curl "http://localhost:8001/tools/search?q=cryptocurrency%20price&limit=5"
```

```python
async def search_tools(query: str, limit: int = 10):
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "http://localhost:8001/tools/search",
            params={"q": query, "limit": limit}
        )
        return response.json()

# Usage
results = await search_tools("cryptocurrency price")
for tool in results["tools"]:
    print(f"{tool['name']}: {tool['similarity_score']:.2f}")
```

---

### 4. Get Tool Details

**GET** `/tools/{tool_name}`

Get detailed information about a specific tool.

#### Path Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `tool_name` | string | Name of the tool |

#### Response

```json
{
  "success": true,
  "tool": {
    "_id": "697e54b1110dd4e8f38cbc29",
    "name": "get_crypto_price",
    "description": "Get the current price of any cryptocurrency",
    "parameters": {
      "type": "object",
      "properties": {
        "symbol": {
          "type": "string",
          "description": "Cryptocurrency symbol (e.g., BTCUSD, ETHUSD)"
        }
      },
      "required": ["symbol"]
    },
    "code": "async def get_crypto_price(symbol: str) -> dict:\n    ...",
    "dependencies": ["httpx"],
    "created_at": "2026-01-31T19:14:57.195000"
  }
}
```

#### Example Usage

```bash
curl http://localhost:8001/tools/get_crypto_price
```

---

### 5. Delete Tool

**DELETE** `/tools/{tool_name}`

Remove a tool from the marketplace.

#### Path Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `tool_name` | string | Name of the tool to delete |

#### Response

```json
{
  "success": true,
  "message": "Tool 'get_crypto_price' deleted successfully"
}
```

#### Example Usage

```bash
curl -X DELETE http://localhost:8001/tools/get_crypto_price
```

---

### 6. Execute Tool

**POST** `/tools/{tool_name}/execute`

Execute a tool with specific parameters.

#### Request Body

```json
{
  "symbol": "BTCUSD"
}
```

#### Response

```json
{
  "success": true,
  "result": {
    "symbol": "BTCUSD",
    "coin": "bitcoin",
    "price": 78055,
    "currency": "USD",
    "market_cap": 1558084519998.399,
    "volume_24h": 74756742542.5399,
    "change_24h": -7.001003917369599,
    "timestamp": "2026-01-31T19:14:53.341563"
  }
}
```

#### Example Usage

```bash
curl -X POST http://localhost:8001/tools/get_crypto_price/execute \
  -H "Content-Type: application/json" \
  -d '{"symbol": "BTCUSD"}'
```

```python
async def execute_tool(tool_name: str, **kwargs):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"http://localhost:8001/tools/{tool_name}/execute",
            json=kwargs
        )
        return response.json()

# Usage
result = await execute_tool("get_crypto_price", symbol="BTCUSD")
print(f"Bitcoin price: ${result['result']['price']}")
```

---

### 7. Generate Tool

**POST** `/tools/generate`

Generate a new tool from specifications.

#### Request Body

```json
{
  "name": "get_weather",
  "description": "Get current weather for a city",
  "parameters": {
    "type": "object",
    "properties": {
      "city": {
        "type": "string",
        "description": "City name"
      }
    },
    "required": ["city"]
  },
  "code": "async def get_weather(city: str) -> dict:\n    import httpx\n    async with httpx.AsyncClient() as client:\n        response = await client.get(f'https://api.openweathermap.org/data/2.5/weather?q={city}')\n        return response.json()",
  "dependencies": ["httpx"]
}
```

#### Response

```json
{
  "success": true,
  "message": "Tool 'get_weather' saved successfully",
  "has_code": true,
  "upserted": true
}
```

---

## Tool Marketplace

### How Tools Work

1. **Tool Creation**
   - Agent analyzes user request
   - Searches for existing tools first (semantic search)
   - If not found, generates new tool from API docs
   - Tool is saved to MongoDB with Voyage AI embeddings

2. **Tool Discovery**
   - Uses Voyage AI embeddings (1024 dimensions)
   - Calculates cosine similarity for semantic search
   - Returns top-N most relevant tools

3. **Tool Execution**
   - Loads tool code from MongoDB
   - Creates safe execution environment
   - Executes async function with provided parameters
   - Returns structured result

### Available Tools

After initial setup, the following tools are available:

#### `get_crypto_price`

Get current cryptocurrency prices from CoinGecko API.

**Parameters:**
```json
{
  "symbol": "BTCUSD"  // e.g., BTCUSD, ETHUSD, LTCUSD
}
```

**Returns:**
```json
{
  "symbol": "BTCUSD",
  "coin": "bitcoin",
  "price": 78055,
  "currency": "USD",
  "market_cap": 1558084519998.399,
  "volume_24h": 74756742542.5399,
  "change_24h": -7.001003917369599,
  "timestamp": "2026-01-31T19:14:53.341563"
}
```

**Supported Symbols:**
- BTCUSD (Bitcoin)
- ETHUSD (Ethereum)
- LTCUSD (Litecoin)
- XRPUSD (Ripple)
- ADAUSD (Cardano)
- DOGUSD (Dogecoin)
- SOLUSD (Solana)
- And more...

---

## Embeddings & Search

### Voyage AI Integration

The system uses **Voyage AI** for generating embeddings:

**Model:** `voyage-4`
**Dimensions:** 1024
**API:** https://api.voyageai.com/v1/embeddings

#### Input Types

- `"document"` - Used when indexing tools (tool name + description)
- `"query"` - Used when searching for tools

#### Search Process

```python
# 1. Generate embedding for search query
query_embedding = voyage_client.embed(
    texts=["cryptocurrency price"],
    model="voyage-4",
    input_type="query"  # Optimized for search
)

# 2. Fetch all tools with embeddings from MongoDB
tools = db.tools.find({"embedding": {"$exists": True}})

# 3. Calculate cosine similarity
for tool in tools:
    similarity = cosine_similarity(query_embedding, tool["embedding"])
    tool["similarity_score"] = similarity

# 4. Sort by similarity and return top-N
tools.sort(key=lambda x: x["similarity_score"], reverse=True)
return tools[:limit]
```

#### Cosine Similarity Formula

```python
def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    return dot_product / (magnitude1 * magnitude2)
```

---

## Code Examples

### Full Integration Example

```python
import httpx
import asyncio

class AgentClient:
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
        self.client = httpx.AsyncClient()

    async def chat(self, message: str, conversation_id: str = None):
        """Send a message to the agent"""
        response = await self.client.post(
            f"{self.base_url}/chat",
            json={
                "message": message,
                "conversation_id": conversation_id
            }
        )
        return response.json()

    async def search_tools(self, query: str, limit: int = 10):
        """Search for tools semantically"""
        response = await self.client.get(
            f"{self.base_url}/tools/search",
            params={"q": query, "limit": limit}
        )
        return response.json()

    async def execute_tool(self, tool_name: str, **kwargs):
        """Execute a specific tool"""
        response = await self.client.post(
            f"{self.base_url}/tools/{tool_name}/execute",
            json=kwargs
        )
        return response.json()

    async def list_tools(self):
        """List all available tools"""
        response = await self.client.get(f"{self.base_url}/tools")
        return response.json()


# Usage Example
async def main():
    agent = AgentClient()

    # 1. Chat with agent
    result = await agent.chat("Get the current Bitcoin price")
    print(f"Agent: {result['response']}")

    # 2. Search for tools
    tools = await agent.search_tools("cryptocurrency price")
    print(f"Found {tools['count']} tools:")
    for tool in tools['tools']:
        print(f"  - {tool['name']} (similarity: {tool['similarity_score']:.2f})")

    # 3. Execute tool directly
    price = await agent.execute_tool("get_crypto_price", symbol="ETHUSD")
    print(f"Ethereum: ${price['result']['price']}")

    # 4. List all tools
    all_tools = await agent.list_tools()
    print(f"Total tools available: {all_tools['count']}")

asyncio.run(main())
```

### Frontend Integration (React Example)

```typescript
// api.ts
import axios from 'axios';

const API_BASE_URL = 'http://localhost:8001';

export interface ChatRequest {
  message: string;
  conversation_id?: string;
  stream?: boolean;
}

export interface ChatResponse {
  response: string;
  conversation_id: string;
  tool_calls?: ToolCall[];
}

export interface ToolCall {
  name: string;
  arguments: Record<string, any>;
  result: any;
}

export const chatAPI = {
  // Send message to agent
  async chat(request: ChatRequest): Promise<ChatResponse> {
    const { data } = await axios.post(`${API_BASE_URL}/chat`, request);
    return data;
  },

  // Search for tools
  async searchTools(query: string, limit: number = 10) {
    const { data } = await axios.get(`${API_BASE_URL}/tools/search`, {
      params: { q: query, limit }
    });
    return data;
  },

  // List all tools
  async listTools() {
    const { data } = await axios.get(`${API_BASE_URL}/tools`);
    return data;
  },

  // Execute a tool
  async executeTool(toolName: string, params: Record<string, any>) {
    const { data } = await axios.post(
      `${API_BASE_URL}/tools/${toolName}/execute`,
      params
    );
    return data;
  }
};

// Usage in React component
import { useState } from 'react';
import { chatAPI } from './api';

function ChatInterface() {
  const [message, setMessage] = useState('');
  const [response, setResponse] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSend = async () => {
    setLoading(true);
    try {
      const result = await chatAPI.chat({ message });
      setResponse(result.response);
    } catch (error) {
      console.error('Error:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <input
        value={message}
        onChange={(e) => setMessage(e.target.value)}
        placeholder="Ask me anything..."
      />
      <button onClick={handleSend} disabled={loading}>
        {loading ? 'Thinking...' : 'Send'}
      </button>
      {response && <div className="response">{response}</div>}
    </div>
  );
}
```

---

## Environment Configuration

### Required Environment Variables

Create a `dev.env` file in the project root:

```bash
# Voyage AI API Key (required for embeddings)
VOYAGE_API_KEY=your_voyage_api_key_here

# Firecrawl API Key (required for web scraping)
FIRECRAWL_API_KEY=your_firecrawl_api_key_here

# MongoDB Connection (optional - defaults to localhost)
MONGODB_URI=mongodb://admin:admin123@localhost:27017/agent_db?authSource=admin

# OpenRouter API Key (optional - for LLM access)
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

### Getting API Keys

1. **Voyage AI**: https://www.voyageai.com/
   - Sign up for free account
   - Navigate to API Keys section
   - Create new API key

2. **Firecrawl**: https://firecrawl.dev/
   - Sign up for account
   - Get API key from dashboard

3. **OpenRouter** (Optional): https://openrouter.ai/
   - Sign up for account
   - Get API key for LLM access

### Installation

```bash
# 1. Clone repository
git clone <repo-url>
cd yc-hack2

# 2. Install dependencies
uv sync

# 3. Set up environment variables
cp .env.example dev.env
# Edit dev.env with your API keys

# 4. Start MongoDB (if using Docker)
docker-compose up -d mongodb

# 5. Run the server
uvicorn server:app --reload --port 8000
```

### MongoDB Setup

The system uses MongoDB with the following collections:

- **tools**: Stores generated tools with embeddings
- **conversations**: Stores conversation history

**Schema for `tools` collection:**
```json
{
  "_id": ObjectId,
  "name": "get_crypto_price",
  "description": "Get current cryptocurrency prices",
  "parameters": { /* JSON Schema */ },
  "code": "async def get_crypto_price(...)...",
  "dependencies": ["httpx"],
  "embedding": [0.123, 0.456, ...],  // 1024 dimensions
  "created_at": ISODate("2026-01-31T19:14:57.195Z")
}
```

---

## Error Handling

### Standard Error Response

```json
{
  "success": false,
  "error": "Error message here",
  "type": "ErrorType",
  "details": {
    // Additional error context
  }
}
```

### Common Error Codes

| HTTP Status | Error Type | Description |
|-------------|------------|-------------|
| 400 | `ValidationError` | Invalid request parameters |
| 404 | `NotFoundError` | Tool or resource not found |
| 500 | `ExecutionError` | Tool execution failed |
| 500 | `DatabaseError` | MongoDB operation failed |
| 503 | `ServiceUnavailable` | External API unavailable |

### Error Handling Example

```python
try:
    result = await agent.chat("Get Bitcoin price")
except httpx.HTTPStatusError as e:
    if e.response.status_code == 404:
        print("Tool not found")
    elif e.response.status_code == 500:
        error_data = e.response.json()
        print(f"Execution error: {error_data['error']}")
```

---

## Rate Limits

### Current Limits

- **Voyage AI**: 300 requests/minute (free tier)
- **Firecrawl**: Varies by plan
- **CoinGecko API**: 30 calls/minute (free tier)

### Best Practices

1. **Implement caching** for frequently accessed data
2. **Batch requests** when possible
3. **Use tool search** before generating new tools
4. **Handle rate limit errors** gracefully with retries

---

## Monitoring & Debugging

### Logs

Logs are written to:
- **Console**: Real-time application logs
- **Files**: (Optional) Configure logging to files

### Debug Mode

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Health Check

```bash
curl http://localhost:8001/health
```

Response:
```json
{
  "status": "healthy",
  "database": "connected",
  "embeddings": "operational"
}
```

---

## Appendix

### Supported Models

- `anthropic/claude-haiku-4.5` (default, fastest)
- `anthropic/claude-sonnet-4.5` (balanced)
- `anthropic/claude-opus-4.5` (most capable)

### Tool Execution Environment

Generated tools have access to:
- `httpx` - HTTP requests
- `json` - JSON operations
- `base64` - Encoding/decoding
- `asyncio` - Async operations
- `datetime`, `time` - Time handling
- `hashlib`, `os`, `re` - Utilities
- `file_write`, `file_read`, `file_list` - File operations

### Security Considerations

⚠️ **Current Status**: Development mode - No authentication
🔒 **Production TODO**:
- Add API key authentication
- Implement rate limiting per user
- Sandbox tool execution environment
- Validate all user inputs
- Enable CORS properly

---

## Support & Resources

- **GitHub**: <repo-url>
- **Issues**: <repo-url>/issues
- **Documentation**: This file
- **Voyage AI Docs**: https://docs.voyageai.com/
- **MongoDB Docs**: https://docs.mongodb.com/

---

**Last Updated**: 2026-01-31
**Version**: 1.0
**Maintained By**: YC Hack2 Team
