# Backend API Requirements for Universal Adapter UI

**Document Version:** 1.0
**Date:** 2026-01-31
**Status:** Requirements Specification
**Target:** Backend Development Team

---

## Executive Summary

This document specifies all backend APIs required to replace placeholder/mock data in the Universal Adapter frontend and make it fully functional. The UI is built with Next.js 16.1.6 and expects a RESTful API with real-time capabilities.

**Current State:** UI has comprehensive placeholder data across 6 major views
**Target State:** Full integration with production backend APIs
**Priority Level:** HIGH - Critical for production deployment

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture Requirements](#architecture-requirements)
3. [API Endpoints Specification](#api-endpoints-specification)
4. [WebSocket/Streaming Requirements](#websocketstreaming-requirements)
5. [Data Models](#data-models)
6. [Authentication & Authorization](#authentication--authorization)
7. [Integration Priority Matrix](#integration-priority-matrix)
8. [Migration from Placeholders](#migration-from-placeholders)
9. [Error Handling Requirements](#error-handling-requirements)
10. [Performance & Scalability](#performance--scalability)

---

## Overview

### Current UI Components Requiring Backend Integration

| Component | Current State | Backend Needs | Priority |
|-----------|---------------|---------------|----------|
| **CommandCenter** | Mock flow simulation | Natural language command processing | P0 (Critical) |
| **LiveDiscoveryHUD** | Hardcoded demo logs | Real-time Firecrawl/MCP event streaming | P0 |
| **MCPForge** | Static markdown & TypeScript | Dynamic API doc parsing & code generation | P0 |
| **ToolMarketplace** | 6 hardcoded tools | Full tool CRUD + semantic search | P0 |
| **TrustGovernanceLedger** | Subset of marketplace tools | Audit trail & verification status | P1 |
| **ActionCenter** | 5 demo action items | Real action feed from agent activity | P1 |
| **ApiAccess** | Client-side mock key generation | Secure API key management | P2 |
| **ResultCard** | Hardcoded weather data | Dynamic tool execution results | P0 |

### Current Backend Implementation (from API_DOCUMENTATION.md)

**Base URL:** `http://localhost:8000`

**Existing Endpoints:**
- ✅ `POST /chat` - Chat with agent
- ✅ `GET /tools` - List tools
- ✅ `GET /tools/search` - Search tools (semantic)
- ✅ `GET /tools/{tool_name}` - Get tool details
- ✅ `DELETE /tools/{tool_name}` - Delete tool
- ✅ `POST /tools/{tool_name}/execute` - Execute tool
- ✅ `POST /tools/generate` - Generate new tool

**Technology Stack (Backend):**
- FastAPI
- MongoDB (tools, conversations)
- Voyage AI (embeddings - voyage-4 model, 1024 dimensions)
- Firecrawl (web scraping)
- OpenRouter (LLM access)

---

## Architecture Requirements

### 1. Base Configuration

```bash
# Environment Variables Expected by UI
NEXT_PUBLIC_API_URL=http://localhost:8000  # Backend base URL
OPENWEATHER_API_KEY=your_key_here         # For weather tool demo
```

### 2. CORS Configuration

**Required Headers:**
```http
Access-Control-Allow-Origin: http://localhost:3000
Access-Control-Allow-Methods: GET, POST, PUT, DELETE, OPTIONS
Access-Control-Allow-Headers: Content-Type, Authorization
Access-Control-Allow-Credentials: true
```

**Production:** Replace localhost with actual frontend domain

### 3. Response Format Standards

**Success Response:**
```json
{
  "success": true,
  "data": { ... },
  "metadata": {
    "timestamp": "2026-01-31T12:00:00Z",
    "version": "1.0"
  }
}
```

**Error Response:**
```json
{
  "success": false,
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable error message",
    "details": { ... }
  }
}
```

---

## API Endpoints Specification

### Priority 0 (P0) - Critical for Core Functionality

#### 1. Natural Language Command Processing

**Current Gap:** UI sends user commands to `/chat` endpoint but needs enhanced response format

**Endpoint:** `POST /chat`

**Current Implementation:**
```json
{
  "message": "string",
  "conversation_id": "optional-string",
  "model": "anthropic/claude-haiku-4.5",
  "stream": false
}
```

**Enhanced Requirements for UI:**

**Request:**
```json
{
  "message": "Get weather for New York",
  "conversation_id": "optional-uuid",
  "model": "anthropic/claude-haiku-4.5",
  "stream": false,
  "context": {
    "ui_mode": "command_center",
    "view": "forge"
  }
}
```

**Response (Enhanced):**
```json
{
  "success": true,
  "response": "I'll get the weather for New York using the OpenWeatherMap tool.",
  "conversation_id": "550e8400-e29b-41d4-a716-446655440000",
  "model": "anthropic/claude-haiku-4.5",

  "workflow_steps": [
    {
      "step": "checking",
      "status": "completed",
      "duration_ms": 150,
      "message": "Analyzing command..."
    },
    {
      "step": "discovering",
      "status": "completed",
      "duration_ms": 300,
      "message": "Found tool: get_current_weather"
    },
    {
      "step": "forging",
      "status": "completed",
      "duration_ms": 500,
      "message": "Preparing tool execution..."
    }
  ],

  "tool_calls": [
    {
      "id": "tc_001",
      "name": "get_current_weather",
      "arguments": {
        "q": "New York",
        "units": "metric"
      },
      "result": {
        "city": "New York",
        "temp": "12°C",
        "condition": "Clear sky",
        "humidity": "72%",
        "timestamp": "2026-01-31T12:00:00Z"
      },
      "execution_time_ms": 1200,
      "status": "success"
    }
  ],

  "actions_logged": [
    {
      "id": "act_001",
      "title": "Agent called OpenWeatherMap API",
      "detail": "Weather fetched for New York",
      "status": "success",
      "timestamp": "2026-01-31T12:00:00Z"
    }
  ],

  "metadata": {
    "total_duration_ms": 2150,
    "tokens_used": 450,
    "cost_usd": 0.0023
  }
}
```

**Why This Change?**
- UI shows step-by-step progress (`checking` → `discovering` → `forging` → `done`)
- UI displays actions in ActionCenter component
- UI shows tool execution results in ResultCard
- UI needs structured data, not just chat response

---

#### 2. Real-Time Discovery Event Streaming

**Current Gap:** UI displays hardcoded demo logs in LiveDiscoveryHUD

**New Endpoint:** `GET /api/discovery/stream` (Server-Sent Events)

**Connection:**
```javascript
const eventSource = new EventSource(
  'http://localhost:8000/api/discovery/stream?conversation_id=550e8400-...'
);

eventSource.onmessage = (event) => {
  const logEntry = JSON.parse(event.data);
  // UI appends to log display
};
```

**Event Format:**
```json
{
  "id": "log_001",
  "timestamp": "00:00:01.234",
  "source": "firecrawl",
  "message": "Crawling https://api.openweathermap.org/docs...",
  "level": "info",
  "metadata": {
    "url": "https://api.openweathermap.org/docs",
    "status_code": 200
  }
}
```

**Event Sources (must match UI expectations):**
- `"firecrawl"` - Web scraping events
- `"mcp"` - Tool generation events
- `"agent"` - Agent decision/execution events
- `"system"` - System-level notifications

**Example Event Sequence:**
```json
// Event 1
{"timestamp": "00:00:01", "source": "firecrawl", "message": "Crawling https://api.openweathermap.org/docs..."}

// Event 2
{"timestamp": "00:00:02", "source": "firecrawl", "message": "[Firecrawl] Found /api/docs/current, /api/docs/forecast"}

// Event 3
{"timestamp": "00:00:03", "source": "firecrawl", "message": "[Firecrawl] Extracted 12 endpoints, 3 auth params"}

// Event 4
{"timestamp": "00:00:04", "source": "mcp", "message": "Generating MCP tool: get_current_weather"}

// Event 5
{"timestamp": "00:00:05", "source": "mcp", "message": "TypeScript definition written to marketplace"}

// Event 6
{"timestamp": "00:00:06", "source": "agent", "message": "Tool registered. Executing with params: {q: 'New York'}"}

// Event 7
{"timestamp": "00:00:07", "source": "system", "message": "✓ Success → Weather: 12°C, Clear sky"}
```

**Implementation Notes:**
- Use Server-Sent Events (SSE) for one-way streaming
- Each event must have monotonically increasing timestamp
- Connection must stay open during tool discovery/execution
- Close connection with `data: [DONE]\n\n` when complete

---

#### 3. Tool Marketplace with Semantic Search

**Current Gap:** UI displays 6 hardcoded tools from `/src/data/marketplace-tools.ts`

**Endpoint (Already Exists):** `GET /tools`

**Current Implementation:** ✅ Already returns tools from MongoDB

**Enhancement Required:** Return additional fields for UI

**Current Response:**
```json
{
  "success": true,
  "count": 3,
  "tools": [
    {
      "_id": "697e54b1110dd4e8f38cbc29",
      "name": "get_crypto_price",
      "description": "Get the current price of any cryptocurrency",
      "parameters": { ... },
      "created_at": "2026-01-31T19:14:57.195000"
    }
  ]
}
```

**Enhanced Response Required:**
```json
{
  "success": true,
  "count": 6,
  "tools": [
    {
      "id": "697e54b1110dd4e8f38cbc29",
      "name": "get_current_weather",
      "description": "Get current weather data for any city worldwide",
      "status": "PROD-READY",
      "source_url": "https://api.openweathermap.org/docs",
      "preview_snippet": "get_current_weather(q: string, units?: 'metric' | 'imperial'): Promise<WeatherData>",
      "created_at": "2026-01-31T19:14:57.195000",
      "mux_playback_id": "demo-playback-123",
      "category": "weather",
      "tags": ["weather", "api", "data"],
      "verified": true,
      "usage_count": 1250
    }
  ]
}
```

**New Fields Needed:**
- `status` - Tool status: `"PROD-READY"`, `"BETA"`, `"DEPRECATED"`
- `source_url` - Original API documentation URL
- `preview_snippet` - Type signature preview
- `mux_playback_id` - Video demo ID (optional, can be null)
- `category` - Tool category
- `tags` - Searchable tags array
- `verified` - Boolean verification status
- `usage_count` - Number of times tool has been executed

**Search Endpoint (Already Exists):** `GET /tools/search?q={query}&limit={limit}`

**Current Implementation:** ✅ Already uses Voyage AI embeddings

**No Changes Required** - Current semantic search is sufficient

---

#### 4. Dynamic MCP Tool Code Generation

**Current Gap:** UI shows static markdown & TypeScript in MCPForge component

**New Endpoint:** `POST /api/forge/generate`

**Purpose:** Generate MCP tool from API documentation URL

**Request:**
```json
{
  "source_url": "https://api.openweathermap.org/docs",
  "force_regenerate": false
}
```

**Response:**
```json
{
  "success": true,
  "tool_id": "get_current_weather",
  "documentation": {
    "markdown": "# OpenWeatherMap API\n\n## GET /data/2.5/weather\n\n**Description:** Get current weather data for a city\n\n**Parameters:**\n- `q` (string, required): City name\n- `appid` (string, required): API key\n- `units` (string, optional): 'metric' or 'imperial'\n\n**Response:**\n```json\n{\n  \"main\": {\"temp\": 12.5, \"humidity\": 72},\n  \"weather\": [{\"description\": \"clear sky\"}]\n}\n```",
    "endpoints_found": 12,
    "auth_params": ["appid"],
    "base_url": "https://api.openweathermap.org"
  },
  "generated_code": {
    "typescript": "// Auto-generated MCP tool from API documentation\n// Source: https://api.openweathermap.org/docs\n// Generated: 2026-01-31T12:00:00Z\n\nimport { z } from 'zod';\n\nexport const get_current_weather = {\n  name: 'get_current_weather',\n  description: 'Get current weather data for any city worldwide',\n  parameters: z.object({\n    q: z.string().describe('City name (e.g., New York, London)'),\n    units: z.enum(['metric', 'imperial']).default('metric').describe('Temperature units'),\n  }),\n  execute: async (params: { q: string; units?: string }) => {\n    const apiKey = process.env.OPENWEATHER_API_KEY;\n    if (!apiKey) throw new Error('OPENWEATHER_API_KEY not configured');\n    \n    const url = `https://api.openweathermap.org/data/2.5/weather?q=${params.q}&units=${params.units ?? 'metric'}&appid=${apiKey}`;\n    const response = await fetch(url);\n    \n    if (!response.ok) {\n      throw new Error(`Weather API error: ${response.statusText}`);\n    }\n    \n    const data = await response.json();\n    return {\n      city: data.name,\n      temp: `${Math.round(data.main.temp)}°${params.units === 'imperial' ? 'F' : 'C'}`,\n      condition: data.weather[0].description,\n      humidity: `${data.main.humidity}%`,\n      timestamp: new Date().toISOString(),\n    };\n  },\n};\n\nexport default get_current_weather;",
    "language": "typescript",
    "framework": "mcp"
  },
  "discovery_logs": [
    {
      "timestamp": "00:00:01",
      "source": "firecrawl",
      "message": "Crawling https://api.openweathermap.org/docs..."
    },
    {
      "timestamp": "00:00:02",
      "source": "firecrawl",
      "message": "[Firecrawl] Found /api/docs/current, /api/docs/forecast"
    },
    {
      "timestamp": "00:00:03",
      "source": "firecrawl",
      "message": "[Firecrawl] Extracted 12 endpoints, 3 auth params"
    },
    {
      "timestamp": "00:00:04",
      "source": "mcp",
      "message": "Generating MCP tool: get_current_weather"
    },
    {
      "timestamp": "00:00:05",
      "source": "mcp",
      "message": "TypeScript definition written to marketplace"
    }
  ],
  "metadata": {
    "generation_time_ms": 4500,
    "firecrawl_pages_crawled": 3,
    "tokens_used": 2500
  }
}
```

**UI Integration:**
- Left panel: Display `documentation.markdown`
- Right panel: Display `generated_code.typescript` with syntax highlighting
- Bottom panel: Stream `discovery_logs` in real-time

---

#### 5. Tool Execution with Detailed Results

**Endpoint (Already Exists):** `POST /tools/{tool_name}/execute`

**Current Implementation:** ✅ Works but needs enhanced response

**Request:**
```json
{
  "q": "New York",
  "units": "metric"
}
```

**Current Response:**
```json
{
  "success": true,
  "result": {
    "symbol": "BTCUSD",
    "price": 78055,
    ...
  }
}
```

**Enhanced Response Required:**
```json
{
  "success": true,
  "tool_name": "get_current_weather",
  "execution_id": "exec_550e8400",
  "result": {
    "city": "New York",
    "temp": "12°C",
    "condition": "Clear sky",
    "humidity": "72%",
    "timestamp": "2026-01-31T12:00:00Z"
  },
  "execution_metadata": {
    "started_at": "2026-01-31T12:00:00.000Z",
    "completed_at": "2026-01-31T12:00:01.234Z",
    "duration_ms": 1234,
    "api_calls_made": 1,
    "cached": false
  },
  "logs": [
    {"timestamp": "00:00:00.100", "message": "Validating parameters..."},
    {"timestamp": "00:00:00.200", "message": "Calling OpenWeatherMap API..."},
    {"timestamp": "00:00:01.234", "message": "Success - Weather data retrieved"}
  ]
}
```

---

### Priority 1 (P1) - Important for Full Experience

#### 6. Action Feed API

**Current Gap:** UI shows 5 hardcoded action items in ActionCenter

**New Endpoint:** `GET /api/actions`

**Query Parameters:**
- `conversation_id` (optional) - Filter by conversation
- `limit` (optional, default: 50) - Max items to return
- `offset` (optional, default: 0) - Pagination offset

**Response:**
```json
{
  "success": true,
  "count": 23,
  "actions": [
    {
      "id": "act_001",
      "title": "Agent called OpenWeatherMap API",
      "detail": "Weather fetched for New York → 12°C, Clear sky",
      "status": "success",
      "timestamp": "2026-01-31T12:00:00Z",
      "github_pr_url": null,
      "tool_name": "get_current_weather",
      "execution_id": "exec_550e8400"
    },
    {
      "id": "act_002",
      "title": "Notification sent to CTO",
      "detail": "Email sent via Resend → team@company.com",
      "status": "success",
      "timestamp": "2026-01-31T11:58:00Z",
      "github_pr_url": null,
      "tool_name": "send_email",
      "execution_id": "exec_550e8401"
    },
    {
      "id": "act_003",
      "title": "MCP tool updated",
      "detail": "get_current_weather — Added error handling",
      "status": "success",
      "timestamp": "2026-01-31T11:55:00Z",
      "github_pr_url": "https://github.com/your-org/mcp-tools/pull/42",
      "tool_name": null,
      "execution_id": null
    },
    {
      "id": "act_004",
      "title": "Firecrawl ingestion started",
      "detail": "OpenWeatherMap docs → 12 endpoints discovered",
      "status": "success",
      "timestamp": "2026-01-31T11:50:00Z",
      "github_pr_url": null,
      "tool_name": null,
      "execution_id": null
    },
    {
      "id": "act_005",
      "title": "Tool execution pending",
      "detail": "Instagram media fetch for @brand_account",
      "status": "pending",
      "timestamp": "2026-01-31T12:01:00Z",
      "github_pr_url": null,
      "tool_name": "get_instagram_media",
      "execution_id": "exec_550e8402"
    }
  ]
}
```

**Status Values:**
- `"success"` - Green checkmark ✓
- `"pending"` - Animated spinner
- `"error"` - Red X

---

#### 7. Trust & Governance Ledger API

**Current Gap:** UI shows subset of marketplace tools in TrustGovernanceLedger

**New Endpoint:** `GET /api/governance/verified-tools`

**Response:**
```json
{
  "success": true,
  "count": 4,
  "tools": [
    {
      "id": "697e54b1110dd4e8f38cbc29",
      "name": "get_current_weather",
      "description": "Get current weather data for any city worldwide",
      "status": "PROD-READY",
      "source_url": "https://api.openweathermap.org/docs",
      "preview_snippet": "get_current_weather(q: string, units?: 'metric' | 'imperial'): Promise<WeatherData>",
      "verification": {
        "verified": true,
        "verified_at": "2026-01-30T10:00:00Z",
        "verified_by": "admin@company.com",
        "trust_score": 98,
        "security_scan_passed": true,
        "last_audit": "2026-01-31T08:00:00Z"
      },
      "governance": {
        "approval_required": false,
        "allowed_users": ["*"],
        "rate_limit_per_minute": 60,
        "cost_per_execution": 0.001
      }
    }
  ]
}
```

---

### Priority 2 (P2) - Nice to Have

#### 8. API Key Management

**Current Gap:** UI generates mock API keys client-side in ApiAccess component

**New Endpoints:**

**8.1 Generate API Key**
```http
POST /api/keys/generate
```

**Request:**
```json
{
  "name": "Production Key",
  "scopes": ["tools:read", "tools:execute"],
  "expires_at": "2027-01-31T00:00:00Z"
}
```

**Response:**
```json
{
  "success": true,
  "key": {
    "id": "key_550e8400",
    "key": "ua_live_a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6",
    "name": "Production Key",
    "scopes": ["tools:read", "tools:execute"],
    "created_at": "2026-01-31T12:00:00Z",
    "expires_at": "2027-01-31T00:00:00Z",
    "last_used_at": null
  },
  "warning": "This is the only time you'll see the full key. Store it securely."
}
```

**8.2 List API Keys**
```http
GET /api/keys
```

**Response:**
```json
{
  "success": true,
  "keys": [
    {
      "id": "key_550e8400",
      "name": "Production Key",
      "key_prefix": "ua_live_a1b2...",
      "scopes": ["tools:read", "tools:execute"],
      "created_at": "2026-01-31T12:00:00Z",
      "expires_at": "2027-01-31T00:00:00Z",
      "last_used_at": "2026-01-31T12:30:00Z"
    }
  ]
}
```

**8.3 Revoke API Key**
```http
DELETE /api/keys/{key_id}
```

**Response:**
```json
{
  "success": true,
  "message": "API key revoked successfully"
}
```

---

## WebSocket/Streaming Requirements

### Server-Sent Events (SSE) for Discovery Logs

**Endpoint:** `GET /api/discovery/stream`

**Connection Flow:**
```javascript
// UI establishes SSE connection
const eventSource = new EventSource(
  'http://localhost:8000/api/discovery/stream?conversation_id=550e8400'
);

// Backend sends events
res.write(`data: ${JSON.stringify(logEntry)}\n\n`);

// Backend closes connection when done
res.write('data: [DONE]\n\n');
res.end();
```

**Event Format:**
```
data: {"timestamp": "00:00:01", "source": "firecrawl", "message": "Crawling..."}

data: {"timestamp": "00:00:02", "source": "mcp", "message": "Generating tool..."}

data: [DONE]
```

---

## Data Models

### Tool Schema (MongoDB)

**Enhanced Schema Required:**
```javascript
{
  _id: ObjectId,
  id: String,                    // Human-readable ID (e.g., "get_current_weather")
  name: String,                  // Display name
  description: String,           // Tool description
  status: String,                // "PROD-READY" | "BETA" | "DEPRECATED"
  source_url: String,            // API docs URL
  preview_snippet: String,       // Type signature
  category: String,              // e.g., "weather", "crypto", "social"
  tags: [String],                // Searchable tags
  verified: Boolean,             // Verification status
  usage_count: Number,           // Execution counter
  mux_playback_id: String,       // Video demo ID (optional)

  parameters: {                  // JSON Schema
    type: "object",
    properties: {...},
    required: [...]
  },

  code: String,                  // Tool implementation code
  dependencies: [String],        // npm packages required

  embedding: [Float],            // Voyage AI embedding (1024 dims)

  verification: {
    verified: Boolean,
    verified_at: ISODate,
    verified_by: String,
    trust_score: Number,         // 0-100
    security_scan_passed: Boolean,
    last_audit: ISODate
  },

  governance: {
    approval_required: Boolean,
    allowed_users: [String],     // ["*"] for all, or specific emails
    rate_limit_per_minute: Number,
    cost_per_execution: Number   // USD
  },

  created_at: ISODate,
  updated_at: ISODate
}
```

### Conversation Schema

```javascript
{
  _id: ObjectId,
  conversation_id: String,       // UUID
  messages: [
    {
      role: "user" | "assistant",
      content: String,
      timestamp: ISODate
    }
  ],
  workflow_steps: [
    {
      step: "checking" | "discovering" | "forging" | "done",
      status: "completed" | "failed",
      duration_ms: Number,
      message: String
    }
  ],
  tool_calls: [
    {
      id: String,
      name: String,
      arguments: Object,
      result: Object,
      execution_time_ms: Number,
      status: "success" | "error",
      timestamp: ISODate
    }
  ],
  created_at: ISODate,
  updated_at: ISODate
}
```

### Action Schema

```javascript
{
  _id: ObjectId,
  id: String,                    // "act_001"
  conversation_id: String,       // Link to conversation
  title: String,                 // Short action description
  detail: String,                // Full details
  status: "success" | "pending" | "error",
  timestamp: ISODate,
  github_pr_url: String,         // Optional PR link
  tool_name: String,             // Optional tool reference
  execution_id: String,          // Optional execution reference
  metadata: Object               // Additional context
}
```

### API Key Schema

```javascript
{
  _id: ObjectId,
  id: String,                    // "key_550e8400"
  user_id: String,               // User who created the key
  key_hash: String,              // Hashed key (store hash, not plaintext)
  key_prefix: String,            // First 12 chars for display
  name: String,                  // User-defined name
  scopes: [String],              // ["tools:read", "tools:execute", ...]
  created_at: ISODate,
  expires_at: ISODate,
  last_used_at: ISODate,
  revoked: Boolean,
  revoked_at: ISODate
}
```

---

## Authentication & Authorization

### Current State
- No authentication in API_DOCUMENTATION.md
- UI expects Bearer token authentication

### Required Implementation

**1. API Key Authentication**

All endpoints except `/health` must require authentication:

```http
Authorization: Bearer ua_live_a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6
```

**2. Middleware**
```python
from fastapi import Security, HTTPException
from fastapi.security import HTTPBearer

security = HTTPBearer()

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)):
    api_key = credentials.credentials
    # Verify key exists and is not expired/revoked
    if not is_valid_key(api_key):
        raise HTTPException(status_code=401, detail="Invalid or expired API key")
    return api_key
```

**3. Rate Limiting**

Per API key:
- 100 requests per minute for standard keys
- 1000 requests per minute for premium keys

**4. Scope-Based Access Control**

Scopes:
- `tools:read` - List and search tools
- `tools:write` - Create/update/delete tools
- `tools:execute` - Execute tools
- `chat:write` - Use chat endpoint
- `keys:manage` - Manage API keys

---

## Integration Priority Matrix

| Priority | Endpoint | Component Affected | Implementation Effort | User Impact |
|----------|----------|-------------------|----------------------|-------------|
| **P0** | `POST /chat` (enhanced) | CommandCenter, ResultCard | Medium | Critical |
| **P0** | `GET /api/discovery/stream` | LiveDiscoveryHUD | High | Critical |
| **P0** | `GET /tools` (enhanced) | ToolMarketplace | Low | Critical |
| **P0** | `POST /api/forge/generate` | MCPForge | High | Critical |
| **P0** | `POST /tools/{name}/execute` (enhanced) | ResultCard | Low | Critical |
| **P1** | `GET /api/actions` | ActionCenter | Medium | High |
| **P1** | `GET /api/governance/verified-tools` | TrustGovernanceLedger | Medium | Medium |
| **P2** | `POST /api/keys/generate` | ApiAccess | Medium | Low |
| **P2** | `GET /api/keys` | ApiAccess | Low | Low |
| **P2** | `DELETE /api/keys/{id}` | ApiAccess | Low | Low |

---

## Migration from Placeholders

### Step-by-Step Integration Plan

#### Phase 1: Core Chat & Tool Execution (Week 1)
1. ✅ Enhance `POST /chat` endpoint with `workflow_steps` and `actions_logged`
2. ✅ Enhance `POST /tools/{name}/execute` with `execution_metadata`
3. ✅ Update Tool schema with new fields (`status`, `source_url`, etc.)
4. ✅ Test CommandCenter → ResultCard flow end-to-end

**UI Changes Required:**
```typescript
// src/components/CommandCenter.tsx
// Replace mock weather data fetch with real API call

const handleSubmit = async (value: string) => {
  setDemoStep("checking");

  const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/chat`, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${apiKey}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ message: value })
  });

  const data = await response.json();

  // Process workflow_steps
  data.workflow_steps.forEach(step => {
    setDemoStep(step.step);
  });

  // Display tool results
  if (data.tool_calls?.length > 0) {
    setResultData(data.tool_calls[0].result);
    setShowResult(true);
  }
};
```

#### Phase 2: Real-Time Discovery (Week 2)
1. ✅ Implement `GET /api/discovery/stream` SSE endpoint
2. ✅ Connect LiveDiscoveryHUD to real event stream
3. ✅ Test Firecrawl → MCP → Agent event flow

**UI Changes Required:**
```typescript
// src/components/LiveDiscoveryHUD.tsx
// Replace DEMO_LOGS with SSE connection

useEffect(() => {
  const eventSource = new EventSource(
    `${process.env.NEXT_PUBLIC_API_URL}/api/discovery/stream?conversation_id=${conversationId}`
  );

  eventSource.onmessage = (event) => {
    if (event.data === '[DONE]') {
      eventSource.close();
      return;
    }

    const logEntry = JSON.parse(event.data);
    setLogs(prev => [...prev, logEntry]);
  };

  return () => eventSource.close();
}, [conversationId]);
```

#### Phase 3: MCP Forge (Week 3)
1. ✅ Implement `POST /api/forge/generate` endpoint
2. ✅ Integrate Firecrawl for API doc scraping
3. ✅ Connect MCPForge to dynamic code generation

**UI Changes Required:**
```typescript
// src/components/MCPForge.tsx
// Replace SAMPLE_MARKDOWN and GENERATED_TS with API data

const generateTool = async (url: string) => {
  const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/forge/generate`, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${apiKey}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ source_url: url })
  });

  const data = await response.json();
  setMarkdownContent(data.documentation.markdown);
  setGeneratedCode(data.generated_code.typescript);

  // Stream discovery logs
  data.discovery_logs.forEach((log, index) => {
    setTimeout(() => setLogs(prev => [...prev, log]), index * 200);
  });
};
```

#### Phase 4: Marketplace & Governance (Week 4)
1. ✅ Enhance `GET /tools` with new fields
2. ✅ Implement `GET /api/governance/verified-tools`
3. ✅ Update ToolMarketplace and TrustGovernanceLedger components

**UI Changes Required:**
```typescript
// src/components/ToolMarketplace.tsx
// Remove hardcoded MARKETPLACE_TOOLS

const fetchTools = async () => {
  const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/tools`, {
    headers: { 'Authorization': `Bearer ${apiKey}` }
  });
  const data = await response.json();
  setTools(data.tools);
};

useEffect(() => { fetchTools(); }, []);
```

#### Phase 5: Actions & API Keys (Week 5)
1. ✅ Implement `GET /api/actions`
2. ✅ Implement API key management endpoints
3. ✅ Update ActionCenter and ApiAccess components

---

## Error Handling Requirements

### Standard Error Codes

| HTTP Status | Error Code | Description | UI Behavior |
|-------------|-----------|-------------|-------------|
| 400 | `INVALID_REQUEST` | Malformed request | Show error toast |
| 401 | `UNAUTHORIZED` | Invalid/missing API key | Redirect to login |
| 403 | `FORBIDDEN` | Insufficient permissions | Show permission error |
| 404 | `NOT_FOUND` | Resource not found | Show "not found" message |
| 429 | `RATE_LIMITED` | Too many requests | Show retry countdown |
| 500 | `INTERNAL_ERROR` | Server error | Show generic error + support link |
| 503 | `SERVICE_UNAVAILABLE` | External API down | Show service status |

### Error Response Format

```json
{
  "success": false,
  "error": {
    "code": "RATE_LIMITED",
    "message": "Rate limit exceeded. Try again in 60 seconds.",
    "details": {
      "retry_after": 60,
      "limit": 100,
      "window": "1 minute"
    },
    "request_id": "req_550e8400"
  }
}
```

### UI Error Handling Template

```typescript
try {
  const response = await fetch(url, options);
  const data = await response.json();

  if (!data.success) {
    throw new Error(data.error.message);
  }

  return data;
} catch (error) {
  console.error('API Error:', error);

  // Show user-friendly error
  toast.error(error.message || 'Something went wrong');

  // Optional: Log to error tracking service
  Sentry.captureException(error);
}
```

---

## Performance & Scalability

### Response Time Requirements

| Endpoint | Target P95 | Max Acceptable |
|----------|-----------|----------------|
| `POST /chat` | < 2s | < 5s |
| `GET /tools` | < 200ms | < 500ms |
| `GET /tools/search` | < 300ms | < 1s |
| `POST /tools/execute` | < 1s | < 3s |
| `POST /api/forge/generate` | < 5s | < 15s |
| `GET /api/actions` | < 200ms | < 500ms |

### Caching Strategy

**1. Tool Listings**
- Cache `GET /tools` for 5 minutes
- Cache `GET /tools/search` for 2 minutes (per unique query)
- Invalidate on tool updates

**2. Tool Executions**
- Cache results for 1 minute (if deterministic)
- Key: `{tool_name}:{hash(params)}`

**3. Discovery Logs**
- No caching (real-time streaming)

### Database Indexes

**Required MongoDB Indexes:**
```javascript
// Tools collection
db.tools.createIndex({ "name": 1 }, { unique: true });
db.tools.createIndex({ "status": 1 });
db.tools.createIndex({ "category": 1 });
db.tools.createIndex({ "tags": 1 });
db.tools.createIndex({ "verified": 1 });
db.tools.createIndex({ "created_at": -1 });

// For vector search (if not using MongoDB Atlas Vector Search)
db.tools.createIndex({ "embedding": "2dsphere" });

// Actions collection
db.actions.createIndex({ "conversation_id": 1, "timestamp": -1 });
db.actions.createIndex({ "timestamp": -1 });

// Conversations collection
db.conversations.createIndex({ "conversation_id": 1 }, { unique: true });
db.conversations.createIndex({ "updated_at": -1 });

// API Keys collection
db.api_keys.createIndex({ "key_hash": 1 }, { unique: true });
db.api_keys.createIndex({ "user_id": 1 });
db.api_keys.createIndex({ "expires_at": 1 });
```

---

## Testing Requirements

### API Contract Tests

Each endpoint must have:
1. ✅ Request validation tests
2. ✅ Response schema validation tests
3. ✅ Error handling tests
4. ✅ Authentication tests
5. ✅ Rate limiting tests

### Sample Test Cases

**Example: POST /chat**
```python
def test_chat_endpoint_success():
    response = client.post("/chat",
        json={"message": "Get weather for NYC"},
        headers={"Authorization": f"Bearer {valid_key}"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] == True
    assert "conversation_id" in data
    assert "workflow_steps" in data
    assert len(data["workflow_steps"]) > 0

def test_chat_endpoint_unauthorized():
    response = client.post("/chat",
        json={"message": "test"}
    )
    assert response.status_code == 401
```

---

## Appendix

### A. Environment Variables Checklist

**Backend (API) Requirements:**
```bash
# Required
VOYAGE_API_KEY=your_key_here
FIRECRAWL_API_KEY=your_key_here
MONGODB_URI=mongodb://localhost:27017/agent_db

# Optional
OPENROUTER_API_KEY=your_key_here
OPENWEATHER_API_KEY=your_key_here
REDIS_URL=redis://localhost:6379  # For caching
SENTRY_DSN=your_sentry_dsn         # For error tracking
```

**Frontend (UI) Requirements:**
```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
OPENWEATHER_API_KEY=your_key_here  # Only needed for client-side demo
```

### B. Sample Tool Data for Testing

Populate MongoDB with these sample tools to match UI placeholders:

```javascript
// 1. OpenWeatherMap
{
  id: "get_current_weather",
  name: "get_current_weather",
  description: "Get current weather data for any city worldwide",
  status: "PROD-READY",
  source_url: "https://api.openweathermap.org/docs",
  preview_snippet: "get_current_weather(q: string, units?: 'metric' | 'imperial'): Promise<WeatherData>",
  category: "weather",
  tags: ["weather", "api", "data"],
  verified: true,
  usage_count: 1250,
  // ... rest of fields
}

// 2. Instagram Basic Display
{
  id: "get_instagram_media",
  name: "get_instagram_media",
  description: "Fetch recent media posts from Instagram account",
  status: "BETA",
  source_url: "https://developers.facebook.com/docs/instagram-basic-display-api",
  preview_snippet: "get_instagram_media(user_id: string, limit?: number): Promise<Media[]>",
  category: "social",
  tags: ["instagram", "social", "media"],
  verified: true,
  usage_count: 890,
  // ... rest of fields
}

// 3. Stripe Payments
{
  id: "create_stripe_charge",
  name: "create_stripe_charge",
  description: "Process payment via Stripe",
  status: "PROD-READY",
  source_url: "https://stripe.com/docs/api/charges/create",
  preview_snippet: "create_stripe_charge(amount: number, currency: string, source: string): Promise<Charge>",
  category: "payments",
  tags: ["stripe", "payments", "billing"],
  verified: true,
  usage_count: 2340,
  // ... rest of fields
}

// 4. Custom Internal API
{
  id: "get_user_profile",
  name: "get_user_profile",
  description: "Fetch user profile from internal database",
  status: "PROD-READY",
  source_url: "https://api.yourcompany.com/docs/users",
  preview_snippet: "get_user_profile(user_id: string): Promise<UserProfile>",
  category: "internal",
  tags: ["users", "profile", "internal"],
  verified: true,
  usage_count: 5670,
  // ... rest of fields
}

// 5. GitHub API
{
  id: "create_github_issue",
  name: "create_github_issue",
  description: "Create issue in GitHub repository",
  status: "PROD-READY",
  source_url: "https://docs.github.com/en/rest/issues",
  preview_snippet: "create_github_issue(repo: string, title: string, body: string): Promise<Issue>",
  category: "developer-tools",
  tags: ["github", "issues", "development"],
  verified: true,
  usage_count: 1120,
  // ... rest of fields
}

// 6. Slack Notifications
{
  id: "send_slack_message",
  name: "send_slack_message",
  description: "Send message to Slack channel",
  status: "PROD-READY",
  source_url: "https://api.slack.com/methods/chat.postMessage",
  preview_snippet: "send_slack_message(channel: string, text: string): Promise<MessageResponse>",
  category: "communication",
  tags: ["slack", "notifications", "messaging"],
  verified: true,
  usage_count: 3450,
  // ... rest of fields
}
```

---

## Contact & Support

**Document Owner:** Frontend Team
**Backend Contact:** [Backend Team Lead Email]
**Last Updated:** 2026-01-31
**Review Cycle:** Weekly during implementation

**Questions?** Create an issue in the repo or reach out to #universal-adapter-dev on Slack

---

**Next Steps:**
1. Backend team reviews this document
2. Schedule kickoff meeting to discuss timeline
3. Create tracking tickets for each endpoint
4. Set up staging environment for integration testing
5. Begin Phase 1 implementation

**Success Criteria:**
- [ ] All P0 endpoints implemented and tested
- [ ] UI successfully removes all placeholder data
- [ ] End-to-end flow works: Command → Discovery → Forge → Execute → Result
- [ ] Real-time log streaming functional
- [ ] Tool marketplace populated with real data
- [ ] API authentication working
- [ ] Error handling tested across all endpoints
