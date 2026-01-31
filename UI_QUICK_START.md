# UI Agent Quick Start Guide

**Quick reference for integrating with the AI Agent API**

---

## 🚀 Quick Start

### Base URL
```
http://localhost:8001
```

### Essential Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/chat` | POST | Send messages to agent |
| `/tools/search` | GET | Search for tools |
| `/tools` | GET | List all tools |
| `/tools/{name}/execute` | POST | Execute a tool |

---

## 💬 Chat with Agent

**The main endpoint you'll use most**

```typescript
POST /chat
```

**Request:**
```json
{
  "message": "Get the current Bitcoin price",
  "conversation_id": "optional-id"
}
```

**Response:**
```json
{
  "response": "Bitcoin is currently $78,055 USD",
  "conversation_id": "20260131_191432",
  "tool_calls": [...]
}
```

**Example:**
```typescript
async function chat(message: string) {
  const response = await fetch('http://localhost:8001/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message })
  });
  return await response.json();
}

// Usage
const result = await chat("Get Bitcoin price");
console.log(result.response);
```

---

## 🔍 Search Tools

**Find relevant tools semantically**

```typescript
GET /tools/search?q={query}&limit={limit}
```

**Example:**
```typescript
async function searchTools(query: string) {
  const response = await fetch(
    `http://localhost:8001/tools/search?q=${encodeURIComponent(query)}&limit=5`
  );
  return await response.json();
}

// Usage
const tools = await searchTools("cryptocurrency price");
console.log(tools); // { count: 1, tools: [...] }
```

---

## 🛠️ Execute Tool Directly

**Run a specific tool with parameters**

```typescript
POST /tools/{tool_name}/execute
```

**Example:**
```typescript
async function executeToolDirectly(toolName: string, params: any) {
  const response = await fetch(
    `http://localhost:8001/tools/${toolName}/execute`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params)
    }
  );
  return await response.json();
}

// Usage
const price = await executeToolDirectly('get_crypto_price', {
  symbol: 'BTCUSD'
});
console.log(price.result.price); // 78055
```

---

## 📋 List All Tools

**Get all available tools**

```typescript
GET /tools
```

**Example:**
```typescript
async function listTools() {
  const response = await fetch('http://localhost:8001/tools');
  return await response.json();
}

// Usage
const { count, tools } = await listTools();
console.log(`Found ${count} tools`);
```

---

## 🎯 Complete React Example

```tsx
import { useState } from 'react';

interface Message {
  role: 'user' | 'assistant';
  content: string;
}

function ChatUI() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);

  const sendMessage = async () => {
    if (!input.trim()) return;

    // Add user message
    const userMessage: Message = { role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      // Call agent API
      const response = await fetch('http://localhost:8001/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: input })
      });

      const data = await response.json();

      // Add assistant response
      const assistantMessage: Message = {
        role: 'assistant',
        content: data.response
      };
      setMessages(prev => [...prev, assistantMessage]);

    } catch (error) {
      console.error('Error:', error);
      const errorMessage: Message = {
        role: 'assistant',
        content: 'Sorry, an error occurred.'
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="chat-container">
      {/* Messages */}
      <div className="messages">
        {messages.map((msg, idx) => (
          <div key={idx} className={`message ${msg.role}`}>
            {msg.content}
          </div>
        ))}
        {loading && <div className="loading">Thinking...</div>}
      </div>

      {/* Input */}
      <div className="input-area">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
          placeholder="Ask me anything..."
          disabled={loading}
        />
        <button onClick={sendMessage} disabled={loading}>
          Send
        </button>
      </div>
    </div>
  );
}

export default ChatUI;
```

---

## 🎨 Complete Vue Example

```vue
<template>
  <div class="chat-container">
    <!-- Messages -->
    <div class="messages">
      <div
        v-for="(msg, idx) in messages"
        :key="idx"
        :class="['message', msg.role]"
      >
        {{ msg.content }}
      </div>
      <div v-if="loading" class="loading">Thinking...</div>
    </div>

    <!-- Input -->
    <div class="input-area">
      <input
        v-model="input"
        @keypress.enter="sendMessage"
        placeholder="Ask me anything..."
        :disabled="loading"
      />
      <button @click="sendMessage" :disabled="loading">Send</button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue';

interface Message {
  role: 'user' | 'assistant';
  content: string;
}

const messages = ref<Message[]>([]);
const input = ref('');
const loading = ref(false);

const sendMessage = async () => {
  if (!input.value.trim()) return;

  // Add user message
  messages.value.push({
    role: 'user',
    content: input.value
  });

  const userMessage = input.value;
  input.value = '';
  loading.value = true;

  try {
    const response = await fetch('http://localhost:8001/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: userMessage })
    });

    const data = await response.json();

    messages.value.push({
      role: 'assistant',
      content: data.response
    });

  } catch (error) {
    console.error('Error:', error);
    messages.value.push({
      role: 'assistant',
      content: 'Sorry, an error occurred.'
    });
  } finally {
    loading.value = false;
  }
};
</script>

<style scoped>
.chat-container {
  display: flex;
  flex-direction: column;
  height: 100vh;
  max-width: 800px;
  margin: 0 auto;
}

.messages {
  flex: 1;
  overflow-y: auto;
  padding: 20px;
}

.message {
  margin: 10px 0;
  padding: 10px 15px;
  border-radius: 8px;
}

.message.user {
  background: #007bff;
  color: white;
  align-self: flex-end;
  margin-left: auto;
  max-width: 70%;
}

.message.assistant {
  background: #f1f3f5;
  color: #333;
  align-self: flex-start;
  max-width: 70%;
}

.input-area {
  display: flex;
  padding: 20px;
  border-top: 1px solid #ddd;
}

.input-area input {
  flex: 1;
  padding: 10px;
  border: 1px solid #ddd;
  border-radius: 4px;
  margin-right: 10px;
}

.input-area button {
  padding: 10px 20px;
  background: #007bff;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}

.input-area button:disabled {
  background: #ccc;
  cursor: not-allowed;
}

.loading {
  color: #666;
  font-style: italic;
}
</style>
```

---

## 🔧 TypeScript Types

```typescript
// types.ts

export interface ChatRequest {
  message: string;
  conversation_id?: string;
  model?: string;
  stream?: boolean;
}

export interface ChatResponse {
  response: string;
  conversation_id: string;
  model: string;
  tool_calls?: ToolCall[];
}

export interface ToolCall {
  name: string;
  arguments: Record<string, any>;
  result: any;
}

export interface Tool {
  _id: string;
  name: string;
  description: string;
  parameters: {
    type: 'object';
    properties: Record<string, any>;
    required: string[];
  };
  created_at: string;
  similarity_score?: number;
}

export interface ToolSearchResponse {
  success: boolean;
  query: string;
  count: number;
  tools: Tool[];
}

export interface ToolExecuteResponse {
  success: boolean;
  result: any;
}

export interface ErrorResponse {
  success: false;
  error: string;
  type: string;
}
```

---

## 📡 API Client Wrapper

```typescript
// api-client.ts

const API_BASE = 'http://localhost:8001';

export class AgentAPI {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE) {
    this.baseUrl = baseUrl;
  }

  async chat(message: string, conversationId?: string): Promise<ChatResponse> {
    const response = await fetch(`${this.baseUrl}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, conversation_id: conversationId })
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    return await response.json();
  }

  async searchTools(query: string, limit: number = 10): Promise<ToolSearchResponse> {
    const response = await fetch(
      `${this.baseUrl}/tools/search?q=${encodeURIComponent(query)}&limit=${limit}`
    );

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    return await response.json();
  }

  async listTools(): Promise<{ success: boolean; count: number; tools: Tool[] }> {
    const response = await fetch(`${this.baseUrl}/tools`);

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    return await response.json();
  }

  async executeTool(toolName: string, params: Record<string, any>): Promise<ToolExecuteResponse> {
    const response = await fetch(`${this.baseUrl}/tools/${toolName}/execute`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params)
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    return await response.json();
  }

  async getTool(toolName: string): Promise<Tool> {
    const response = await fetch(`${this.baseUrl}/tools/${toolName}`);

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const data = await response.json();
    return data.tool;
  }
}

// Usage
export const api = new AgentAPI();

// In your components:
// import { api } from './api-client';
// const result = await api.chat("Get Bitcoin price");
```

---

## 🎯 Common Use Cases

### 1. Simple Chat Interface
```typescript
const result = await api.chat("Get the current Bitcoin price");
console.log(result.response);
```

### 2. Show Available Tools
```typescript
const { tools, count } = await api.listTools();
console.log(`${count} tools available:`);
tools.forEach(tool => console.log(`- ${tool.name}: ${tool.description}`));
```

### 3. Search and Execute
```typescript
// Search for crypto tools
const search = await api.searchTools("cryptocurrency price");
const tool = search.tools[0];

// Execute the tool
const result = await api.executeTool(tool.name, { symbol: "BTCUSD" });
console.log(`Bitcoin: $${result.result.price}`);
```

### 4. Conversation with Context
```typescript
let conversationId: string | undefined;

// First message
const msg1 = await api.chat("Get Bitcoin price");
conversationId = msg1.conversation_id;

// Follow-up with context
const msg2 = await api.chat("What about Ethereum?", conversationId);
console.log(msg2.response);
```

---

## 🚨 Error Handling

```typescript
try {
  const result = await api.chat("Get Bitcoin price");
  console.log(result.response);
} catch (error) {
  if (error instanceof Error) {
    console.error('Error:', error.message);

    // Show user-friendly message
    alert('Failed to communicate with the agent. Please try again.');
  }
}
```

---

## 📊 Display Tool Results

```tsx
interface ToolResultProps {
  toolCall: ToolCall;
}

function ToolResult({ toolCall }: ToolResultProps) {
  return (
    <div className="tool-result">
      <h4>🔧 {toolCall.name}</h4>
      <div className="arguments">
        <strong>Input:</strong>
        <pre>{JSON.stringify(toolCall.arguments, null, 2)}</pre>
      </div>
      <div className="result">
        <strong>Output:</strong>
        <pre>{JSON.stringify(toolCall.result, null, 2)}</pre>
      </div>
    </div>
  );
}

// Usage
function ChatMessage({ message }: { message: ChatResponse }) {
  return (
    <div>
      <p>{message.response}</p>
      {message.tool_calls?.map((call, idx) => (
        <ToolResult key={idx} toolCall={call} />
      ))}
    </div>
  );
}
```

---

## 🎨 Styling Examples

```css
/* Chat Container */
.chat-container {
  display: flex;
  flex-direction: column;
  height: 100vh;
  background: #f5f5f5;
}

/* Messages Area */
.messages {
  flex: 1;
  overflow-y: auto;
  padding: 20px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

/* Individual Message */
.message {
  padding: 12px 16px;
  border-radius: 12px;
  max-width: 70%;
  word-wrap: break-word;
}

.message.user {
  align-self: flex-end;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border-bottom-right-radius: 4px;
}

.message.assistant {
  align-self: flex-start;
  background: white;
  color: #333;
  border: 1px solid #e0e0e0;
  border-bottom-left-radius: 4px;
}

/* Tool Results */
.tool-result {
  margin-top: 10px;
  padding: 12px;
  background: #f8f9fa;
  border-left: 3px solid #007bff;
  border-radius: 4px;
}

.tool-result h4 {
  margin: 0 0 8px 0;
  color: #007bff;
  font-size: 14px;
}

.tool-result pre {
  background: #fff;
  padding: 8px;
  border-radius: 4px;
  overflow-x: auto;
  font-size: 12px;
}

/* Input Area */
.input-area {
  display: flex;
  padding: 20px;
  background: white;
  border-top: 1px solid #e0e0e0;
  gap: 10px;
}

.input-area input {
  flex: 1;
  padding: 12px 16px;
  border: 2px solid #e0e0e0;
  border-radius: 24px;
  font-size: 14px;
  outline: none;
  transition: border-color 0.2s;
}

.input-area input:focus {
  border-color: #667eea;
}

.input-area button {
  padding: 12px 24px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border: none;
  border-radius: 24px;
  font-weight: 600;
  cursor: pointer;
  transition: transform 0.2s, opacity 0.2s;
}

.input-area button:hover:not(:disabled) {
  transform: scale(1.05);
}

.input-area button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

/* Loading Indicator */
.loading {
  align-self: flex-start;
  padding: 12px 16px;
  background: white;
  border-radius: 12px;
  color: #666;
  font-style: italic;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

.loading::after {
  content: '...';
  animation: pulse 1.5s infinite;
}
```

---

## ✅ Testing

```typescript
// Simple test
async function testAPI() {
  try {
    console.log('Testing chat endpoint...');
    const chat = await api.chat('Hello!');
    console.log('✅ Chat:', chat.response);

    console.log('Testing tool search...');
    const tools = await api.searchTools('crypto');
    console.log('✅ Found tools:', tools.count);

    console.log('Testing tool list...');
    const list = await api.listTools();
    console.log('✅ Total tools:', list.count);

    console.log('All tests passed! 🎉');
  } catch (error) {
    console.error('❌ Test failed:', error);
  }
}

testAPI();
```

---

## 📝 Notes

- **CORS**: Make sure your API allows CORS from your UI domain
- **Error Handling**: Always wrap API calls in try/catch
- **Loading States**: Show loading indicators during API calls
- **Conversation ID**: Store it to maintain context across messages
- **Typing Indicators**: Show when agent is "thinking"

---

**Happy Building! 🚀**

For full API documentation, see `API_DOCUMENTATION.md`
