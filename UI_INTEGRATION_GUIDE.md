# Universal Adapter UI Integration Guide

**Complete integration guide for Next.js 16.1.6 frontend with Enhanced Backend API v2.0**

**Date:** 2026-01-31
**Backend Version:** 2.0.0
**Target:** Frontend Development Team

---

## Table of Contents

1. [Overview](#overview)
2. [Environment Setup](#environment-setup)
3. [Enhanced TypeScript Types](#enhanced-typescript-types)
4. [API Client with Enhanced Features](#api-client-with-enhanced-features)
5. [Component Integration](#component-integration)
   - [CommandCenter](#1-commandcenter)
   - [LiveDiscoveryHUD](#2-livediscoveryhud)
   - [MCPForge](#3-mcpforge)
   - [ToolMarketplace](#4-toolmarketplace)
   - [ResultCard](#5-resultcard)
   - [ActionCenter](#6-actioncenter)
   - [TrustGovernanceLedger](#7-trustgovernanceledger)
6. [State Management](#state-management)
7. [Error Handling Patterns](#error-handling-patterns)
8. [Testing](#testing)

---

## Overview

This guide shows how to integrate each Universal Adapter UI component with the enhanced backend API that implements all P0 and P1 requirements from `BACKEND_API_REQUIREMENTS.md`.

### What's New in Backend v2.0

✅ **Workflow Steps** - Real-time progress tracking (checking → discovering → forging → done)
✅ **Enhanced Tool Schema** - 8 new fields for rich UI display
✅ **Server-Sent Events** - Real-time discovery log streaming
✅ **Tool Execution** - Detailed execution metadata and logs
✅ **Action Feed** - Integrated action logging
✅ **Governance** - Verification and trust metadata

---

## Environment Setup

### 1. Install Dependencies

```bash
npm install axios @tanstack/react-query zustand
# or
pnpm add axios @tanstack/react-query zustand
```

### 2. Environment Variables

Create `.env.local`:

```bash
NEXT_PUBLIC_API_URL=http://localhost:8001
NEXT_PUBLIC_API_VERSION=2.0.0
```

### 3. Verify Backend is Running

```bash
curl http://localhost:8001/health
# Should return: {"status":"healthy","service":"universal-adapter-api","version":"2.0.0"}
```

---

## Enhanced TypeScript Types

Create `types/api.ts`:

```typescript
// ============================================
// Enhanced API Types for Backend v2.0
// ============================================

/** Workflow step in agent execution */
export interface WorkflowStep {
  step: 'checking' | 'discovering' | 'forging' | 'done';
  status: 'completed' | 'failed';
  duration_ms: number;
  message: string;
}

/** Tool call with execution results */
export interface ToolCall {
  id: string;
  name: string;
  arguments: Record<string, any>;
  result: Record<string, any> | null;
  execution_time_ms: number;
  status: 'success' | 'error';
}

/** Action log entry for action feed */
export interface ActionLog {
  id: string;
  title: string;
  detail: string;
  status: 'success' | 'pending' | 'error';
  timestamp: string;
  github_pr_url?: string;
  tool_name?: string;
  execution_id?: string;
}

/** Chat response metadata */
export interface ChatMetadata {
  total_duration_ms: number;
  tokens_used: number;
  cost_usd: number;
}

/** Enhanced chat request */
export interface ChatRequest {
  message: string;
  conversation_id?: string;
  model?: string;
  stream?: boolean;
  context?: {
    ui_mode?: string;
    view?: string;
  };
}

/** Enhanced chat response with workflow tracking */
export interface ChatResponse {
  success: boolean;
  response: string;
  conversation_id: string;
  model: string;
  workflow_steps: WorkflowStep[];
  tool_calls: ToolCall[];
  actions_logged: ActionLog[];
  metadata: ChatMetadata;
}

/** Enhanced tool with UI fields */
export interface EnhancedTool {
  id: string;
  name: string;
  description: string;
  status: 'PROD-READY' | 'BETA' | 'DEPRECATED';
  source_url?: string;
  preview_snippet?: string;
  category: string;
  tags: string[];
  verified: boolean;
  usage_count: number;
  mux_playback_id?: string;
  parameters: {
    type: 'object';
    properties: Record<string, any>;
    required?: string[];
  };
  code?: string;
  created_at?: string;
  similarity_score?: number;
}

/** Discovery log entry for SSE streaming */
export interface DiscoveryLog {
  id?: string;
  timestamp: string;
  source: 'firecrawl' | 'mcp' | 'agent' | 'system';
  message: string;
  level: 'info' | 'success' | 'error' | 'warning';
  metadata?: Record<string, any>;
}

/** Tool execution response */
export interface ToolExecuteResponse {
  success: boolean;
  tool_name: string;
  execution_id: string;
  result: Record<string, any>;
  execution_metadata: {
    started_at: string;
    completed_at: string;
    duration_ms: number;
    api_calls_made: number;
    cached: boolean;
  };
  logs: Array<{
    timestamp: string;
    message: string;
  }>;
}

/** Forge generation request */
export interface ForgeRequest {
  source_url: string;
  force_regenerate?: boolean;
}

/** Forge generation response */
export interface ForgeResponse {
  success: boolean;
  tool_id: string;
  documentation: {
    markdown: string;
    endpoints_found: number;
    auth_params: string[];
    base_url: string;
  };
  generated_code: {
    typescript: string;
    language: string;
    framework: string;
  };
  discovery_logs: DiscoveryLog[];
  metadata: {
    generation_time_ms: number;
    firecrawl_pages_crawled: number;
    tokens_used: number;
  };
}

/** Action from action feed */
export interface Action {
  id: string;
  conversation_id?: string;
  title: string;
  detail: string;
  status: 'success' | 'pending' | 'error';
  timestamp: string;
  github_pr_url?: string;
  tool_name?: string;
  execution_id?: string;
}

/** Verified tool with governance */
export interface VerifiedTool extends EnhancedTool {
  verification: {
    verified: boolean;
    verified_at: string;
    verified_by: string;
    trust_score: number;
    security_scan_passed: boolean;
    last_audit: string;
  };
  governance: {
    approval_required: boolean;
    allowed_users: string[];
    rate_limit_per_minute: number;
    cost_per_execution: number;
  };
}

/** Error response */
export interface APIError {
  success: false;
  error: {
    code: string;
    message: string;
    details?: Record<string, any>;
  };
}
```

---

## API Client with Enhanced Features

Create `lib/api-client.ts`:

```typescript
import axios, { AxiosInstance, AxiosError } from 'axios';
import type {
  ChatRequest,
  ChatResponse,
  EnhancedTool,
  ToolExecuteResponse,
  ForgeRequest,
  ForgeResponse,
  Action,
  VerifiedTool,
  APIError,
  DiscoveryLog
} from '@/types/api';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001';

export class UniversalAdapterAPI {
  private client: AxiosInstance;

  constructor(baseURL: string = API_BASE) {
    this.client = axios.create({
      baseURL,
      headers: {
        'Content-Type': 'application/json',
      },
      timeout: 30000, // 30 seconds
    });

    // Add response interceptor for error handling
    this.client.interceptors.response.use(
      (response) => response,
      (error: AxiosError<APIError>) => {
        if (error.response?.data) {
          throw new APIClientError(
            error.response.data.error.message,
            error.response.data.error.code,
            error.response.status
          );
        }
        throw error;
      }
    );
  }

  // ============================================
  // Chat & Agent
  // ============================================

  /**
   * Send message to agent with enhanced workflow tracking
   */
  async chat(request: ChatRequest): Promise<ChatResponse> {
    const { data } = await this.client.post<ChatResponse>('/chat', request);
    return data;
  }

  // ============================================
  // Discovery Stream (SSE)
  // ============================================

  /**
   * Open Server-Sent Events connection for discovery logs
   */
  openDiscoveryStream(
    conversationId?: string,
    onMessage?: (log: DiscoveryLog) => void,
    onDone?: () => void,
    onError?: (error: Error) => void
  ): EventSource {
    const url = conversationId
      ? `${API_BASE}/api/discovery/stream?conversation_id=${conversationId}`
      : `${API_BASE}/api/discovery/stream`;

    const eventSource = new EventSource(url);

    eventSource.onmessage = (event) => {
      if (event.data === '[DONE]') {
        eventSource.close();
        onDone?.();
        return;
      }

      try {
        const log: DiscoveryLog = JSON.parse(event.data);
        onMessage?.(log);
      } catch (error) {
        console.error('Failed to parse discovery log:', error);
      }
    };

    eventSource.onerror = (error) => {
      console.error('Discovery stream error:', error);
      onError?.(new Error('Discovery stream connection failed'));
      eventSource.close();
    };

    return eventSource;
  }

  // ============================================
  // Tools
  // ============================================

  /**
   * List all tools with enhanced UI fields
   */
  async listTools(limit = 50, skip = 0): Promise<EnhancedTool[]> {
    const { data } = await this.client.get<EnhancedTool[]>('/tools', {
      params: { limit, skip },
    });
    return data;
  }

  /**
   * Search tools semantically
   */
  async searchTools(query: string, limit = 10): Promise<{
    query: string;
    count: number;
    tools: EnhancedTool[];
  }> {
    const { data } = await this.client.get('/tools/search', {
      params: { q: query, limit },
    });
    return data;
  }

  /**
   * Get specific tool by name
   */
  async getTool(name: string): Promise<EnhancedTool> {
    const { data } = await this.client.get<EnhancedTool>(`/tools/${name}`);
    return data;
  }

  /**
   * Delete a tool
   */
  async deleteTool(name: string): Promise<{ success: boolean; message: string }> {
    const { data } = await this.client.delete(`/tools/${name}`);
    return data;
  }

  /**
   * Execute a tool with parameters
   */
  async executeTool(
    toolName: string,
    params: Record<string, any>
  ): Promise<ToolExecuteResponse> {
    const { data } = await this.client.post<ToolExecuteResponse>(
      `/tools/${toolName}/execute`,
      params
    );
    return data;
  }

  // ============================================
  // Forge (Tool Generation)
  // ============================================

  /**
   * Generate MCP tool from API documentation URL
   */
  async forgeGenerate(request: ForgeRequest): Promise<ForgeResponse> {
    const { data } = await this.client.post<ForgeResponse>(
      '/api/forge/generate',
      request
    );
    return data;
  }

  // ============================================
  // Actions
  // ============================================

  /**
   * Get action feed with optional conversation filter
   */
  async getActions(
    conversationId?: string,
    limit = 50,
    offset = 0
  ): Promise<Action[]> {
    const { data } = await this.client.get<Action[]>('/api/actions', {
      params: { conversation_id: conversationId, limit, offset },
    });
    return data;
  }

  // ============================================
  // Governance
  // ============================================

  /**
   * Get verified tools with governance metadata
   */
  async getVerifiedTools(): Promise<VerifiedTool[]> {
    const { data } = await this.client.get<VerifiedTool[]>(
      '/api/governance/verified-tools'
    );
    return data;
  }

  // ============================================
  // Health Check
  // ============================================

  async healthCheck(): Promise<{
    status: string;
    service: string;
    version: string;
  }> {
    const { data } = await this.client.get('/health');
    return data;
  }
}

/** Custom API error class */
export class APIClientError extends Error {
  constructor(
    message: string,
    public code: string,
    public status?: number
  ) {
    super(message);
    this.name = 'APIClientError';
  }
}

/** Singleton API client instance */
export const api = new UniversalAdapterAPI();
```

---

## Component Integration

### 1. CommandCenter

**Component Purpose:** Main chat interface with workflow step animations

**Backend Endpoint:** `POST /chat`

**Key Features:**
- Workflow step progression (checking → discovering → forging → done)
- Tool call display
- Result cards
- Action logging

**Implementation:**

```typescript
// components/CommandCenter.tsx
'use client';

import { useState, useRef, useEffect } from 'react';
import { api } from '@/lib/api-client';
import type { ChatResponse, WorkflowStep } from '@/types/api';
import { ResultCard } from './ResultCard';
import { WorkflowProgress } from './WorkflowProgress';

type DemoStep = 'idle' | 'checking' | 'discovering' | 'forging' | 'done';

export function CommandCenter() {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<ChatResponse[]>([]);
  const [loading, setLoading] = useState(false);
  const [currentStep, setCurrentStep] = useState<DemoStep>('idle');
  const [conversationId, setConversationId] = useState<string>();
  const inputRef = useRef<HTMLInputElement>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const userMessage = input;
    setInput('');
    setLoading(true);
    setCurrentStep('checking');

    try {
      // Call enhanced chat API
      const response = await api.chat({
        message: userMessage,
        conversation_id: conversationId,
        context: {
          ui_mode: 'command_center',
        },
      });

      // Store conversation ID for context
      setConversationId(response.conversation_id);

      // Animate through workflow steps
      for (const step of response.workflow_steps) {
        setCurrentStep(step.step as DemoStep);
        // Wait for step duration (for animation)
        await new Promise((resolve) =>
          setTimeout(resolve, Math.min(step.duration_ms, 1000))
        );
      }

      // Add message to history
      setMessages((prev) => [...prev, response]);
      setCurrentStep('done');

      // Reset to idle after a moment
      setTimeout(() => setCurrentStep('idle'), 1000);
    } catch (error) {
      console.error('Chat error:', error);
      alert('Failed to process command. Please try again.');
      setCurrentStep('idle');
    } finally {
      setLoading(false);
      inputRef.current?.focus();
    }
  };

  return (
    <div className="command-center">
      {/* Workflow Progress Indicator */}
      <WorkflowProgress currentStep={currentStep} loading={loading} />

      {/* Messages History */}
      <div className="messages-container">
        {messages.map((msg, idx) => (
          <div key={idx} className="message-group">
            {/* Agent Response */}
            <div className="assistant-message">
              {msg.response}
            </div>

            {/* Tool Calls (if any) */}
            {msg.tool_calls.map((toolCall, tcIdx) => (
              <ResultCard
                key={tcIdx}
                toolName={toolCall.name}
                arguments={toolCall.arguments}
                result={toolCall.result}
                executionTime={toolCall.execution_time_ms}
                status={toolCall.status}
              />
            ))}

            {/* Metadata */}
            <div className="message-metadata">
              <span>{msg.metadata.tokens_used} tokens</span>
              <span>{msg.metadata.total_duration_ms}ms</span>
              <span>${msg.metadata.cost_usd.toFixed(4)}</span>
            </div>
          </div>
        ))}
      </div>

      {/* Input Form */}
      <form onSubmit={handleSubmit} className="command-input">
        <input
          ref={inputRef}
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Enter your command..."
          disabled={loading}
          autoFocus
        />
        <button type="submit" disabled={loading || !input.trim()}>
          {loading ? 'Processing...' : 'Execute'}
        </button>
      </form>
    </div>
  );
}
```

**Workflow Progress Component:**

```typescript
// components/WorkflowProgress.tsx
type Step = 'idle' | 'checking' | 'discovering' | 'forging' | 'done';

interface Props {
  currentStep: Step;
  loading: boolean;
}

export function WorkflowProgress({ currentStep, loading }: Props) {
  const steps: Step[] = ['checking', 'discovering', 'forging', 'done'];

  const getStepLabel = (step: Step): string => {
    const labels: Record<Step, string> = {
      idle: '',
      checking: 'Analyzing Command',
      discovering: 'Finding Tools',
      forging: 'Executing',
      done: 'Complete',
    };
    return labels[step];
  };

  const isStepActive = (step: Step): boolean => {
    if (!loading) return false;
    const currentIndex = steps.indexOf(currentStep);
    const stepIndex = steps.indexOf(step);
    return stepIndex <= currentIndex;
  };

  if (!loading && currentStep === 'idle') return null;

  return (
    <div className="workflow-progress">
      {steps.map((step, idx) => (
        <div
          key={step}
          className={`step ${isStepActive(step) ? 'active' : ''} ${
            currentStep === step ? 'current' : ''
          }`}
        >
          <div className="step-indicator">
            {currentStep === step ? (
              <div className="spinner" />
            ) : isStepActive(step) ? (
              '✓'
            ) : (
              idx + 1
            )}
          </div>
          <span className="step-label">{getStepLabel(step)}</span>
        </div>
      ))}
    </div>
  );
}
```

---

### 2. LiveDiscoveryHUD

**Component Purpose:** Real-time discovery log streaming display

**Backend Endpoint:** `GET /api/discovery/stream` (SSE)

**Key Features:**
- Server-Sent Events connection
- Real-time log streaming
- Source-based styling (firecrawl, mcp, agent, system)
- Auto-scroll to bottom

**Implementation:**

```typescript
// components/LiveDiscoveryHUD.tsx
'use client';

import { useState, useEffect, useRef } from 'react';
import { api } from '@/lib/api-client';
import type { DiscoveryLog } from '@/types/api';

interface Props {
  conversationId?: string;
  autoStart?: boolean;
}

export function LiveDiscoveryHUD({ conversationId, autoStart = false }: Props) {
  const [logs, setLogs] = useState<DiscoveryLog[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string>();
  const logsEndRef = useRef<HTMLDivElement>(null);
  const eventSourceRef = useRef<EventSource>();

  // Auto-scroll to bottom when new logs arrive
  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  const startStreaming = () => {
    setIsStreaming(true);
    setError(undefined);
    setLogs([]);

    eventSourceRef.current = api.openDiscoveryStream(
      conversationId,
      // onMessage
      (log) => {
        setLogs((prev) => [...prev, log]);
      },
      // onDone
      () => {
        setIsStreaming(false);
        console.log('Discovery stream completed');
      },
      // onError
      (err) => {
        setError(err.message);
        setIsStreaming(false);
      }
    );
  };

  const stopStreaming = () => {
    eventSourceRef.current?.close();
    setIsStreaming(false);
  };

  useEffect(() => {
    if (autoStart) {
      startStreaming();
    }

    return () => {
      eventSourceRef.current?.close();
    };
  }, [autoStart]);

  const getSourceColor = (source: DiscoveryLog['source']): string => {
    const colors: Record<typeof source, string> = {
      firecrawl: 'text-blue-400',
      mcp: 'text-purple-400',
      agent: 'text-green-400',
      system: 'text-yellow-400',
    };
    return colors[source];
  };

  const getSourceIcon = (source: DiscoveryLog['source']): string => {
    const icons: Record<typeof source, string> = {
      firecrawl: '🌐',
      mcp: '🔧',
      agent: '🤖',
      system: '⚙️',
    };
    return icons[source];
  };

  return (
    <div className="discovery-hud">
      {/* Header with controls */}
      <div className="hud-header">
        <h3>Live Discovery</h3>
        <div className="controls">
          {!isStreaming ? (
            <button onClick={startStreaming} className="btn-start">
              Start Stream
            </button>
          ) : (
            <button onClick={stopStreaming} className="btn-stop">
              Stop Stream
            </button>
          )}
          <button
            onClick={() => setLogs([])}
            className="btn-clear"
            disabled={logs.length === 0}
          >
            Clear
          </button>
        </div>
      </div>

      {/* Status indicator */}
      <div className="status-bar">
        <span className={`status-dot ${isStreaming ? 'streaming' : 'idle'}`} />
        <span>{isStreaming ? 'Streaming...' : 'Idle'}</span>
        <span className="log-count">{logs.length} events</span>
      </div>

      {/* Error display */}
      {error && (
        <div className="error-banner">
          ⚠️ {error}
        </div>
      )}

      {/* Logs display */}
      <div className="logs-container">
        {logs.length === 0 ? (
          <div className="empty-state">
            {isStreaming ? 'Waiting for events...' : 'No events yet'}
          </div>
        ) : (
          logs.map((log, idx) => (
            <div
              key={idx}
              className={`log-entry ${log.level}`}
              data-source={log.source}
            >
              <span className="timestamp">{log.timestamp}</span>
              <span className={`source ${getSourceColor(log.source)}`}>
                {getSourceIcon(log.source)} {log.source}
              </span>
              <span className="message">{log.message}</span>
            </div>
          ))
        )}
        <div ref={logsEndRef} />
      </div>
    </div>
  );
}
```

**Styling (Tailwind CSS):**

```css
/* app/globals.css or component styles */
.discovery-hud {
  @apply border border-gray-700 rounded-lg overflow-hidden bg-gray-900;
}

.hud-header {
  @apply flex items-center justify-between px-4 py-3 bg-gray-800 border-b border-gray-700;
}

.status-bar {
  @apply flex items-center gap-3 px-4 py-2 bg-gray-850 text-sm text-gray-400;
}

.status-dot {
  @apply w-2 h-2 rounded-full;
}

.status-dot.streaming {
  @apply bg-green-500 animate-pulse;
}

.status-dot.idle {
  @apply bg-gray-600;
}

.logs-container {
  @apply max-h-96 overflow-y-auto p-4 font-mono text-sm space-y-2;
}

.log-entry {
  @apply flex items-start gap-3 py-1;
}

.log-entry.error {
  @apply bg-red-900/20 px-2 -mx-2 rounded;
}

.log-entry .timestamp {
  @apply text-gray-500 text-xs whitespace-nowrap;
}

.log-entry .source {
  @apply font-medium whitespace-nowrap;
}

.log-entry .message {
  @apply text-gray-300 flex-1;
}
```

---

### 3. MCPForge

**Component Purpose:** Generate MCP tools from API documentation URLs

**Backend Endpoint:** `POST /api/forge/generate`

**Key Features:**
- URL input for API docs
- Three-panel layout: Documentation | Code | Logs
- Real-time generation progress
- Copy generated code

**Implementation:**

```typescript
// components/MCPForge.tsx
'use client';

import { useState } from 'react';
import { api } from '@/lib/api-client';
import type { ForgeResponse, DiscoveryLog } from '@/types/api';
import { CodeBlock } from './CodeBlock';

export function MCPForge() {
  const [url, setUrl] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ForgeResponse>();
  const [logs, setLogs] = useState<DiscoveryLog[]>([]);
  const [activeTab, setActiveTab] = useState<'docs' | 'code' | 'logs'>('docs');

  const handleGenerate = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!url.trim() || loading) return;

    setLoading(true);
    setLogs([]);
    setResult(undefined);

    try {
      const response = await api.forgeGenerate({
        source_url: url,
        force_regenerate: false,
      });

      setResult(response);
      setLogs(response.discovery_logs);
      setActiveTab('code'); // Switch to code tab on success
    } catch (error) {
      console.error('Forge generation error:', error);
      alert('Failed to generate tool. Please check the URL and try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="mcp-forge">
      {/* Input Section */}
      <form onSubmit={handleGenerate} className="forge-input">
        <input
          type="url"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          placeholder="https://api.example.com/docs"
          disabled={loading}
          required
        />
        <button type="submit" disabled={loading || !url.trim()}>
          {loading ? 'Generating...' : 'Generate Tool'}
        </button>
      </form>

      {/* Results Section */}
      {(result || loading) && (
        <div className="forge-results">
          {/* Tab Navigation */}
          <div className="tab-nav">
            <button
              className={activeTab === 'docs' ? 'active' : ''}
              onClick={() => setActiveTab('docs')}
            >
              📄 Documentation
            </button>
            <button
              className={activeTab === 'code' ? 'active' : ''}
              onClick={() => setActiveTab('code')}
            >
              💻 Generated Code
            </button>
            <button
              className={activeTab === 'logs' ? 'active' : ''}
              onClick={() => setActiveTab('logs')}
            >
              📋 Discovery Logs ({logs.length})
            </button>
          </div>

          {/* Tab Content */}
          <div className="tab-content">
            {/* Documentation Tab */}
            {activeTab === 'docs' && result && (
              <div className="docs-panel">
                <div className="docs-stats">
                  <div className="stat">
                    <span className="label">Endpoints Found:</span>
                    <span className="value">
                      {result.documentation.endpoints_found}
                    </span>
                  </div>
                  <div className="stat">
                    <span className="label">Auth Params:</span>
                    <span className="value">
                      {result.documentation.auth_params.join(', ')}
                    </span>
                  </div>
                  <div className="stat">
                    <span className="label">Base URL:</span>
                    <span className="value">
                      {result.documentation.base_url}
                    </span>
                  </div>
                </div>
                <div className="markdown-content">
                  <CodeBlock
                    code={result.documentation.markdown}
                    language="markdown"
                  />
                </div>
              </div>
            )}

            {/* Code Tab */}
            {activeTab === 'code' && result && (
              <div className="code-panel">
                <div className="code-header">
                  <h4>Generated Tool: {result.tool_id}</h4>
                  <button
                    onClick={() => {
                      navigator.clipboard.writeText(
                        result.generated_code.typescript
                      );
                      alert('Code copied to clipboard!');
                    }}
                    className="btn-copy"
                  >
                    📋 Copy Code
                  </button>
                </div>
                <CodeBlock
                  code={result.generated_code.typescript}
                  language={result.generated_code.language}
                />
              </div>
            )}

            {/* Logs Tab */}
            {activeTab === 'logs' && (
              <div className="logs-panel">
                {loading && logs.length === 0 ? (
                  <div className="loading-state">
                    Generating tool... This may take a few seconds.
                  </div>
                ) : (
                  <div className="log-entries">
                    {logs.map((log, idx) => (
                      <div key={idx} className="log-entry">
                        <span className="timestamp">{log.timestamp}</span>
                        <span className={`source source-${log.source}`}>
                          [{log.source}]
                        </span>
                        <span className="message">{log.message}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Metadata Footer */}
          {result && (
            <div className="forge-metadata">
              <span>
                Generation Time: {result.metadata.generation_time_ms}ms
              </span>
              <span>
                Pages Crawled: {result.metadata.firecrawl_pages_crawled}
              </span>
              <span>Tokens Used: {result.metadata.tokens_used}</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
```

---

### 4. ToolMarketplace

**Component Purpose:** Browse and search tools with enhanced metadata

**Backend Endpoint:** `GET /tools`, `GET /tools/search`

**Key Features:**
- Enhanced tool cards with status badges
- Category filtering
- Semantic search
- Usage statistics
- Verification badges

**Implementation:**

```typescript
// components/ToolMarketplace.tsx
'use client';

import { useState, useEffect } from 'react';
import { api } from '@/lib/api-client';
import type { EnhancedTool } from '@/types/api';
import { ToolCard } from './ToolCard';

export function ToolMarketplace() {
  const [tools, setTools] = useState<EnhancedTool[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [sortBy, setSortBy] = useState<'usage' | 'recent'>('usage');

  // Fetch tools on mount
  useEffect(() => {
    loadTools();
  }, []);

  const loadTools = async () => {
    setLoading(true);
    try {
      const data = await api.listTools(100, 0);
      setTools(data);
    } catch (error) {
      console.error('Failed to load tools:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleSearch = async () => {
    if (!searchQuery.trim()) {
      loadTools();
      return;
    }

    setLoading(true);
    try {
      const result = await api.searchTools(searchQuery, 20);
      setTools(result.tools);
    } catch (error) {
      console.error('Search failed:', error);
    } finally {
      setLoading(false);
    }
  };

  // Get unique categories
  const categories = ['all', ...new Set(tools.map((t) => t.category))];

  // Filter and sort tools
  const filteredTools = tools
    .filter((tool) =>
      selectedCategory === 'all' ? true : tool.category === selectedCategory
    )
    .sort((a, b) => {
      if (sortBy === 'usage') {
        return b.usage_count - a.usage_count;
      } else {
        return (
          new Date(b.created_at || 0).getTime() -
          new Date(a.created_at || 0).getTime()
        );
      }
    });

  return (
    <div className="tool-marketplace">
      {/* Header */}
      <div className="marketplace-header">
        <h2>Tool Marketplace</h2>
        <p className="subtitle">
          {tools.length} tools available • {tools.filter((t) => t.verified).length} verified
        </p>
      </div>

      {/* Search and Filters */}
      <div className="controls">
        <div className="search-bar">
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
            placeholder="Search tools semantically..."
          />
          <button onClick={handleSearch} disabled={loading}>
            🔍 Search
          </button>
        </div>

        <div className="filters">
          <select
            value={selectedCategory}
            onChange={(e) => setSelectedCategory(e.target.value)}
          >
            {categories.map((cat) => (
              <option key={cat} value={cat}>
                {cat === 'all' ? 'All Categories' : cat}
              </option>
            ))}
          </select>

          <select value={sortBy} onChange={(e) => setSortBy(e.target.value as any)}>
            <option value="usage">Most Used</option>
            <option value="recent">Recently Added</option>
          </select>
        </div>
      </div>

      {/* Tools Grid */}
      {loading ? (
        <div className="loading-state">Loading tools...</div>
      ) : filteredTools.length === 0 ? (
        <div className="empty-state">
          No tools found. Try a different search or category.
        </div>
      ) : (
        <div className="tools-grid">
          {filteredTools.map((tool) => (
            <ToolCard key={tool.id} tool={tool} onExecute={() => {}} />
          ))}
        </div>
      )}
    </div>
  );
}
```

**ToolCard Component:**

```typescript
// components/ToolCard.tsx
import type { EnhancedTool } from '@/types/api';

interface Props {
  tool: EnhancedTool;
  onExecute: (toolName: string) => void;
}

export function ToolCard({ tool, onExecute }: Props) {
  const getStatusColor = (status: EnhancedTool['status']): string => {
    const colors = {
      'PROD-READY': 'bg-green-500',
      BETA: 'bg-yellow-500',
      DEPRECATED: 'bg-red-500',
    };
    return colors[status];
  };

  return (
    <div className="tool-card">
      {/* Header */}
      <div className="card-header">
        <div className="title-row">
          <h3 className="tool-name">{tool.name}</h3>
          {tool.verified && <span className="verified-badge">✓ Verified</span>}
        </div>
        <span className={`status-badge ${getStatusColor(tool.status)}`}>
          {tool.status}
        </span>
      </div>

      {/* Description */}
      <p className="description">{tool.description}</p>

      {/* Code Preview */}
      {tool.preview_snippet && (
        <code className="code-snippet">{tool.preview_snippet}</code>
      )}

      {/* Tags */}
      <div className="tags">
        <span className="category-tag">{tool.category}</span>
        {tool.tags.slice(0, 3).map((tag) => (
          <span key={tag} className="tag">
            {tag}
          </span>
        ))}
      </div>

      {/* Stats */}
      <div className="stats">
        <span className="stat">
          <span className="icon">🔢</span>
          {tool.usage_count} uses
        </span>
        {tool.source_url && (
          <a
            href={tool.source_url}
            target="_blank"
            rel="noopener noreferrer"
            className="source-link"
          >
            📚 Docs
          </a>
        )}
      </div>

      {/* Actions */}
      <div className="card-actions">
        <button
          onClick={() => onExecute(tool.name)}
          className="btn-primary"
        >
          Execute
        </button>
        <button className="btn-secondary">View Details</button>
      </div>

      {/* Video preview if available */}
      {tool.mux_playback_id && (
        <div className="video-preview">
          <span className="video-icon">🎥 Demo Available</span>
        </div>
      )}
    </div>
  );
}
```

---

### 5. ResultCard

**Component Purpose:** Display tool execution results

**Used In:** CommandCenter after tool calls

**Key Features:**
- Structured result display
- Execution time
- Status indicator
- JSON formatting

**Implementation:**

```typescript
// components/ResultCard.tsx
interface Props {
  toolName: string;
  arguments: Record<string, any>;
  result: Record<string, any> | null;
  executionTime: number;
  status: 'success' | 'error';
}

export function ResultCard({
  toolName,
  arguments: args,
  result,
  executionTime,
  status,
}: Props) {
  const formatJSON = (obj: Record<string, any>): string => {
    return JSON.stringify(obj, null, 2);
  };

  return (
    <div className={`result-card ${status}`}>
      {/* Header */}
      <div className="result-header">
        <div className="tool-info">
          <span className="tool-icon">🔧</span>
          <span className="tool-name">{toolName}</span>
          <span className={`status-indicator ${status}`}>
            {status === 'success' ? '✓' : '✗'}
          </span>
        </div>
        <span className="execution-time">{executionTime}ms</span>
      </div>

      {/* Arguments */}
      <div className="section">
        <h4 className="section-title">Input</h4>
        <pre className="json-display">{formatJSON(args)}</pre>
      </div>

      {/* Result */}
      <div className="section">
        <h4 className="section-title">
          {status === 'success' ? 'Output' : 'Error'}
        </h4>
        {result ? (
          <pre className="json-display">{formatJSON(result)}</pre>
        ) : (
          <p className="no-result">No result available</p>
        )}
      </div>
    </div>
  );
}
```

---

### 6. ActionCenter

**Component Purpose:** Display action feed from agent activity

**Backend Endpoint:** `GET /api/actions`

**Key Features:**
- Real-time action updates
- Status indicators
- GitHub PR links
- Tool execution links

**Implementation:**

```typescript
// components/ActionCenter.tsx
'use client';

import { useState, useEffect } from 'react';
import { api } from '@/lib/api-client';
import type { Action } from '@/types/api';

interface Props {
  conversationId?: string;
  autoRefresh?: boolean;
  refreshInterval?: number; // in ms
}

export function ActionCenter({
  conversationId,
  autoRefresh = false,
  refreshInterval = 5000,
}: Props) {
  const [actions, setActions] = useState<Action[]>([]);
  const [loading, setLoading] = useState(true);

  const loadActions = async () => {
    try {
      const data = await api.getActions(conversationId, 50, 0);
      setActions(data);
    } catch (error) {
      console.error('Failed to load actions:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadActions();

    if (autoRefresh) {
      const interval = setInterval(loadActions, refreshInterval);
      return () => clearInterval(interval);
    }
  }, [conversationId, autoRefresh]);

  const getStatusIcon = (status: Action['status']): string => {
    const icons = {
      success: '✓',
      pending: '⏳',
      error: '✗',
    };
    return icons[status];
  };

  return (
    <div className="action-center">
      <div className="header">
        <h3>Action Feed</h3>
        <button onClick={loadActions} disabled={loading}>
          🔄 Refresh
        </button>
      </div>

      {loading ? (
        <div className="loading-state">Loading actions...</div>
      ) : actions.length === 0 ? (
        <div className="empty-state">No actions yet</div>
      ) : (
        <div className="actions-list">
          {actions.map((action) => (
            <div key={action.id} className={`action-item ${action.status}`}>
              <div className="action-header">
                <span className={`status-icon ${action.status}`}>
                  {getStatusIcon(action.status)}
                </span>
                <span className="action-title">{action.title}</span>
                <span className="action-time">
                  {new Date(action.timestamp).toLocaleTimeString()}
                </span>
              </div>

              <p className="action-detail">{action.detail}</p>

              <div className="action-footer">
                {action.tool_name && (
                  <span className="tool-badge">{action.tool_name}</span>
                )}
                {action.github_pr_url && (
                  <a
                    href={action.github_pr_url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="pr-link"
                  >
                    View PR →
                  </a>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
```

---

### 7. TrustGovernanceLedger

**Component Purpose:** Display verified tools with governance metadata

**Backend Endpoint:** `GET /api/governance/verified-tools`

**Key Features:**
- Trust scores
- Security scan status
- Rate limits
- Governance policies

**Implementation:**

```typescript
// components/TrustGovernanceLedger.tsx
'use client';

import { useState, useEffect } from 'react';
import { api } from '@/lib/api-client';
import type { VerifiedTool } from '@/types/api';

export function TrustGovernanceLedger() {
  const [tools, setTools] = useState<VerifiedTool[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadVerifiedTools();
  }, []);

  const loadVerifiedTools = async () => {
    try {
      const data = await api.getVerifiedTools();
      setTools(data);
    } catch (error) {
      console.error('Failed to load verified tools:', error);
    } finally {
      setLoading(false);
    }
  };

  const getTrustScoreColor = (score: number): string => {
    if (score >= 90) return 'text-green-500';
    if (score >= 70) return 'text-yellow-500';
    return 'text-red-500';
  };

  return (
    <div className="governance-ledger">
      <div className="header">
        <h2>Trust & Governance Ledger</h2>
        <p className="subtitle">{tools.length} verified tools</p>
      </div>

      {loading ? (
        <div className="loading-state">Loading verified tools...</div>
      ) : (
        <div className="tools-table">
          <table>
            <thead>
              <tr>
                <th>Tool</th>
                <th>Trust Score</th>
                <th>Security</th>
                <th>Last Audit</th>
                <th>Rate Limit</th>
                <th>Cost</th>
              </tr>
            </thead>
            <tbody>
              {tools.map((tool) => (
                <tr key={tool.id}>
                  <td>
                    <div className="tool-cell">
                      <span className="tool-name">{tool.name}</span>
                      <span className="tool-status">{tool.status}</span>
                    </div>
                  </td>
                  <td>
                    <span
                      className={`trust-score ${getTrustScoreColor(
                        tool.verification.trust_score
                      )}`}
                    >
                      {tool.verification.trust_score}/100
                    </span>
                  </td>
                  <td>
                    <span
                      className={`security-badge ${
                        tool.verification.security_scan_passed
                          ? 'passed'
                          : 'failed'
                      }`}
                    >
                      {tool.verification.security_scan_passed ? '✓ Passed' : '✗ Failed'}
                    </span>
                  </td>
                  <td>
                    {new Date(tool.verification.last_audit).toLocaleDateString()}
                  </td>
                  <td>{tool.governance.rate_limit_per_minute}/min</td>
                  <td>${tool.governance.cost_per_execution.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
```

---

## State Management

For complex state management across components, use Zustand:

```typescript
// store/useAgentStore.ts
import { create } from 'zustand';
import type { ChatResponse, EnhancedTool, Action } from '@/types/api';

interface AgentStore {
  // Chat state
  conversationId: string | undefined;
  messages: ChatResponse[];
  setConversationId: (id: string) => void;
  addMessage: (message: ChatResponse) => void;
  clearMessages: () => void;

  // Tools state
  tools: EnhancedTool[];
  setTools: (tools: EnhancedTool[]) => void;

  // Actions state
  actions: Action[];
  addAction: (action: Action) => void;
  setActions: (actions: Action[]) => void;
}

export const useAgentStore = create<AgentStore>((set) => ({
  // Chat
  conversationId: undefined,
  messages: [],
  setConversationId: (id) => set({ conversationId: id }),
  addMessage: (message) =>
    set((state) => ({ messages: [...state.messages, message] })),
  clearMessages: () => set({ messages: [] }),

  // Tools
  tools: [],
  setTools: (tools) => set({ tools }),

  // Actions
  actions: [],
  addAction: (action) =>
    set((state) => ({ actions: [action, ...state.actions] })),
  setActions: (actions) => set({ actions }),
}));
```

**Usage in components:**

```typescript
import { useAgentStore } from '@/store/useAgentStore';

function MyComponent() {
  const { conversationId, setConversationId, addMessage } = useAgentStore();

  // Use state...
}
```

---

## Error Handling Patterns

Create a centralized error handler:

```typescript
// lib/error-handler.ts
import { APIClientError } from './api-client';
import { toast } from 'sonner'; // or your preferred toast library

export function handleAPIError(error: unknown): void {
  if (error instanceof APIClientError) {
    // Handle API-specific errors
    switch (error.code) {
      case 'RATE_LIMITED':
        toast.error('Too many requests. Please wait a moment.');
        break;
      case 'NOT_FOUND':
        toast.error('Resource not found.');
        break;
      case 'INTERNAL_ERROR':
        toast.error('Server error. Please try again later.');
        break;
      default:
        toast.error(error.message || 'An error occurred.');
    }
  } else if (error instanceof Error) {
    // Generic errors
    toast.error(error.message);
  } else {
    toast.error('An unexpected error occurred.');
  }

  // Log to console for debugging
  console.error('Error:', error);
}
```

**Usage:**

```typescript
import { handleAPIError } from '@/lib/error-handler';

try {
  await api.chat({ message: 'Hello' });
} catch (error) {
  handleAPIError(error);
}
```

---

## Testing

### API Client Tests

```typescript
// __tests__/api-client.test.ts
import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { UniversalAdapterAPI } from '@/lib/api-client';

describe('UniversalAdapterAPI', () => {
  let api: UniversalAdapterAPI;

  beforeAll(() => {
    api = new UniversalAdapterAPI('http://localhost:8001');
  });

  it('should send chat message', async () => {
    const response = await api.chat({
      message: 'Get Bitcoin price',
    });

    expect(response.success).toBe(true);
    expect(response.conversation_id).toBeDefined();
    expect(response.workflow_steps).toBeInstanceOf(Array);
  });

  it('should list tools', async () => {
    const tools = await api.listTools();

    expect(Array.isArray(tools)).toBe(true);
    expect(tools.length).toBeGreaterThan(0);
    expect(tools[0]).toHaveProperty('name');
    expect(tools[0]).toHaveProperty('status');
  });

  it('should search tools semantically', async () => {
    const result = await api.searchTools('cryptocurrency price');

    expect(result.count).toBeGreaterThan(0);
    expect(result.tools[0]).toHaveProperty('similarity_score');
  });
});
```

### Component Tests

```typescript
// __tests__/CommandCenter.test.tsx
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { CommandCenter } from '@/components/CommandCenter';
import { api } from '@/lib/api-client';

// Mock API
jest.mock('@/lib/api-client');

describe('CommandCenter', () => {
  it('should send message and display response', async () => {
    // Mock API response
    (api.chat as jest.Mock).mockResolvedValue({
      success: true,
      response: 'Bitcoin is $78,055',
      conversation_id: 'test-123',
      model: 'claude-haiku-4.5',
      workflow_steps: [
        { step: 'checking', status: 'completed', duration_ms: 100, message: 'Checking...' },
        { step: 'done', status: 'completed', duration_ms: 0, message: 'Done' },
      ],
      tool_calls: [],
      actions_logged: [],
      metadata: { total_duration_ms: 1000, tokens_used: 50, cost_usd: 0.001 },
    });

    render(<CommandCenter />);

    // Type message
    const input = screen.getByPlaceholderText('Enter your command...');
    fireEvent.change(input, { target: { value: 'Get Bitcoin price' } });

    // Submit
    const button = screen.getByText('Execute');
    fireEvent.click(button);

    // Wait for response
    await waitFor(() => {
      expect(screen.getByText('Bitcoin is $78,055')).toBeInTheDocument();
    });
  });
});
```

---

## Summary

This guide provides complete integration examples for all Universal Adapter UI components with the enhanced backend API v2.0.

**Key Takeaways:**

1. **Use the Enhanced API Client** - Centralized API access with error handling
2. **Handle Workflow Steps** - Animate progress through checking → discovering → forging → done
3. **Stream Discovery Logs** - Use SSE for real-time updates
4. **Display Enhanced Tool Metadata** - Show status, categories, verification badges
5. **Implement Error Handling** - Centralized error handler for consistent UX
6. **Test Components** - Unit and integration tests for reliability

**Next Steps:**

1. Set up environment variables
2. Install dependencies
3. Integrate API client
4. Implement components one-by-one
5. Test with backend running on `localhost:8001`

**Reference Documents:**

- `UI_QUICK_START.md` - Basic API examples
- `API_DOCUMENTATION.md` - Complete API reference
- `IMPLEMENTATION_GUIDE.md` - Backend implementation details
- `BACKEND_API_REQUIREMENTS.md` - Full requirements specification

---

**Happy Coding! 🚀**

For questions or issues, refer to the backend team or check the server logs at `http://localhost:8001/health`.
