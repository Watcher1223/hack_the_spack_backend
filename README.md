# YC Hack 2 - Serverless MCP Marketplace

## Overview
A hackathon project implementing a "serverless" MCP (Model Context Protocol) approach where AI agents can dynamically generate and use tools without running actual server processes.

## Architecture

### Core Components

1. **LLM Agent** (`services/llm.py`)
   - OpenRouter-powered agent with tool calling capabilities
   - Handles multi-turn conversations with tool execution
   - Iterative tool calling until final answer is reached

2. **Base Tools** (`services/tools.py`)
   - **file_read**: Read files from artifacts directory
   - **file_write**: Write/save files to artifacts directory
   - **file_list**: List files and directories in artifacts
   - **firecrawl_search**: Search the web and get full page content
   - **firecrawl_scrape**: Scrape content from a single webpage
   - **firecrawl_crawl**: Crawl entire websites and extract content
   - **generate_tool**: Create and save new tool definitions

3. **File System** (artifacts directory)
   - All file operations happen in `artifacts/` directory
   - Safe path validation prevents directory traversal
   - Agent can read, write, and list files
   - Supports subdirectories

4. **Tool Storage** (File-based for now)
   - Tools saved as JSON files in `generated_tools/` directory
   - Simple CRUD operations: save, load, list
   - Ready to migrate to MongoDB later

## Setup

1. Install dependencies:
```bash
uv sync
```

2. Create `dev.env` file:
```env
OPENROUTER_API_KEY=your_key_here
FIRECRAWL_API_KEY=your_key_here
```

3. Run the agent:
```bash
python main.py
```

4. Test tool generation:
```bash
python test_tools.py
```

## How It Works

1. User asks a question
2. Agent determines if tools are needed
3. Agent calls appropriate tools (search, scrape, etc.)
4. If no tool exists, agent uses `generate_tool` to create one **and it's immediately available**
5. Tool results are processed and final answer is returned
6. Full conversation history is automatically saved to `conversations/`

**Key Feature:** Generated tools are auto-reloaded and can be used in the same conversation!

## Conversation Logging

Every conversation is automatically saved with:
- Complete message history (user, assistant, tool calls, tool results)
- Token usage statistics
- Timestamps
- Model used
- Final output

**View saved conversations:**
```bash
python view_conversations.py
```

**Disable conversation saving:**
```python
agent = Agent(save_conversations=False)
```

## Next Steps
- Integrate MongoDB for tool storage (see `understand/` directory for specs)
- Implement code execution engine for generated tools
- Add tool versioning and marketplace features
