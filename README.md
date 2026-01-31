# YC Hack 2 - Serverless MCP Marketplace

## Overview
An autonomous AI agent that can dynamically generate and execute tools by searching API documentation, creating Python code, and running it on-demand. Built for hackathons - no database setup required, works out of the box with just API keys.

## Key Features
- 🤖 **Fully Autonomous**: Give it a task, it completes it end-to-end without user intervention
- 🔧 **Dynamic Tool Generation**: Searches for API docs, generates working Python code, and executes it
- 📁 **File Operations**: Can read, write, and manage files in the `artifacts/` directory
- 🔄 **Auto-Reload**: Generated tools are immediately available in the same conversation
- 💾 **Conversation Logging**: All interactions saved to `conversations/` for debugging
- 🌐 **Web Integration**: Built-in web search, scraping, and crawling via Firecrawl

## Quick Start

### 1. Prerequisites
- Python 3.13+
- [uv](https://github.com/astral-sh/uv) package manager
- OpenRouter API key ([get one here](https://openrouter.ai/keys))
- Firecrawl API key ([get one here](https://firecrawl.dev))

### 2. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd yc-hack2

# Install dependencies
uv sync
```

### 3. Configuration

Create a `dev.env` file in the project root:

```bash
cp .env.example dev.env
```

Edit `dev.env` and add your API keys:

```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
FIRECRAWL_API_KEY=your_firecrawl_api_key_here
```

### 4. Run the Agent

Pass your question as a command-line argument:

```bash
uv run main.py "Download an image of a cat and save it to the cat directory"
```

Or try other examples:

```bash
# Get real-time data
uv run main.py "Get me the real-time stream flow data for the Mississippi River"

# Fetch weather data
uv run main.py "Get current weather for Austin, TX and save it to a file"

# Download and process content
uv run main.py "Scrape the latest news from HackerNews and summarize it"
```

The agent will autonomously:
1. Search for relevant API documentation
2. Scrape the API specs
3. Generate a Python tool with working code
4. Execute the tool to complete your request
5. Save results to the `artifacts/` directory

## Architecture

### Core Components

1. **LLM Agent** (`services/llm.py`)
   - OpenRouter-powered (Claude Haiku 4.5 by default)
   - Autonomous workflow: search → scrape → generate → execute
   - Multi-turn tool calling with auto-reload
   - Conversation logging and error recovery

2. **Base Tools** (`services/tools.py`)
   - **file_read/write/list**: File operations in `artifacts/` directory
   - **firecrawl_search**: Search the web for API docs
   - **firecrawl_scrape**: Extract content from webpages
   - **firecrawl_crawl**: Crawl entire websites
   - **generate_tool**: Create executable Python tools

3. **Tool Execution**
   - Generated tools stored as JSON in `generated_tools/`
   - Code executed using Python's `exec()` with base64 support
   - Auto-reload makes new tools immediately available
   - Supports both text and binary file operations

4. **File System**
   - `artifacts/` - Agent's workspace for all file operations
   - `conversations/` - Saved conversation logs
   - `generated_tools/` - Tool definitions with executable code
   - `logs/` - Application logs

## How It Works

### Example: "Download an image of a cat"

```
1. Agent searches for image APIs
   → firecrawl_search("cat image API download")

2. Agent scrapes API documentation
   → firecrawl_scrape("https://thecatapi.com/docs")

3. Agent generates a download tool
   → generate_tool(name="download_cat_image", code="...", ...)
   → Tool auto-reloads, immediately available

4. Agent executes the new tool
   → download_cat_image(output_path="cat/image.jpg")

5. Agent saves the result
   → file_write("cat/image.jpg", base64_data, mode="wb")

6. Done! Image saved to artifacts/cat/image.jpg
```

All of this happens **autonomously** in one execution, no user intervention required.

## Testing

### Basic Tests
```bash
# Test file operations
uv run test_file_operations.py

# Test tool execution
uv run test_tool_execution.py

# Test auto-reload feature
uv run test_auto_reload.py
```

### Example Scripts
```bash
# Run example workflows
uv run example_usage.py
```

## Advanced Usage

### Running Different Tasks

Simply pass different questions as command-line arguments:

```bash
# API data retrieval
uv run main.py "Fetch Bitcoin price from CoinGecko API"

# Web scraping and analysis
uv run main.py "Find and download the top 5 Python repos on GitHub"

# Data processing
uv run main.py "Get COVID-19 statistics and create a summary report"

# Image/file operations
uv run main.py "Download the Eiffel Tower image and save it"
```

The agent will automatically:
1. Search for relevant APIs
2. Generate tools to interact with them
3. Execute the tools
4. Save results to `artifacts/`

### View Conversation Logs

```bash
uv run view_conversations.py
```

Shows complete conversation history with all tool calls and results.

### Modify Behavior

To change max iterations or other settings, edit `main.py`:

```python
# Increase iterations for complex tasks
res = await agent.run(question, max_iterations=50)

# Change the model
agent = Agent(model="anthropic/claude-opus-4.5")

# Disable conversation logging
agent = Agent(save_conversations=False)
```

## Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENROUTER_API_KEY` | Yes | API key from OpenRouter.ai |
| `FIRECRAWL_API_KEY` | Yes | API key from Firecrawl.dev |

### Agent Configuration

Customize the agent in `services/llm.py`:

```python
agent = Agent(
    model="anthropic/claude-haiku-4.5",  # Model to use
    save_conversations=True,              # Save conversation logs
    firecrawl_api_key="...",             # Optional: override env var
)
```

## Project Structure

```
yc-hack2/
├── main.py                      # Main entry point
├── dev.env                      # Your API keys (gitignored)
├── .env.example                 # Template for dev.env
├── services/
│   ├── llm.py                  # Agent with autonomous workflow
│   ├── tools.py                # Base tools + tool execution
│   ├── env.py                  # Environment variables
│   └── logging.py              # Logging configuration
├── artifacts/                   # Agent's file workspace (gitignored)
├── generated_tools/            # Tool definitions (gitignored)
├── conversations/              # Conversation logs (gitignored)
├── logs/                       # Application logs (gitignored)
├── test_*.py                   # Test scripts
├── example_usage.py            # Example workflows
├── view_conversations.py       # View saved conversations
└── understand/                 # Reference documentation (temporary)
```

## Troubleshooting

### "No module named 'services'"
Run `uv sync` to install dependencies.

### "OPENROUTER_API_KEY not found"
Create `dev.env` file with your API keys (see `.env.example`).

### Firecrawl Timeout Errors
Normal for large documentation pages. The agent handles these gracefully and continues with alternate URLs.

### Agent Hits Max Iterations
Increase `max_iterations` in `main.py` for complex tasks.

## Documentation

- **TOOL_EXECUTION_GUIDE.md** - How tool generation and execution works
- **BINARY_FILES_GUIDE.md** - Working with images, PDFs, and binary data
- **IMPLEMENTATION_SUMMARY.md** - What's implemented and what's next
- **understand/** - Architecture, flows, and data models

## Contributing

This is a hackathon project. Feel free to extend it with:
- MongoDB integration for tool storage
- Enhanced sandboxing for code execution
- Tool versioning and marketplace features
- Multi-agent collaboration
- Web UI for agent interaction

## License

MIT
