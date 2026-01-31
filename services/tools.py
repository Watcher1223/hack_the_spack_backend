import base64
import httpx
import inspect
import json
import os

from pathlib import Path
from typing import Optional, Any, Callable
from pydantic import BaseModel

# ============================================
# File Operations (artifacts directory)
# ============================================

ARTIFACTS_DIR = "artifacts"


def _get_safe_path(filepath: str) -> Path:
    """
    Get a safe path within the artifacts directory.
    Prevents directory traversal attacks.
    """
    # Create artifacts directory if it doesn't exist
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # Get absolute path and ensure it's within artifacts
    base_path = Path(ARTIFACTS_DIR).resolve()
    target_path = (base_path / filepath).resolve()

    # Check if target is within base directory
    if not str(target_path).startswith(str(base_path)):
        raise ValueError(
            f"Access denied: Path '{filepath}' is outside artifacts directory"
        )

    return target_path


async def file_read(
    filepath: str, mode: str = "r", encoding: str = "utf-8"
) -> dict[str, Any]:
    """
    Read a file from the artifacts directory.
    Handles both text and binary data.

    Args:
        filepath: Path relative to artifacts directory
        mode: Read mode - 'r' (text) or 'rb' (binary, returns base64)
        encoding: Text encoding (default: utf-8), ignored for binary mode
    """
    try:
        safe_path = _get_safe_path(filepath)

        if not safe_path.exists():
            return {"success": False, "error": f"File not found: {filepath}"}

        if not safe_path.is_file():
            return {"success": False, "error": f"Not a file: {filepath}"}

        # Validate mode
        if mode not in ["r", "rb"]:
            return {
                "success": False,
                "error": f"Invalid mode: {mode}. Use 'r' (text) or 'rb' (binary)",
            }

        # Read binary data
        if mode == "rb":
            binary_data = safe_path.read_bytes()
            content = base64.b64encode(binary_data).decode("ascii")
            content_type = "binary_base64"
        # Read text data
        else:
            content = safe_path.read_text(encoding=encoding)
            content_type = "text"

        return {
            "success": True,
            "filepath": filepath,
            "content": content,
            "content_type": content_type,
            "size": safe_path.stat().st_size,
        }

    except Exception as e:
        return {"success": False, "error": f"Error reading file: {str(e)}"}


async def file_write(
    filepath: str, content: str, mode: str = "w", encoding: str = "utf-8"
) -> dict[str, Any]:
    """
    Write content to a file in the artifacts directory.
    Handles both text and binary data automatically.

    Args:
        filepath: Path relative to artifacts directory
        content: Content to write (string for text, base64 string for binary)
        mode: Write mode - 'w' (overwrite), 'a' (append), 'wb' (binary overwrite), 'ab' (binary append)
        encoding: Text encoding (default: utf-8), ignored for binary mode
    """
    try:
        safe_path = _get_safe_path(filepath)

        # Create parent directories if they don't exist
        safe_path.parent.mkdir(parents=True, exist_ok=True)

        # Validate mode
        valid_modes = ["w", "a", "wb", "ab"]
        if mode not in valid_modes:
            return {
                "success": False,
                "error": f"Invalid mode: {mode}. Use 'w', 'a', 'wb', or 'ab'",
            }

        is_binary = mode in ["wb", "ab"]
        is_append = mode in ["a", "ab"]

        # Handle binary data
        if is_binary:
            # Content should be base64 encoded for binary
            try:
                binary_data = base64.b64decode(content)
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Binary mode requires base64 encoded content: {str(e)}",
                }

            if is_append:
                with open(safe_path, "ab") as f:
                    f.write(binary_data)
            else:
                safe_path.write_bytes(binary_data)

        # Handle text data
        else:
            if is_append:
                with open(safe_path, "a", encoding=encoding) as f:
                    f.write(content)
            else:
                safe_path.write_text(content, encoding=encoding)

        return {
            "success": True,
            "filepath": filepath,
            "absolute_path": str(safe_path),
            "size": safe_path.stat().st_size,
            "mode": "binary" if is_binary else "text",
            "operation": "appended" if is_append else "overwritten",
        }

    except Exception as e:
        return {"success": False, "error": f"Error writing file: {str(e)}"}


async def file_list(directory: str = ".") -> dict[str, Any]:
    """
    List files and directories in the artifacts directory.
    """
    try:
        safe_path = _get_safe_path(directory)

        if not safe_path.exists():
            return {"success": False, "error": f"Directory not found: {directory}"}

        if not safe_path.is_dir():
            return {"success": False, "error": f"Not a directory: {directory}"}

        items = []
        for item in safe_path.iterdir():
            items.append(
                {
                    "name": item.name,
                    "type": "directory" if item.is_dir() else "file",
                    "size": item.stat().st_size if item.is_file() else None,
                }
            )

        return {
            "success": True,
            "directory": directory,
            "items": items,
            "count": len(items),
        }

    except Exception as e:
        return {"success": False, "error": f"Error listing directory: {str(e)}"}


# ============================================
# Firecrawl API Client
# ============================================


class Firecrawl:
    def __init__(self, api_key: str):
        self.base_url = "https://api.firecrawl.dev/v2"
        self.api_key = api_key

    async def search(
        self, query: str, limit: int = 5, timeout: int = 60000
    ) -> dict[str, Any]:
        """
        Search the web using Firecrawl and get full page content.
        """
        url = f"{self.base_url}/search"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {"query": query, "limit": limit, "timeout": timeout}

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()

    async def scrape(
        self,
        url: str,
        formats: list[str] = None,
        only_main_content: bool = True,
        timeout: int = 30000,
    ) -> dict[str, Any]:
        """
        Scrape a single URL and extract content in specified formats.
        """
        api_url = f"{self.base_url}/scrape"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "url": url,
            "formats": formats or ["markdown"],
            "onlyMainContent": only_main_content,
            "timeout": timeout,
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(api_url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()

    async def crawl(
        self, url: str, limit: int = 10, max_depth: int = 2, timeout: int = 120000
    ) -> dict[str, Any]:
        """
        Crawl an entire website starting from a URL.
        """
        api_url = f"{self.base_url}/crawl"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "url": url,
            "limit": limit,
            "maxDepth": max_depth,
            "timeout": timeout,
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(api_url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()


# ============================================
# Tool Storage (Pure Python - File-based)
# ============================================

TOOLS_DIR = "generated_tools"


class ToolDefinition(BaseModel):
    name: str
    description: str
    parameters: dict
    code: Optional[str] = None  # Python function code as string
    dependencies: Optional[list[str]] = None  # Required packages


def save_tool(tool_definition: dict) -> dict:
    """
    Save a generated tool definition to local storage.
    Includes both schema and optional executable code.
    """
    os.makedirs(TOOLS_DIR, exist_ok=True)

    # Validate and create tool definition
    tool = ToolDefinition(**tool_definition)
    file_path = os.path.join(TOOLS_DIR, f"{tool.name}.json")

    with open(file_path, "w") as f:
        json.dump(tool.model_dump(), f, indent=2)

    return {
        "success": True,
        "message": f"Tool '{tool.name}' saved successfully",
        "path": file_path,
        "has_code": tool.code is not None,
    }


def load_tool(name: str) -> dict:
    """
    Load a tool definition from local storage.
    """
    file_path = os.path.join(TOOLS_DIR, f"{name}.json")

    if not os.path.exists(file_path):
        return {"success": False, "error": f"Tool '{name}' not found"}

    with open(file_path, "r") as f:
        return json.load(f)


def list_tools() -> list[dict]:
    """
    List all saved tool definitions.
    """
    if not os.path.exists(TOOLS_DIR):
        return []

    tools = []
    for filename in os.listdir(TOOLS_DIR):
        if filename.endswith(".json"):
            with open(os.path.join(TOOLS_DIR, filename), "r") as f:
                tools.append(json.load(f))
    return tools


async def execute_generated_tool(tool_name: str, arguments: dict) -> dict:
    """
    Execute a generated tool by loading its code and running it.
    Uses exec() to run the stored Python code string.
    """
    # Load the tool
    tool_data = load_tool(tool_name)

    if not tool_data.get("success", True):
        return {"error": f"Tool '{tool_name}' not found"}

    code = tool_data.get("code")
    if not code:
        return {
            "error": f"Tool '{tool_name}' has no executable code. It's just a schema definition.",
            "suggestion": "You need to generate the implementation code for this tool first.",
        }

    try:
        # Create a restricted execution environment
        exec_globals = {
            "__builtins__": __builtins__,
            "httpx": httpx,
            "json": json,
            "os": os,
        }
        exec_locals = {}

        # Execute the code to define the function
        exec(code, exec_globals, exec_locals)

        # Get the function from locals (assume it's the tool name)
        if tool_name not in exec_locals:
            return {"error": f"Function '{tool_name}' not found in generated code"}

        func = exec_locals[tool_name]

        # Execute the function with arguments
        if inspect.iscoroutinefunction(func):
            result = await func(**arguments)
        else:
            result = func(**arguments)

        return {"success": True, "result": result}

    except Exception as e:
        return {
            "error": f"Error executing generated tool '{tool_name}': {str(e)}",
            "type": type(e).__name__,
        }


def load_generated_tools() -> tuple[list[dict], dict[str, Callable]]:
    """
    Load all generated tools and return their schemas and execution functions.

    Returns:
        (tool_schemas, tool_functions)
    """
    tools = list_tools()

    tool_schemas = []
    tool_functions = {}

    def make_async_wrapper(tool_name: str):
        """Create an async wrapper for a tool with proper closure"""

        async def wrapper(**kwargs):
            return await execute_generated_tool(tool_name, kwargs)

        return wrapper

    for tool in tools:
        # Convert to OpenRouter format
        schema = {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"],
            },
        }
        tool_schemas.append(schema)

        # Create async execution wrapper
        tool_name = tool["name"]
        tool_functions[tool_name] = make_async_wrapper(tool_name)

    return tool_schemas, tool_functions


# ============================================
# Tool Registry - Convert to OpenRouter Format
# ============================================


def get_base_tools(firecrawl_api_key: str) -> tuple[list[dict], dict[str, Callable]]:
    """
    Returns base tools in OpenRouter format and their execution functions.

    Returns:
        (tool_schemas, tool_functions)
    """
    firecrawl = Firecrawl(firecrawl_api_key)

    # Tool schemas in OpenRouter format
    tool_schemas = [
        {
            "type": "function",
            "function": {
                "name": "file_read",
                "description": "Read content from a file in the artifacts directory. Supports both text and binary files. Binary files (images, PDFs) are returned as base64 encoded strings.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filepath": {
                            "type": "string",
                            "description": "Path to the file relative to artifacts directory (e.g., 'data.txt', 'images/cat.jpg')",
                        },
                        "mode": {
                            "type": "string",
                            "description": "Read mode: 'r' for text files, 'rb' for binary files (returns base64)",
                            "enum": ["r", "rb"],
                            "default": "r",
                        },
                        "encoding": {
                            "type": "string",
                            "description": "Text encoding (only for text mode, default: utf-8)",
                            "default": "utf-8",
                        },
                    },
                    "required": ["filepath"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "file_write",
                "description": "Write content to a file in the artifacts directory. Supports both text and binary data (images, PDFs, etc.). For binary data, use mode 'wb' and provide base64 encoded content.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filepath": {
                            "type": "string",
                            "description": "Path to the file relative to artifacts directory (e.g., 'output.txt', 'cat/image.jpg')",
                        },
                        "content": {
                            "type": "string",
                            "description": "Content to write. For text: plain string. For binary (images, etc.): base64 encoded string",
                        },
                        "mode": {
                            "type": "string",
                            "description": "Write mode: 'w' (text overwrite), 'a' (text append), 'wb' (binary overwrite), 'ab' (binary append)",
                            "enum": ["w", "a", "wb", "ab"],
                            "default": "w",
                        },
                        "encoding": {
                            "type": "string",
                            "description": "Text encoding (only for text mode, default: utf-8)",
                            "default": "utf-8",
                        },
                    },
                    "required": ["filepath", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "file_list",
                "description": "List files and directories in the artifacts directory. Use this to explore what files exist.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "directory": {
                            "type": "string",
                            "description": "Directory path relative to artifacts (default: '.' for root)",
                            "default": ".",
                        }
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "firecrawl_search",
                "description": "Search the web and get full page content. Use this to find information across the internet.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query"},
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results (1-100)",
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "firecrawl_scrape",
                "description": "Scrape content from a single webpage. Use this to extract content from a specific URL.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "The URL to scrape"},
                        "formats": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Content formats to extract (markdown, html, links, screenshot)",
                            "default": ["markdown"],
                        },
                        "only_main_content": {
                            "type": "boolean",
                            "description": "Extract only main content, excluding headers/footers/nav",
                            "default": True,
                        },
                    },
                    "required": ["url"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "firecrawl_crawl",
                "description": "Crawl an entire website starting from a URL. Use this to explore and extract content from multiple pages.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The starting URL to crawl",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of pages to crawl",
                            "default": 10,
                        },
                        "max_depth": {
                            "type": "integer",
                            "description": "Maximum depth of crawling",
                            "default": 2,
                        },
                    },
                    "required": ["url"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "generate_tool",
                "description": "Generate and save a new tool definition with executable code. Use this when you need a capability that doesn't exist yet.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "The name of the new tool (snake_case, must match function name)",
                        },
                        "description": {
                            "type": "string",
                            "description": "What the tool does",
                        },
                        "parameters": {
                            "type": "object",
                            "description": "JSON schema for the tool's parameters",
                        },
                        "code": {
                            "type": "string",
                            "description": "Python function code as a string. Must be a complete async function definition.",
                        },
                        "dependencies": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of required Python packages (e.g., ['httpx', 'beautifulsoup4'])",
                        },
                    },
                    "required": ["name", "description", "parameters", "code"],
                },
            },
        },
    ]

    # Tool execution functions
    tool_functions = {
        "file_read": file_read,
        "file_write": file_write,
        "file_list": file_list,
        "firecrawl_search": firecrawl.search,
        "firecrawl_scrape": firecrawl.scrape,
        "firecrawl_crawl": firecrawl.crawl,
        "generate_tool": lambda **kwargs: save_tool(kwargs),
    }

    return tool_schemas, tool_functions
