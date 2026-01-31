import os

from dotenv import load_dotenv

load_dotenv("dev.env")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", None)
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY", None)