import logging
import math

from bson import ObjectId
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

import voyageai
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

from services.env import MONGODB_URI, VOYAGE_API_KEY

logger = logging.getLogger(__name__)

voyage_client = voyageai.Client(api_key=VOYAGE_API_KEY)

# Global MongoDB client (singleton)
_client: Optional[MongoClient] = None
_db: Optional[Database] = None


def get_db() -> Database:
    """Get or create MongoDB database connection"""
    global _client, _db

    if _db is None:
        logger.info("Connecting to MongoDB...")
        _client = MongoClient(MONGODB_URI)
        _db = _client.get_default_database()
        logger.info("MongoDB connected successfully")

    return _db


def close_db():
    """Close MongoDB connection"""
    global _client, _db

    if _client:
        _client.close()
        _client = None
        _db = None
        logger.info("MongoDB connection closed")


# ============================================
# Embeddings
# ============================================


def generate_embedding(
    text: str, model: str = "voyage-4", input_type: str = "document"
) -> List[float]:
    """
    Generate embeddings for text using Voyage AI.

    Args:
        text: Text to embed
        model: Embedding model to use (default: voyage-4, 1024 dimensions)
        input_type: Type of input - "query" for search queries, "document" for documents to search

    Returns:
        List of floats representing the embedding vector
    """
    try:
        logger.info(f"Generating embedding for text: {text[:100]}...")
        result = voyage_client.embed(texts=[text], model=model, input_type=input_type)
        embedding = result.embeddings[0]
        logger.info(f"Generated embedding of dimension {len(embedding)}")
        return embedding
    except Exception as e:
        logger.exception(f"Error generating embedding: {e}")
        # Return zero vector as fallback (1024 dims for voyage-4)
        logger.warning("Returning zero vector as fallback")
        return [0.0] * 1024


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors"""

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    return dot_product / (magnitude1 * magnitude2)


# ============================================
# Conversations
# ============================================


def save_conversation(conversation_data: Dict[str, Any]) -> str:
    """
    Save a conversation to MongoDB.

    Args:
        conversation_data: Dictionary containing conversation details

    Returns:
        Inserted conversation ID
    """
    db = get_db()
    conversations: Collection = db.conversations

    # Add timestamp if not present
    if "created_at" not in conversation_data:
        conversation_data["created_at"] = datetime.utcnow()

    result = conversations.insert_one(conversation_data)
    conversation_id = str(result.inserted_id)

    logger.info(f"Conversation saved with ID: {conversation_id}")
    return conversation_id


def get_conversation(conversation_id: str) -> Optional[Dict[str, Any]]:
    """Get a conversation by ID"""
    db = get_db()
    conversations: Collection = db.conversations

    conversation = conversations.find_one({"_id": ObjectId(conversation_id)})

    if conversation:
        conversation["_id"] = str(conversation["_id"])

    return conversation


def list_conversations(limit: int = 50, skip: int = 0) -> List[Dict[str, Any]]:
    """List recent conversations"""
    db = get_db()
    conversations: Collection = db.conversations

    cursor = conversations.find().sort("created_at", -1).skip(skip).limit(limit)

    results = []
    for conv in cursor:
        conv["_id"] = str(conv["_id"])
        results.append(conv)

    return results


# ============================================
# Tools
# ============================================


def save_tool(tool_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Save a generated tool to MongoDB with vector embeddings and enhanced metadata.

    Args:
        tool_data: Tool definition with name, description, parameters, code, and optional metadata

    Returns:
        Result dictionary with success status
    """
    try:
        logger.info(f"Starting save_tool for: {tool_data.get('name', 'unknown')}")

        db = get_db()
        tools: Collection = db.tools
        logger.info(f"Got database connection: {db.name}")

        # Add timestamp
        tool_data["created_at"] = datetime.now(timezone.utc)

        # Set default enhanced fields for UI compatibility (BACKEND_API_REQUIREMENTS.md)
        tool_data.setdefault("status", "PROD-READY")
        tool_data.setdefault("category", "general")
        tool_data.setdefault("tags", [tool_data["name"]])
        tool_data.setdefault("verified", True)
        tool_data.setdefault("usage_count", 0)

        # Generate preview snippet if not provided
        if "preview_snippet" not in tool_data:
            params_str = ", ".join(
                tool_data.get("parameters", {}).get("properties", {}).keys()
            )
            tool_data["preview_snippet"] = f"{tool_data['name']}({params_str})"

        # Generate embedding from tool name and description
        embedding_text = f"{tool_data['name']}: {tool_data['description']}"
        logger.info("Generating embedding...")
        embedding = generate_embedding(embedding_text)
        tool_data["embedding"] = embedding
        logger.info(f"Embedding generated, length: {len(embedding)}")

        # Remove _id field if present (can't update immutable _id field)
        update_data = {k: v for k, v in tool_data.items() if k != "_id"}

        # Update if exists, insert if new (upsert based on name)
        logger.info(f"Saving to MongoDB collection: {tools.name}")
        result = tools.update_one(
            {"name": update_data["name"]}, {"$set": update_data}, upsert=True
        )

        logger.info(
            f"Tool '{tool_data['name']}' saved to MongoDB. Upserted: {result.upserted_id is not None}, Modified: {result.modified_count}"
        )

        return {
            "success": True,
            "message": f"Tool '{tool_data['name']}' saved successfully",
            "has_code": "code" in tool_data and tool_data["code"] is not None,
            "upserted": result.upserted_id is not None,
        }
    except Exception as e:
        logger.exception(f"Error in save_tool: {e}")
        return {"success": False, "error": str(e)}


def get_tool(name: str) -> Optional[Dict[str, Any]]:
    """Get a tool by name"""
    db = get_db()
    tools: Collection = db.tools

    tool = tools.find_one({"name": name})

    if tool:
        tool["_id"] = str(tool["_id"])

    return tool


def list_tools() -> List[Dict[str, Any]]:
    """List all tools"""
    db = get_db()
    tools: Collection = db.tools

    cursor = tools.find().sort("created_at", -1)

    results = []
    for tool in cursor:
        tool["_id"] = str(tool["_id"])
        results.append(tool)

    return results


def delete_tool(name: str) -> bool:
    """Delete a tool by name"""
    db = get_db()
    tools: Collection = db.tools

    result = tools.delete_one({"name": name})
    deleted = result.deleted_count > 0

    if deleted:
        logger.info(f"Tool '{name}' deleted from MongoDB")

    return deleted


def search_tools(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Search for tools using vector similarity.
    Returns the top N most relevant tools based on the query.

    Args:
        query: Search query describing what the tool should do
        limit: Maximum number of tools to return (default: 10)

    Returns:
        List of tool definitions sorted by relevance
    """
    db = get_db()
    tools: Collection = db.tools

    # Generate embedding for the query with input_type="query" for optimal retrieval
    query_embedding = generate_embedding(query, input_type="query")

    # Get all tools with embeddings
    all_tools = list(tools.find({"embedding": {"$exists": True}}))

    if not all_tools:
        logger.warning("No tools with embeddings found in database")
        return []

    # Calculate similarity scores
    tools_with_scores = []
    for tool in all_tools:
        if "embedding" in tool:
            similarity = cosine_similarity(query_embedding, tool["embedding"])
            tool["_id"] = str(tool["_id"])
            tool["similarity_score"] = similarity
            tools_with_scores.append(tool)

    # Sort by similarity (highest first) and return top N
    tools_with_scores.sort(key=lambda x: x["similarity_score"], reverse=True)
    top_tools = tools_with_scores[:limit]

    logger.info(f"Found {len(top_tools)} tools matching query: '{query}'")
    if top_tools:
        logger.info(
            f"Top match: {top_tools[0]['name']} (score: {top_tools[0]['similarity_score']:.3f})"
        )

    return top_tools
