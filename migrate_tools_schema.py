"""
Database migration script to add enhanced fields to existing tools.
Run this once to update existing tools with UI-required fields.

Usage:
    python migrate_tools_schema.py
"""
import logging
from datetime import datetime, timezone
from services import db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def migrate_tools():
    """Add enhanced fields to all existing tools in MongoDB"""
    logger.info("Starting tool schema migration...")

    database = db.get_db()
    tools_collection = database.tools

    # Get all tools
    all_tools = list(tools_collection.find())
    logger.info(f"Found {len(all_tools)} tools to migrate")

    migrated_count = 0
    skipped_count = 0

    for tool in all_tools:
        tool_name = tool.get("name", "unknown")
        updates = {}

        # Add status if missing
        if "status" not in tool:
            updates["status"] = "PROD-READY"

        # Add category if missing
        if "category" not in tool:
            # Try to infer category from name
            name_lower = tool_name.lower()
            if any(x in name_lower for x in ["weather", "climate", "forecast"]):
                category = "weather"
            elif any(x in name_lower for x in ["crypto", "bitcoin", "ethereum", "price", "coin"]):
                category = "crypto"
            elif any(x in name_lower for x in ["instagram", "twitter", "facebook", "social"]):
                category = "social"
            elif any(x in name_lower for x in ["stripe", "payment", "charge"]):
                category = "payments"
            elif any(x in name_lower for x in ["github", "repo", "issue"]):
                category = "developer-tools"
            elif any(x in name_lower for x in ["slack", "email", "notify", "message"]):
                category = "communication"
            else:
                category = "general"
            updates["category"] = category

        # Add tags if missing
        if "tags" not in tool:
            tags = [tool_name]
            # Add category as tag
            if "category" in updates:
                tags.append(updates["category"])
            elif "category" in tool:
                tags.append(tool["category"])
            # Add "api" tag if description mentions API
            if "api" in tool.get("description", "").lower():
                tags.append("api")
            updates["tags"] = tags

        # Add verified if missing
        if "verified" not in tool:
            updates["verified"] = True

        # Add usage_count if missing
        if "usage_count" not in tool:
            updates["usage_count"] = 0

        # Add preview_snippet if missing
        if "preview_snippet" not in tool:
            params = tool.get("parameters", {}).get("properties", {})
            param_names = list(params.keys())
            if param_names:
                params_str = ", ".join(param_names[:3])  # Limit to first 3 params
                if len(param_names) > 3:
                    params_str += ", ..."
            else:
                params_str = ""
            updates["preview_snippet"] = f"{tool_name}({params_str})"

        # Perform update if there are changes
        if updates:
            logger.info(f"Migrating tool '{tool_name}': {list(updates.keys())}")
            tools_collection.update_one(
                {"_id": tool["_id"]},
                {"$set": updates}
            )
            migrated_count += 1
        else:
            logger.info(f"Tool '{tool_name}' already has all enhanced fields")
            skipped_count += 1

    logger.info("=" * 60)
    logger.info(f"Migration complete!")
    logger.info(f"  Migrated: {migrated_count} tools")
    logger.info(f"  Skipped: {skipped_count} tools (already had all fields)")
    logger.info(f"  Total: {len(all_tools)} tools")
    logger.info("=" * 60)

    # Print sample of updated tools
    logger.info("\nSample of updated tools:")
    sample_tools = list(tools_collection.find().limit(3))
    for tool in sample_tools:
        logger.info(f"\n  Tool: {tool['name']}")
        logger.info(f"    Status: {tool.get('status', 'N/A')}")
        logger.info(f"    Category: {tool.get('category', 'N/A')}")
        logger.info(f"    Tags: {tool.get('tags', [])}")
        logger.info(f"    Verified: {tool.get('verified', False)}")
        logger.info(f"    Usage Count: {tool.get('usage_count', 0)}")
        logger.info(f"    Preview: {tool.get('preview_snippet', 'N/A')}")


if __name__ == "__main__":
    try:
        migrate_tools()
    except Exception as e:
        logger.exception(f"Migration failed: {e}")
        raise
