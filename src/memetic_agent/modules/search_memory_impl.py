from typing import Dict, Any, Optional, List
from datetime import datetime
from src.log_config import log_event, log_error
from src.base_agent.type import Agent


async def search_memory_impl(agent: Agent, query: str, keywords: Optional[List[str]] = None) -> Dict[str, Any]:
    """Use to search your memory using a 'query' and (optional) 'keywords'.
    
    Args:
        query: The search query string
        keywords: Optional list of keywords to help filter results
        
    Returns:
        Dict containing search results and metadata
    """
    try:
        # Ensure keywords is a list if provided
        if keywords and not isinstance(keywords, list):
            keywords = [keywords]

        # Create metadata filter if keywords provided
        metadata_filter = None
        if keywords:
            metadata_filter = {
                "content": {"$in": keywords}
            }

        # Search both short and long term memory
        memories = await agent.memory.retrieve(
            query=query,
            collection_names=["short_term", "long_term"],
            n_results=5,
            metadata_filter=metadata_filter
        )

        log_event(agent.logger, "memory.searched", 
                    f"Memory search for '{query}' returned {len(memories)} results")

        return {
            "query": query,
            "results": memories,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        error_msg = f"Failed to search memory: {str(e)}"
        log_error(agent.logger, error_msg, exc_info=e)
        return {
            "query": query,
            "error": error_msg,
            "timestamp": datetime.now().isoformat()
        }