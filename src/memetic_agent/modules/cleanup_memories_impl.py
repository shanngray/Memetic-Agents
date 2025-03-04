from src.base_agent.type import Agent
from src.log_config import log_event, log_error
from datetime import datetime, timedelta

async def cleanup_memories_impl(agent: Agent, days_threshold: int = 0, collection_name: str = "short_term") -> None:
    """Clean up old memories from specified collection after consolidation.
    
    Args:
        days_threshold: Number of days worth of memories to keep.
                        Memories older than this will be deleted.
                        Default is 0 (clean up all processed memories).
        collection_name: Name of collection to clean up.
                        Default is "short_term".
    """
    try:
        # Ensure days_threshold is an integer
        try:
            days_threshold = int(days_threshold)
        except (TypeError, ValueError):
            log_error(agent.logger, f"Invalid days_threshold value: {days_threshold}. Using default of 0.")
            days_threshold = 0

        log_event(agent.logger, "memory.cleanup.start", 
                    f"Starting cleanup of {collection_name} memories older than {days_threshold} days")
        
        # Retrieve all memories from specified collection
        old_memories = await agent.memory.retrieve(
            query="",
            collection_names=[collection_name],
            n_results=1000
        )
        
        # Filter based on threshold
        cutoff_time = datetime.now() - timedelta(days=days_threshold)
        old_memories = [
            memory for memory in old_memories
            if datetime.fromisoformat(memory["metadata"].get("timestamp", "")) < cutoff_time
        ]
        
        # Track conversation IDs to clean up
        conversations_to_remove = set()
        
        # Delete old memories using the ChromaDB ID from metadata
        deleted_count = 0
        for memory in old_memories:
            chroma_id = memory["metadata"].get("chroma_id")
            
            # Track conversation ID for cleanup
            if collection_name == "short_term":
                conversation_id = memory["metadata"].get("conversation_id")
                if conversation_id:
                    conversations_to_remove.add(conversation_id)
                    agent.old_conversation_list[conversation_id] = "placeholder_name"
            
            if chroma_id:
                await agent.memory.delete(chroma_id, collection_name)
                deleted_count += 1
        
        # Clean up conversations from memory
        for conv_id in conversations_to_remove:
            if conv_id in agent.conversations:
                del agent.conversations[conv_id]
                log_event(agent.logger, "memory.cleanup", 
                            f"Removed conversation {conv_id} from active conversations")
        
        log_event(agent.logger, "memory.cleanup", 
                    f"Cleaned up {deleted_count} memories from {collection_name} collection and {len(conversations_to_remove)} conversations")
            
    except Exception as e:
        log_error(agent.logger, f"Failed to cleanup {collection_name} memories", exc_info=e)