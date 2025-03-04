from src.base_agent.type import Agent
from src.log_config import log_event, log_error
from datetime import datetime
import json
from src.base_agent.models import Message

async def load_memory_impl(agent: Agent) -> None:
    """Load short term memories into conversations and list of long term memories into old_conversation_list."""
    try:
        log_event(agent.logger, "memory.load.start", "Beginning memory load process")
        
        # Initialize empty containers
        agent.conversations = {}
        agent.old_conversation_list = {}
        
        collection_names = agent.memory.collections.keys()

        # Ensure collections are initialized before proceeding
        for collection_name in collection_names:
            if collection_name not in agent.memory.collections:
                raise ValueError(f"Collection '{collection_name}' not found. Available collections: {list(agent.memory.collections.keys())}")

        # Part 1: Load conversations from short-term memory
        short_term_collection = agent.memory._get_collection("short_term")
        results = short_term_collection.get()
        
        if not results["documents"]:
            log_event(agent.logger, "memory.load.complete", 
                        "Short-term memory collection is empty - starting fresh")
            return
        
        # Group results by conversation_id
        conversation_groups: Dict[str, Dict[str, Any]] = {}
        for doc, metadata in zip(results["documents"], results["metadatas"]):
            conv_id = metadata.get("conversation_id")
            if conv_id:
                if conv_id not in conversation_groups:
                    conversation_groups[conv_id] = {
                        "content": [],
                        "participants": set(),
                        "timestamps": []
                    }
                conversation_groups[conv_id]["content"].append(doc)
                if "participants" in metadata:
                    conversation_groups[conv_id]["participants"].update(
                        metadata["participants"].split(",")
                    )
                if "timestamp" in metadata:
                    conversation_groups[conv_id]["timestamps"].append(metadata["timestamp"])

        # Convert grouped content into conversations
        loaded_conversations = 0
        for conv_id, group in conversation_groups.items():
            try:
                # Start with system prompt
                messages = [Message(role="user" if agent.config.model == "o1-mini" else "developer" if agent.config.model == "o3-mini" else "system", content=agent.prompt.system.content)]
                
                # Combine all content for this conversation
                combined_content = "\n".join(group["content"])
                
                # Parse the combined content into messages
                message_chunks = combined_content.split("\n")
                for chunk in message_chunks:
                    if chunk.strip():
                        # Try to parse role and content from chunk
                        if ": " in chunk:
                            role, content = chunk.split(": ", 1)
                            # Convert role to standard format
                            role = role.lower()
                            if role not in ["system", "user", "assistant", "tool"]:
                                role = "user"
                        else:
                            # Default to user role if format is unclear
                            role = "user"
                            content = chunk
                            
                        messages.append(Message(
                            role=role,
                            content=content,
                            timestamp=min(group["timestamps"]) if group["timestamps"] else datetime.now().isoformat()
                        ))
            
                agent.conversations[conv_id] = messages
                loaded_conversations += 1
                log_event(agent.logger, "memory.loaded", 
                            f"Loaded conversation {conv_id} with {len(messages)} messages")
                
            except Exception as e:
                log_error(agent.logger, 
                            f"Error parsing conversation {conv_id}: {str(e)}")
                continue

        # Part 2: Load old conversations list
        old_conversations_file = agent.files_path / "old_conversations.json"
        if old_conversations_file.exists():
            try:
                with open(old_conversations_file, "r") as f:
                    agent.old_conversation_list = json.load(f)
                log_event(agent.logger, "memory.loaded", 
                            f"Loaded {len(agent.old_conversation_list)} old conversations")
            except json.JSONDecodeError as e:
                log_error(agent.logger, 
                            f"Error loading old conversations: {str(e)}")
                # Initialize empty if file is corrupted
                agent.old_conversation_list = {}
        else:
            # Initialize empty if file doesn't exist
            agent.old_conversation_list = {}

        log_event(agent.logger, "memory.load.complete", 
                    f"Successfully loaded {loaded_conversations} conversations")

    except Exception as e:
        log_error(agent.logger, "Failed to load memory", exc_info=e)
        # Initialize empty containers on error
        agent.conversations = {}
        agent.old_conversation_list = {}