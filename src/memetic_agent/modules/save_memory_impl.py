from datetime import datetime
from log_config import log_error, log_event
from filelock import FileLock
from src.base_agent.type import Agent

async def save_memory_impl(agent: Agent) -> None:
    """Save new messages from conversations to memory store."""
    try:
        # Get the latest timestamp for each conversation from short-term memory
        short_term = agent.memory._get_collection("short_term")
        latest_timestamps = {}
        
        for conversation_id in agent.conversations:
            try:
                # Protect against ChromaDB query failures
                results = short_term.get(
                    where={"conversation_id": conversation_id}
                )
                
                # Validate results structure
                if not isinstance(results, dict) or "metadatas" not in results:
                    log_error(agent.logger, 
                                f"Invalid results format for conversation {conversation_id}")
                    continue
                
                # Safely extract timestamps with validation
                if results["metadatas"]:
                    try:
                        timestamps = []
                        for metadata in results["metadatas"]:
                            if not isinstance(metadata, dict):
                                continue
                            timestamp = metadata.get("timestamp")
                            if timestamp:
                                # Validate timestamp format
                                try:
                                    datetime.fromisoformat(timestamp)
                                    timestamps.append(timestamp)
                                except ValueError:
                                    log_error(agent.logger,
                                            f"Invalid timestamp format in metadata: {timestamp}")
                                    continue
                        latest_timestamps[conversation_id] = max(timestamps) if timestamps else None
                    except Exception as e:
                        log_error(agent.logger,
                                f"Error processing timestamps for conversation {conversation_id}: {str(e)}")
                        latest_timestamps[conversation_id] = None
                else:
                    latest_timestamps[conversation_id] = None
                
            except Exception as e:
                log_error(agent.logger,
                            f"Failed to get timestamps for conversation {conversation_id}: {str(e)}")
                latest_timestamps[conversation_id] = None

        # Save only new messages for each conversation
        for conversation_id, conversation in agent.conversations.items():
            try:
                latest_timestamp = latest_timestamps.get(conversation_id)
                
                # Filter and validate messages
                new_messages = []
                for msg in conversation:
                    try:
                        if msg.role == "system":
                            continue
                            
                        # Validate message attributes
                        if not all(hasattr(msg, attr) for attr in ['content', 'role', 'timestamp']):
                            log_error(agent.logger,
                                    f"Message missing required attributes in conversation {conversation_id}")
                            continue
                            
                        # Validate timestamp format
                        try:
                            msg_time = datetime.fromisoformat(msg.timestamp)
                        except (ValueError, TypeError):
                            log_error(agent.logger,
                                    f"Invalid message timestamp format in conversation {conversation_id}")
                            continue
                            
                        # Compare timestamps if we have a latest_timestamp
                        if latest_timestamp:
                            try:
                                if msg_time > datetime.fromisoformat(latest_timestamp):
                                    new_messages.append(msg)
                            except ValueError:
                                log_error(agent.logger,
                                        f"Timestamp comparison failed in conversation {conversation_id}")
                                continue
                        else:
                            new_messages.append(msg)
                            
                    except Exception as e:
                        log_error(agent.logger,
                                f"Error processing message in conversation {conversation_id}: {str(e)}")
                        continue

                if not new_messages:
                    continue

                # Format new messages with validation
                try:
                    formatted_messages = []
                    for msg in new_messages:
                        if not isinstance(msg.content, str) or not isinstance(msg.role, str):
                            log_error(agent.logger,
                                    f"Invalid message format in conversation {conversation_id}")
                            continue
                        formatted_messages.append(f"{msg.role}: {msg.content}")
                    
                    content = "\n".join(formatted_messages)
                    
                    # Validate content is not empty
                    if not content.strip():
                        log_error(agent.logger,
                                f"Empty content generated for conversation {conversation_id}")
                        continue
                        
                    # Get participants with validation
                    participants = set()
                    for msg in new_messages:
                        participant = getattr(msg, 'sender', None) or msg.role
                        if isinstance(participant, str):
                            participants.add(participant)
                    
                    participants_str = ",".join(sorted(participants))
                    
                    metadata = {
                        "conversation_id": conversation_id,
                        "timestamp": datetime.now().isoformat(),
                        "participants": participants_str
                    }
                    
                except Exception as e:
                    log_error(agent.logger,
                                f"Failed to format messages for conversation {conversation_id}: {str(e)}")
                    continue

                # Save to memory store
                try:
                    await agent.memory.store(
                        content=content,
                        collection_name="short_term",
                        metadata=metadata
                    )
                    log_event(agent.logger, "memory.saved", 
                                f"Saved {len(new_messages)} new messages for conversation {conversation_id}")
                except Exception as e:
                    log_error(agent.logger,
                                f"Failed to save to memory store for conversation {conversation_id}: {str(e)}")
                    continue
                
                # Save to disk with error handling
                try:
                    memory_dump_path = agent.files_path / f"{agent.config.agent_name}_short_term_memory_dump.md"
                    
                    # Prepare content before acquiring lock
                    dump_content = (
                        f"\n\n## Memory Entry {metadata['timestamp']}\n"
                        f"conversation_id: {metadata['conversation_id']}\n"
                        f"participants: {metadata['participants']}\n"
                        f"timestamp: {metadata['timestamp']}\n\n"
                        f"{content}\n---\n"
                    )
                    
                    try:
                        with FileLock(f"{memory_dump_path}.lock", timeout=10):  # 10 second timeout
                            with open(memory_dump_path, "a", encoding="utf-8") as f:
                                f.write(dump_content)
                        log_event(agent.logger, "memory.dumped", 
                                f"Saved conversation {conversation_id} to disk")
                    except TimeoutError:
                        log_error(agent.logger,
                                f"Failed to acquire file lock for conversation {conversation_id}")
                    
                except Exception as e:
                    log_error(agent.logger,
                                f"Failed to save to disk for conversation {conversation_id}: {str(e)}")
                    # Continue processing as memory store save was successful
                    
            except Exception as e:
                log_error(agent.logger,
                            f"Failed to process conversation {conversation_id}: {str(e)}")
                continue
            
    except Exception as e:
        log_error(agent.logger, "Failed to save memory", exc_info=e)
