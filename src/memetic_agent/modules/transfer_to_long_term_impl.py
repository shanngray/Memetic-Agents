from src.base_agent.type import Agent
from src.log_config import log_event, log_error
from datetime import datetime, timedelta
from base_agent.models import AgentStatus
from typing import List, Dict, Any
import uuid
import json

async def transfer_to_long_term_impl(agent: Agent, days_threshold: int = 0) -> None:
    """Transfer short-term memories into long-term storage as atomic memories with SPO metadata.
    
    Args:
        days_threshold: Number of days worth of memories to keep in short-term. 
                        Memories older than this will be processed into long-term storage.
                        Default is 0 (process all memories).
    """
    try:
        if agent.status != AgentStatus.MEMORISING:
            log_error(agent.logger, "Agent must be in MEMORISING state to transfer to long-term")
            return

        log_event(agent.logger, "agent.memorising", "Beginning atomic memory extraction process", level="DEBUG")
        
        # Retrieve recent memories from short-term storage
        short_term_memories = await agent.memory.retrieve(
            query="",  # Empty query to get all memories
            collection_names=["short_term"],
            n_results=100
        )
        
        # Add debug logging
        for memory in short_term_memories:
            log_event(agent.logger, "memory.debug", 
                        f"Memory structure: {memory}", level="DEBUG")

        # Filter memories based on threshold
        threshold_date = datetime.now() - timedelta(days=days_threshold)
        short_term_memories = [
            memory for memory in short_term_memories
            if datetime.fromisoformat(memory["metadata"].get("timestamp", "")) < threshold_date
        ]
        
        if not short_term_memories:
            log_error(agent.logger, "memory.error", "No recent memories found for atomization")
            return

        # Group memories by conversation ID
        conversation_memories = {}
        for memory in short_term_memories:
            conv_id = memory["metadata"].get("conversation_id")
            if conv_id not in conversation_memories:
                conversation_memories[conv_id] = []
            conversation_memories[conv_id].append(memory)

        # Sort memories within each conversation by timestamp
        for conv_id in conversation_memories:
            conversation_memories[conv_id].sort(
                key=lambda x: datetime.fromisoformat(x["metadata"].get("timestamp", ""))
            )

        # Process each conversation group
        for conv_id, memories in conversation_memories.items():
            try:
                # Combine memory content in chronological order
                combined_content = "\n".join(memory["content"] for memory in memories)
                
                # Store original metadata from first memory in conversation
                original_metadata = memories[0]["metadata"] if memories else {}
                
                feedback_items = await agent.memory.retrieve(
                    query="",
                    collection_names=["feedback"],
                    n_results=100,
                    metadata_filter={"conversation_id": conv_id}
                )

                # Add feedback content if any exists
                if feedback_items:
                    feedback_content = "\n".join(item["content"] for item in feedback_items)
                    combined_content += f"\n\n{feedback_content}"

                log_event(agent.logger, "memory.transfer.content",
                            f"Combined content:\n{combined_content[:100]}...",
                            level="DEBUG")
                # Extract atomic memories using LLM
                
                full_prompt = agent.prompt.xfer_long_term.content + "\n\nFormat your response as a JSON array of objects with the following schema:\n" + agent.prompt.xfer_long_term.schema_content

                atomic_response = await agent.client.chat.completions.create(
                    model=agent.config.submodel,
                    messages=[
                        {"role": "system", "content": full_prompt},
                        {"role": "user", "content": f"Memory content:\n{combined_content}"}
                    ],
                    response_format={ "type": "json_object" }
                )
                
                # Parse and validate the LLM response
                raw_response = atomic_response.choices[0].message.content
                log_event(agent.logger, "memory.atomic.response", f"Raw LLM response: {raw_response}", level="DEBUG")

                try:
                    atomic_memories = json.loads(raw_response)
                    # Check if response has memories/atomic_memories key and convert to list
                    if isinstance(atomic_memories, dict):
                        for key in ["memories", "atomic_memories", "atomized_memories", "output_format"]:
                            if key in atomic_memories:
                                atomic_memories = atomic_memories[key]
                                break
                    
                    # Ensure atomic_memories is a list
                    if not isinstance(atomic_memories, list):
                        atomic_memories = [atomic_memories]
                    
                    # Validate each memory has required fields
                    validated_memories = []
                    required_fields = ["statement", "metatags", "thoughts", "confidence", "category"]
                    
                    for memory in atomic_memories:
                        if isinstance(memory, dict) and all(k in memory for k in required_fields):
                            validated_memories.append(memory)
                        else:
                            log_event(agent.logger, "memory.atomic.invalid", 
                                        f"Invalid memory format, missing required fields: {memory}",
                                        level="DEBUG")
                    
                    if not validated_memories:
                        # If no valid memories, retry with LLM with more explicit instructions
                        retry_prompt = (
                            f"{agent.prompt.xfer_long_term.content}\n\n"
                            "IMPORTANT: Your response must be a JSON array of objects with this exact structure:\n"
                            f"{agent.prompt.xfer_long_term.schema_content}"
                        )
                        
                        retry_response = await agent.client.chat.completions.create(
                            model=agent.config.submodel,
                            messages=[
                                {"role": "system", "content": retry_prompt},
                                {"role": "user", "content": f"Previous response was invalid. Please reformat this content into the exact structure specified:\n{raw_response}"}
                            ],
                            response_format={ "type": "json_object" }
                        )
                        
                        retry_content = retry_response.choices[0].message.content
                        log_event(agent.logger, "memory.atomic.retry", f"Retry response: {retry_content}", level="DEBUG")
                        
                        try:
                            retry_json = json.loads(retry_content)
                            if isinstance(retry_json, dict) and "atomic_memories" in retry_json:
                                validated_memories = retry_json["atomic_memories"]
                            else:
                                log_error(agent.logger, "Retry failed to produce valid format")
                                continue
                        except json.JSONDecodeError:
                            log_error(agent.logger, f"Failed to parse retry response as JSON: {retry_content}")
                            continue

                    # Continue processing with validated_memories
                    for atomic in validated_memories:
                        try:
                            # Get original memory metadata safely
                            original_metadata = memory.get("metadata", {})
                            
                            metadata = {
                                "memory_id": str(uuid.uuid4()),
                                "original_timestamp": original_metadata.get("timestamp", datetime.now().isoformat()),
                                "source_type": "atomic_memory",
                                "confidence": atomic.get("confidence", 0.5),  # Default confidence if missing
                                "category": atomic.get("category", "uncategorized"),  # Default category if missing
                                "timestamp": datetime.now().isoformat(),
                                "original_memory_id": original_metadata.get("chroma_id", "unknown"),  # Track original memory
                                "conversation_id": original_metadata.get("conversation_id", "unknown")
                            }

                            formatted_memory = (
                                f"{atomic['statement']}\n\n"
                                f"MetaTags: {', '.join(atomic.get('metatags', []))}\n\n"
                                f"Thoughts: {atomic.get('thoughts', 'No additional thoughts')}"
                            )
                            
                            await agent.memory.store(
                                content=atomic["statement"],
                                collection_name="long_term",
                                metadata=metadata
                            )
                            
                            log_event(agent.logger, "memory.atomic.stored",
                                        f"Stored atomic memory: {atomic['statement']} ({atomic.get('category', 'uncategorized')})",
                                        level="DEBUG")
                        except Exception as e:
                            log_error(agent.logger, f"Failed to store atomic memory: {str(e)}")
                            continue

                    # Save to disk with the stored original metadata
                    await agent._save_atomic_memory_to_disk(atomic_memories, original_metadata)

                except json.JSONDecodeError:
                    log_error(agent.logger, f"Failed to parse LLM response as JSON: {raw_response}")
                    raise

            except Exception as e:
                log_error(agent.logger, f"Failed to process conversation {conv_id} into atomic form: {str(e)}", exc_info=e)
                continue
        
        log_event(agent.logger, "memory.atomization.complete",
                    f"Completed memory atomization for {len(short_term_memories)} memories",
                    level="DEBUG")
            
    except Exception as e:
        log_error(agent.logger, "Failed to atomize memories", exc_info=e)
