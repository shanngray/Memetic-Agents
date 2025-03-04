from src.base_agent.type import Agent
from src.log_config import log_event, log_error
from datetime import datetime, timedelta
from base_agent.models import AgentStatus
from typing import List, Dict, Any
import uuid
import json

async def extract_learnings_impl(agent: Agent, days_threshold: int = 0) -> None:
    """Extract learning opportunities from short-term memories and feedback.
    
    Args:
        days_threshold: Number of days worth of memories to keep in short-term. 
                        Memories older than this will be processed into long-term storage.
                        Default is 0 (process all memories).
    """
    try:
        if agent.status != AgentStatus.MEMORISING:
            log_error(agent.logger, "Agent must be in MEMORISING state to extract learnings")
            return
            # await self.set_status(AgentStatus.MEMORISING, "Extracting Learnings triggered")
        
        log_event(agent.logger, "agent.memorising", "Beginning learning memory extraction process", level="DEBUG")
        
        # Retrieve and filter memories
        short_term_memories = await agent.memory.retrieve(
            query="",
            collection_names=["short_term"],
            n_results=1000  # Hard limit (1000 entries) prevents infinite processing
        )
        
        # Filter memories based on threshold
        threshold_date = datetime.now() - timedelta(days=days_threshold)
        short_term_memories = [
            memory for memory in short_term_memories
            if datetime.fromisoformat(memory["metadata"].get("timestamp", "")) < threshold_date
        ]
        
        total_memories = len(short_term_memories)
        log_event(agent.logger, "memory.extraction.progress", 
                    f"Found {total_memories} memories to process")
        
        if not short_term_memories:
            log_error(agent.logger, "memory.error", "No recent memories found for reflection")
            return

        # Group and sort memories
        conversation_memories = {}
        for memory in short_term_memories:
            conv_id = memory["metadata"].get("conversation_id")
            if conv_id not in conversation_memories:
                conversation_memories[conv_id] = []
            conversation_memories[conv_id].append(memory)

        total_conversations = len(conversation_memories)
        log_event(agent.logger, "memory.extraction.progress", 
                    f"Grouped into {total_conversations} conversations")

        # Process each conversation group
        for conv_idx, (conv_id, memories) in enumerate(conversation_memories.items(), 1):
            try:
                log_event(agent.logger, "memory.extraction.progress", 
                            f"Processing conversation {conv_idx} of {total_conversations}")
                
                # Sort memories within conversation by timestamp
                memories.sort(key=lambda x: datetime.fromisoformat(x["metadata"].get("timestamp", "")))
                
                # Combine memory content
                combined_content = "\n".join(memory["content"] for memory in memories)
                
                # Get feedback for this conversation
                feedback_items = await agent.memory.retrieve(
                    query="",
                    collection_names=["feedback"],
                    n_results=100,  # Hard limit prevents infinite feedback processing
                    metadata_filter={"conversation_id": conv_id}
                )

                if feedback_items:
                    feedback_content = "\n".join(item["content"] for item in feedback_items)
                    combined_content += f"\n\nFeedback:\n{feedback_content}"

                full_prompt = agent.prompt.reflect_memories.content + "\n\nFormat your response as a JSON array of objects with the following schema:\n" + agent.prompt.reflect_memories.schema_content

                # Extract learnings using LLM
                reflection_response = await agent.client.chat.completions.create(
                    model=agent.config.submodel,
                    messages=[
                        {"role": "system", "content": full_prompt},
                        {"role": "user", "content": f"Memory content:\n{combined_content}"}
                    ],
                    response_format={ "type": "json_object" }
                )
                
                raw_response = reflection_response.choices[0].message.content
                
                # Process reflections with one retry attempt
                try:
                    reflections = process_reflection_response(agent, raw_response)
                    if not reflections:  # If initial processing fails, try retry prompt
                        reflections = await retry_reflection_processing(agent, 
                            agent.prompt.reflect_memories.content, raw_response
                        )
                        if not reflections:  # If retry fails, skip this conversation
                            log_event(agent.logger, "memory.reflection.skip", 
                                        f"Skipping conversation {conv_id} due to invalid response format")
                            continue
                    
                    # Store validated reflections
                    total_reflections = len(reflections)
                    for refl_idx, reflection in enumerate(reflections, 1):
                        log_event(agent.logger, "memory.reflection.progress", 
                                    f"Storing reflection {refl_idx} of {total_reflections} for conversation {conv_idx}")
                        
                        await store_reflection(agent, reflection, conv_id, memories[0]["metadata"])
                    
                    # Save to disk for debugging
                    await agent._save_reflection_to_disk(reflections, memories[0]["metadata"])
                    
                except Exception as e:
                    log_error(agent.logger, f"Failed to process reflections for conversation {conv_id}: {str(e)}")
                    continue

            except Exception as e:
                log_error(agent.logger, f"Failed to process conversation {conv_id}: {str(e)}")
                continue
        
        log_event(agent.logger, "memory.reflection.complete",
                    f"Completed memory reflection for {total_conversations} conversations")
            
    except Exception as e:
        log_error(agent.logger, "Failed to process reflections", exc_info=e)

def process_reflection_response(agent: Agent, raw_response: str) -> List[Dict]:
    """Process and validate reflection response from LLM."""
    try:
        reflections = json.loads(raw_response)
        # Check for various possible key names
        if isinstance(reflections, dict):
            for key in ["learning_opportunities", "lessons", "reflections", "learnings", "output_format"]:
                if key in reflections:
                    reflections = reflections[key]
                    break
        
        # Ensure reflections is a list
        if not isinstance(reflections, list):
            reflections = [reflections]
        
        # Validate required fields
        validated_reflections = []
        required_fields = ["lesson", "importance", "category", "thoughts"]
        
        for reflection in reflections:
            # Check if reflection is a dictionary and has all required fields
            if isinstance(reflection, dict) and all(k in reflection for k in required_fields):
                # Check if category is valid
                if reflection["category"] in [
                    "tools", "agentic structure", "give feedback", "feedback reflection", "memory reflection", "long term memory transfer", 
                    "thought loop", "reasoning", "self improvement", "system prompt", "evaluator"
                ]:
                    validated_reflections.append(reflection)
                else:
                    log_error(agent.logger, f"Invalid reflection category: {reflection['category']}")
            else:
                log_error(agent.logger, f"Invalid reflection format: {reflection}")
        
        return validated_reflections
    except json.JSONDecodeError:
        return []

async def retry_reflection_processing(agent: Agent, base_prompt: str, failed_response: str) -> List[Dict]:
    """Retry processing reflections with more explicit instructions."""
    retry_prompt = (
        f"{base_prompt}\n\n"
        "IMPORTANT: Your response must be a JSON object with this exact structure:\n"
        '{\n'
        '    "learning_opportunities": [\n'
        '        {\n'
        '            "lesson": "The main learning point",\n'
        '            "importance": 0.8,\n'
        '            "category": "category_name",\n'
        '            "thoughts": "Additional context and reasoning"\n'
        '        }\n'
        '    ]\n'
        '}'
    )
    
    retry_response = await agent.client.chat.completions.create(
        model=agent.config.submodel,
        messages=[
            {"role": "system", "content": retry_prompt},
            {"role": "user", "content": f"Previous response was invalid. Please reformat this content into the exact structure specified:\n{failed_response}"}
        ],
        response_format={ "type": "json_object" }
    )
    
    return process_reflection_response(agent, retry_response.choices[0].message.content)

async def store_reflection(agent: Agent, reflection: Dict, conv_id: str, original_metadata: Dict) -> None:
    """Store a single reflection in memory."""
    metadata = {
        "memory_id": str(uuid.uuid4()),
        "conversation_id": conv_id,
        "original_timestamp": original_metadata.get("timestamp"),
        "source_type": "learning_reflection",
        "importance": normalise_score(agent, reflection["importance"]),
        "category": reflection["category"],
        "timestamp": datetime.now().isoformat()
    }

    content = (
        f"Lesson: {reflection['lesson']}\n"
        f"Thoughts: {reflection['thoughts']}\n"
    )

    await agent.memory.store(
        content=reflection["lesson"],
        collection_name="reflections",
        metadata=metadata
    )
    
    log_event(agent.logger, "memory.reflection.stored",
                f"Stored learning reflection: {reflection['lesson']} ({reflection['category']})", level="DEBUG")

def normalise_score(agent: Agent, score: Any) -> float:
    """Convert various score inputs to a float between 0-1.
    
    Handles:
    - Strings (including percentages)
    - Floats
    - Invalid types by defaulting to 0.5
    """
    try:
        # Log original value and type
        log_event(agent.logger, "confidence.evaluation", 
                 f"Original score value: {score} (type: {type(score).__name__})", 
                 level="DEBUG")
        
        # Handle string inputs
        if isinstance(score, str):
            # Remove any non-numeric characters except decimal points
            score = ''.join(c for c in score if c.isdigit() or c == '.')
            if not score:
                return 0.5  # Default if string contains no numbers
            score = float(score)
        
        # Convert to float
        score_float = float(score)
        
        # Handle values larger than 1
        if score_float > 1:
            score_float = score_float / 100 if score_float <= 100 else 1.0
        
        # Clamp between 0-1
        normalised_score = max(0.0, min(1.0, score_float))
        
        log_event(agent.logger, "confidence.evaluation", 
                 f"Normalised score: {normalised_score} (from original: {score} of type {type(score).__name__})", 
                 level="DEBUG")
        
        return normalised_score
        
    except (ValueError, TypeError) as e:
        log_error(agent.logger, 
                 f"Score normalization failed for value '{score}' of type {type(score).__name__}. Using default score of 0.5.")
        return 0.5