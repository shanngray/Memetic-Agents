from src.base_agent.type import Agent
from src.base_agent.models import AgentStatus
from src.log_config import log_event, log_error, log_agent_message
from typing import Any, Dict
import json

async def _get_highest_priority_category(agent) -> str | None:
    """Get the reflection category with highest total importance score.
    
    Returns:
        str | None: Category with highest importance score, or None if no valid categories found
    """
    try:
        # Categories to ignore as they're not yet implemented
        IGNORED_CATEGORIES = {
            "tools",  # Tool learning not implemented
            "agentic structure",  # Architecture learning not implemented
            "insight"  # Insight learning not implemented
        }

        # Get all reflections
        reflections = await agent.memory.retrieve(
            query="",
            collection_names=["reflections"],
            n_results=1000  # Set high to get all reflections
        )

        if not reflections:
            log_event(agent.logger, "learning.priority", "No reflections found to analyze")
            return None

        # Group reflections by category and sum importance scores
        category_scores = {}
        for reflection in reflections:
            # Handle nested metadata structure
            metadata = reflection.get("metadata", {})
            category = metadata.get("category")
            # Convert importance to float and default to 0 if not found or invalid
            try:
                importance = float(metadata.get("importance", 0))
            except (TypeError, ValueError):
                importance = 0
                log_event(agent.logger, "learning.priority", 
                         f"Invalid importance value for category {category}, defaulting to 0",
                         level="DEBUG")
            
            if category not in IGNORED_CATEGORIES:
                category_scores[category] = category_scores.get(category, 0) + importance

        if not category_scores:
            log_event(agent.logger, "learning.priority", "No valid categories found in reflections")
            return None

        # Find category with highest score
        highest_category = max(category_scores.items(), key=lambda x: x[1])
        
        log_event(agent.logger, "learning.priority", 
                 f"Highest priority category: {highest_category[0]} (score: {highest_category[1]})",
                 level="DEBUG")
        
        return highest_category[0]

    except Exception as e:
        log_error(agent.logger, f"Failed to determine highest priority category: {str(e)}")
        return None

async def learning_subroutine(agent: Agent, category: str = None) -> None:
    """Run the learning subroutine.
    
    Args:
        agent: The agent instance
        category: Optional category to learn about. If None, highest priority category is used.
    """
    if category is None:
        category = await _get_highest_priority_category(agent)
        if category is None:
            await agent.set_status(AgentStatus.AVAILABLE, "No learning opportunities found")
            return

    output_schema = "No schema provided"
    prompt_type = None
    match category:
        case "tools":
            pass
        case "agentic structure":
            pass            
        case "give feedback":
            existing_prompt = agent._give_feedback_prompt
            output_schema = agent._give_feedback_schema
            prompt_type = "give_feedback"
        case "memory reflection":
            existing_prompt = agent._reflect_memories_prompt
            output_schema = agent._reflect_memories_schema
            prompt_type = "reflect_memories"
        case "long term memory transfer":
            existing_prompt = agent._xfer_long_term_prompt
            output_schema = agent._xfer_long_term_schema
            prompt_type = "xfer_long_term"
        case "thought loop":
            existing_prompt = agent._thought_loop_prompt
            prompt_type = "thought_loop"
        case "reasoning":
            existing_prompt = agent._reasoning_prompt
            prompt_type = "reasoning"
        case "self improvement":
            existing_prompt = agent._self_improvement_prompt
            output_schema = agent._self_improvement_schema
            prompt_type = "self_improvement"
        case "insight":
            pass
    
    full_prompt = (
        agent._self_improvement_prompt + 
        "\n\nFormat your response as a simple JSON object with these fields:\n\n"
        + agent._self_improvement_schema
    )
   
    log_event(agent.logger, "agent.learning", f"Full prompt:\n {full_prompt}\n", level="DEBUG")

    learning_opps = await agent.memory.retrieve(
            query="",
            collection_names=["reflections"],
            metadata_filter={"category": category}
        )
    
    log_event(agent.logger, "agent.learning", f"Retrieved {len(learning_opps)} learning opportunities of type: {category}", level="DEBUG")
    
    consolidated_content = (f"Review your prompt on {category.title()} and your recent reflections on the subject and provide an improved prompt.\n\n"
                            f"## Existing Prompt:\n{existing_prompt}\n\n"
                            f"## Recent Reflections on {category.title()}:")  
    num = 1
    for opp in learning_opps:
        try:
            log_event(agent.logger, "agent.learning", f"Processing learning opp {num}: {opp}", level="DEBUG")
            importance = opp.get('metadata', {}).get('importance', 'N/A')  # Updated to check metadata
            content = opp.get('content', 'No content available')
            
            consolidated_content += f"\n\n#{num}. {content}\nImportance: {importance}"
            num += 1
        except Exception as e:
            log_error(agent.logger, f"Error processing learning opportunity {num}: {str(e)}\nData: {opp}")
            continue

    log_event(agent.logger, "agent.learning", 
             f"Consolidated content (first 500 chars): {consolidated_content[:500]}...", 
             level="DEBUG")

    improved_prompt_response = await agent.client.chat.completions.create(
            model=agent.config.model,
            messages=[
                {"role": "system", "content": full_prompt},
                {"role": "user", "content": consolidated_content}
            ],
            response_format={ "type": "json_object" }
        )

    try:
        improved_json = json.loads(improved_prompt_response.choices[0].message.content)
        improved_prompt = improved_json["prompt"]
        thoughts = improved_json["thoughts"]
        log_event(agent.logger, "agent.learning", 
                 f"Raw JSON: {improved_json}",
                 level="DEBUG")
    except Exception as e:
        log_event(agent.logger, "agent.learning", 
                 f"Raw Output: {improved_prompt_response.choices[0].message.content}",
                 level="DEBUG")
        log_error(agent.logger, f"Error parsing improved prompt response: {str(e)}")
        await agent.set_status(AgentStatus.AVAILABLE, "learning subroutine failed")
        return
    
    log_event(agent.logger, "agent.learning", 
            f"Thoughts: {thoughts} - Updated Prompt: {improved_prompt}",
            level="DEBUG")



    updated_score = await agent._evaluate_prompt(f"{prompt_type}_prompt", improved_prompt) # Always run before updating prompt
    await agent._calculate_updated_confidence_score(prompt_type, updated_score)
    await agent.update_prompt_module(f"{prompt_type}_prompt", improved_prompt)

    await agent.set_status(AgentStatus.AVAILABLE, "completed learning subroutine")