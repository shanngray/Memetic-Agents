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
        # Get unimplemented categories from prompt_dict
        unimplemented_categories = {
            prompt_data["category"] 
            for prompt_data in agent.prompt_dict.values() 
            if not prompt_data["implemented"]
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
            
            if category not in unimplemented_categories:
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
    try:
        if category is None:
            category = await _get_highest_priority_category(agent)
            if category is None:
                await agent.set_status(AgentStatus.AVAILABLE, "No learning opportunities found")
                return

        # Find matching prompt data
        prompt_data = next(
            (data for data in agent.prompt_dict.values() if data["category"] == category),
            None
        )
        
        if not prompt_data or not prompt_data["implemented"]:
            await agent.set_status(AgentStatus.AVAILABLE, f"Category not implemented: {category}")
            return

        existing_prompt = prompt_data["content"]
        output_schema = prompt_data["schema"] or "No schema provided"
        prompt_type = prompt_data["name"].replace("_prompt", "")

        try:
            full_prompt = (
                agent.prompt.self_improvement.content + 
                "\n\nFormat your response as a simple JSON object with these fields:\n\n"
                + agent.prompt.self_improvement.schema_content
            )
            
            learning_opps = await agent.memory.retrieve(
                query="",
                collection_names=["reflections"],
                metadata_filter={"category": category}
            )
            
            if not learning_opps:
                log_event(agent.logger, "agent.learning", f"No learning opportunities found for category: {category}")
                await agent.set_status(AgentStatus.AVAILABLE, f"No learning opportunities for: {category}")
                return
                
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

        except Exception as e:
            log_error(agent.logger, f"Error preparing learning data: {str(e)}")
            await agent.set_status(AgentStatus.AVAILABLE, "Failed to prepare learning data")
            return

        try:
            improved_prompt_response = await agent.client.chat.completions.create(
                model=agent.config.model,
                messages=[
                    {"role": "system", "content": full_prompt},
                    {"role": "user", "content": consolidated_content}
                ],
                response_format={ "type": "json_object" },
                **({"reasoning_effort": agent.config.reasoning_effort} if agent.config.model == "o3-mini" else {})
            )
            
            improved_json = json.loads(improved_prompt_response.choices[0].message.content)
            improved_prompt = improved_json["prompt"]
            thoughts = improved_json["thoughts"]
            
            if not improved_prompt or not thoughts:
                raise ValueError("Missing required fields in model response")
                
        except Exception as e:
            log_error(agent.logger, f"Error getting improved prompt: {str(e)}")
            await agent.set_status(AgentStatus.AVAILABLE, "Failed to generate improved prompt")
            return

        try:
            updated_score = await agent._evaluate_prompt(f"{prompt_type}_prompt", improved_prompt)
            await agent._calculate_updated_confidence_score(prompt_type, updated_score)
            await agent.update_prompt_module(f"{prompt_type}_prompt", improved_prompt)

            # Cleanup used reflections
            deleted_count = await agent.memory.delete_by_metadata(
                collection_name="reflections",
                metadata_filter={"category": category}
            )
            log_event(agent.logger, "agent.learning", 
                     f"Cleaned up {deleted_count} processed reflections for category: {category}")
                     
        except Exception as e:
            log_error(agent.logger, f"Error updating prompt and cleaning up: {str(e)}")
            await agent.set_status(AgentStatus.AVAILABLE, "Failed to update prompt")
            return

        await agent.set_status(AgentStatus.AVAILABLE, "completed learning subroutine")
        
    except Exception as e:
        log_error(agent.logger, f"Unexpected error in learning subroutine: {str(e)}")
        await agent.set_status(AgentStatus.AVAILABLE, "Learning subroutine failed unexpectedly")