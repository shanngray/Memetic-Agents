import asyncio
import json
import math

from src.base_agent.type import Agent
from src.log_config import log_event, log_error, log_agent_message
from src.memetic_agent.modules.normalise_score import normalise_score

async def _evaluate_prompt_impl(agent: Agent, prompt_type: str, updated_prompt: str) -> int:
    """Evaluate a prompt and update confidence scores.
    
    Args:
        agent: The agent instance
        prompt_type: The type of prompt being scored
        updated_prompt: The updated version of the prompt
    
    Returns:
        int: The score for the prompt (0-10)
    """
    
    prompt_mapping = agent.get_prompt_mapping()

    existing_prompt = prompt_mapping[prompt_type]

    combined_evaluator_prompt = (
        f"{agent._evaluator_prompt}\n\nType of prompt being evaluated: {prompt_type}"
        f"\n\nOld version of prompt: {existing_prompt}"
        "\n\nFormat the output as a JSON object with the following schema:\n"
        "{\n"
        "\"thoughts\": \"Any thought you have about the prompt.\",\n"
        "\"score\": \"Numeric score (0-10) on prompt's effectiveness. "
        "Zero means the prompt is totally ineffective. 10 means the prompt is perfect and cannot be improved further.\"\n"
        "}"
    )

    combined_evaluator_message = (
        f"Your current job is to evaluate an updated version of your prompt used for \"{prompt_type}\" and provide a confidence score from 0 to 10.\n\n"
        f"The version of the prompt your are evaluating is: {updated_prompt}"
    )
    
    try:

        response = await asyncio.wait_for(
            agent.client.chat.completions.create(
                model=agent.config.model,
                temperature=agent.config.temperature,
                messages=[
                    {"role": "system", "content": combined_evaluator_prompt},
                    {"role": "user", "content": combined_evaluator_message}
                ],
                timeout=3000,
                response_format={ "type": "json_object" }
            ),
            timeout=3050
        )

        response_json = json.loads(response.choices[0].message.content)

        thoughts = response_json["thoughts"]
        raw_score = response_json["score"]
        
        if not isinstance(raw_score, int) or raw_score < 0 or raw_score > 10:
            score = normalise_score(agent, raw_score)
            log_event(agent.logger, "confidence.evaluation", 
                    f"Invalid raw_score: {raw_score} of type {type(raw_score).__name__}. Normalised to {score}")
        else:
            score = raw_score

        log_event(agent.logger, "confidence.evaluation", 
                    f"Updated {prompt_type} has been evaluated. Thoughts: {thoughts} - Score: {score}")

    except Exception as e:
        log_error(agent.logger, "Error during prompt evaluation", exc_info=e)
        raise

    return score

async def _calculate_updated_confidence_score_impl(agent: Agent, prompt_type: str, new_score: int) -> None:
    """
    Calculate a new confidence score using an exponential weighting method.

    Args:
        agent: The agent instance
        prompt_type: The type of prompt being scored
        new_score: The new score for the prompt (0-10)
        
    """
    difficulty: float = 1.0 # 1.0 is average difficulty. Higher values make it harder to achive a high score.
    weight_factor: float = 0.2 # 0.5 is average weighting. Lower values mean more weight is given to the existing score (making it hard to change).
    max_score: float = 10

    existing_score = agent._prompt_confidence_scores[prompt_type]

    # Input validation
    if not (0 <= existing_score <= max_score and 
            0 <= new_score <= max_score and 
            0 < difficulty and 
            0 <= weight_factor <= 1):
        raise ValueError("Invalid input parameters")
    
    # Normalize scores to 0-1 range
    normalized_new = new_score / max_score
    normalized_existing = existing_score / max_score
    
    # Calculate exponential weight
    exp_weight = (1 - math.exp(-normalized_new * difficulty)) / (1 - math.exp(-difficulty))
    
    # Calculate weighted average
    weighted_score = (1 - weight_factor) * normalized_existing + weight_factor * exp_weight
    
    # Convert back to original scale
    updated_score = weighted_score * max_score

    log_event(agent.logger, "confidence.update", 
                f"Updated {prompt_type} confidence score: {updated_score}")

    # Update the confidence score
    agent._prompt_confidence_scores[prompt_type] = updated_score