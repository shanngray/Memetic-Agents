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
                **({"temperature": agent.config.temperature} if agent.config.model not in ["o1-mini", "o3-mini"] else {}),
                messages=[
                    {"role": "system", "content": combined_evaluator_prompt},
                    {"role": "user", "content": combined_evaluator_message}
                ],
                timeout=3000,
                response_format={ "type": "json_object" },
                **({"reasoning_effort": agent.config.reasoning_effort} if agent.config.model == "o3-mini" else {})
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

    max_score = 10
    weighting = 2 # Higher is more difficult to change
    existing_score = agent._prompt_confidence_scores[prompt_type]

    # Input validation
    if not (0 <= existing_score <= max_score and 
            0 <= new_score <= max_score and 
            0 <= weighting <= 1):
        raise ValueError("Invalid input parameters")
    
    relative_score = (new_score - existing_score) / max_score

    # Calculate weight factor based on how close to max score (closer = more weight)
    weight_factor = (max_score - existing_score) / weighting
    
    if relative_score > 0:
        updated_score = existing_score + (relative_score * weight_factor)
    else:
        updated_score = existing_score + (relative_score * (1 - weight_factor))

    log_event(agent.logger, "confidence.update", 
                f"Updated {prompt_type} confidence score: {updated_score}")

    # Update the confidence score
    agent._prompt_confidence_scores[prompt_type] = updated_score