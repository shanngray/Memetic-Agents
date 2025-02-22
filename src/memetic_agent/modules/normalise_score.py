from src.base_agent.type import Agent
from src.log_config import log_event, log_error, log_agent_message
from typing import Any

def normalise_score(agent: Agent, score: Any) -> int:
    """Convert various score inputs to an integer between 0-10.
    
    Handles:
    - Strings (including percentages)
    - Floats
    - Invalid types by defaulting to 5
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
                return 5  # Default if string contains no numbers
            score = float(score)
        
        # Convert to float first
        score_float = float(score)
        
        # Handle percentage-like values (0-100)
        if score_float > 10:
            score_float = score_float / 10
        
        # Round to nearest integer and clamp between 0-10
        normalised_score = max(0, min(10, round(score_float)))
        
        log_event(agent.logger, "confidence.evaluation", 
                 f"Normalised score: {normalised_score} (from original: {score} of type {type(score).__name__})", 
                 level="DEBUG")
        
        return normalised_score
        
    except (ValueError, TypeError) as e:
        log_error(agent.logger, 
                 f"Score normalization failed for value '{score}' of type {type(score).__name__}. Using default score of 5.")
        return 5