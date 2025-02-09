from src.base_agent.type import Agent
from src.log_config import log_event, log_error 
import json
from filelock import FileLock
import datetime
import asyncio


async def _record_score_impl(agent: Agent, prompt_type: str, prompt_score: int, conversation_id: str, score_type: str) -> None:
    """Record a score for a prompt and update confidence scores when appropriate.
    
    Args:
        agent: Agent instance
        prompt_type: Type of prompt being scored (e.g. "reasoning", "memory_reflection", etc.)
        prompt_score: Score given to the prompt (0-10)
        conversation_id: ID of the conversation this score is from
        score_type: Type of score ("friends_initial_eval", "friends_updated_eval", "self_eval")
        
    Raises:
        ValueError: If prompt_score is not an integer between 0 and 10
    """
    try:
        # Validate score - This is a redundant sanity check (first check in normalise_score)
        if not isinstance(prompt_score, int) or prompt_score < 0 or prompt_score > 10:
            raise ValueError("prompt_score must be an integer between 0 and 10.\n"
                             "Received: " + str(prompt_score) + "of type " + str(type(prompt_score).__name__))
        
        # Use a single scores file for all conversations
        scores_file = agent.scores_path / "conversation_scores.json"
        
        async def read_scores():
            if scores_file.exists():
                def _read():
                    with FileLock(f"{scores_file}.lock"):
                        return json.loads(scores_file.read_text())
                return await asyncio.to_thread(_read)
            return {}
        
        scores_data = await read_scores()
        
        # Initialize conversation if not exists
        if conversation_id not in scores_data:
            scores_data[conversation_id] = {
                "timestamp": str(datetime.datetime.now(datetime.UTC)),
                "prompts": {}
            }
        
        # Initialize prompt type if not exists
        if prompt_type not in scores_data[conversation_id]["prompts"]:
            scores_data[conversation_id]["prompts"][prompt_type] = {}
        
        # Record the score
        scores_data[conversation_id]["prompts"][prompt_type][score_type] = prompt_score
        
        async def write_scores():
            def _write():
                with FileLock(f"{scores_file}.lock"):
                    scores_file.write_text(json.dumps(scores_data, indent=2))
            await asyncio.to_thread(_write)
        
        await write_scores()
        
        # Check if we have all required scores to calculate new confidence
        required_scores = {
            "friends_initial_eval",
            "friends_updated_eval", 
            "self_eval"
        }
        
        if required_scores.issubset(scores_data[conversation_id]["prompts"][prompt_type].keys()):
            await agent._update_confidence_score(
                prompt_type,
                scores_data[conversation_id]["prompts"][prompt_type]["friends_initial_eval"],
                scores_data[conversation_id]["prompts"][prompt_type]["friends_updated_eval"],
                scores_data[conversation_id]["prompts"][prompt_type]["self_eval"]
            )
        
        log_event(agent.logger, "confidence.recorded",
                    f"Recorded {score_type} score of {prompt_score} for {prompt_type}")
        
    except Exception as e:
        log_error(agent.logger, f"Failed to record score: {str(e)}")
        raise

async def _update_confidence_score_impl(agent: Agent, prompt_type: str, initial_friend_score: int, 
                                    updated_friend_score: int, self_eval_score: int) -> None:
    """Calculate and update confidence score for a prompt type based on interaction scores.
    
    Args:
        agent: Agent instance
        prompt_type: Type of prompt being scored
        initial_friend_score: Friend's initial evaluation score (0-10)
        updated_friend_score: Friend's evaluation of updated prompt (0-10)
        self_eval_score: Agent's self-evaluation score (0-10)
        
    Raises:
        ValueError: If any score is not an integer between 0 and 10
    """
    try:
        # Validate scores
        for score in (initial_friend_score, updated_friend_score, self_eval_score):
            if not isinstance(score, int) or score < 0 or score > 10:
                raise ValueError("All input scores must be integers between 0 and 10")
        
        # Get current confidence score (defaults to 5.0)
        current_confidence = agent._prompt_confidence_scores.get(prompt_type, 5.0)
        
        # Calculate improvement factor (-1 to 1)
        score_improvement = (updated_friend_score - initial_friend_score) / 10
        
        # Calculate alignment factor (-1 to 1) based on how close self eval was to friend's eval
        alignment_factor = 1 - (abs(self_eval_score - initial_friend_score) / 10)
        
        # Weighted adjustment calculation
        # More weight on improvement (0.6) than alignment (0.4)
        adjustment = ((score_improvement * 0.6) + (alignment_factor * 0.4)) * 2
        
        # Calculate new confidence score (bounded between 0 and 10)
        new_confidence = max(0.0, min(10.0, current_confidence + adjustment))
        
        # Update confidence scores
        agent._prompt_confidence_scores[prompt_type] = new_confidence
        
        async def save_confidence_scores():
            def _save():
                with FileLock(f"{agent._prompt_confidence_scores_path}.lock"):
                    agent._prompt_confidence_scores_path.write_text(
                        json.dumps(agent._prompt_confidence_scores, indent=2)
                    )
            await asyncio.to_thread(_save)
        
        await save_confidence_scores()
        
        log_event(agent.logger, "confidence.updated",
                    f"Updated confidence score for {prompt_type} from {current_confidence:.2f} to {new_confidence:.2f}")
        
    except Exception as e:
        log_error(agent.logger, f"Failed to update confidence score: {str(e)}")
        raise
