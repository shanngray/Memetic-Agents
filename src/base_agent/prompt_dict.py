from pathlib import Path
from typing import Dict, Any

def create_prompt_dict(prompt_path: Path, 
                      system_prompt: str,
                      reasoning_prompt: str,
                      give_feedback_prompt: str,
                      reflect_feedback_prompt: str,
                      reflect_memories_prompt: str,
                      self_improvement_prompt: str,
                      thought_loop_prompt: str,
                      xfer_long_term_prompt: str,
                      evaluator_prompt: str,
                      give_feedback_schema: str,
                      reflect_memories_schema: str,
                      self_improvement_schema: str,
                      xfer_long_term_schema: str) -> Dict[str, Dict[str, Any]]:
    """Create a dictionary containing all prompt-related information.
    
    Args:
        prompt_path: Base path for prompt files
        *_prompt: Content for each prompt type
        *_schema: Schema content for applicable prompts
    """
    return {
        "system_prompt": {
            "name": "system_prompt",
            "content": system_prompt,
            "confidence": 0.0,
            "schema": None,
            "path": None
        },
        "reasoning": {
            "name": "reasoning_prompt",
            "content": reasoning_prompt,
            "confidence": 0.0,
            "schema": None,
            "path": prompt_path / "reasoning_prompt.md"
        },
        "give_feedback": {
            "name": "give_feedback_prompt",
            "content": give_feedback_prompt,
            "confidence": 0.0,
            "schema": give_feedback_schema,
            "path": prompt_path / "give_feedback_prompt.md"
        },
        "reflect_feedback": {
            "name": "reflect_feedback_prompt",
            "content": reflect_feedback_prompt,
            "confidence": 0.0,
            "schema": None,
            "path": None
        },
        "reflect_memories": {
            "name": "reflect_memories_prompt",
            "content": reflect_memories_prompt,
            "confidence": 0.0,
            "schema": reflect_memories_schema,
            "path": prompt_path / "reflect_memories_prompt.md"
        },
        "self_improvement": {
            "name": "self_improvement_prompt", 
            "content": self_improvement_prompt,
            "confidence": 0.0,
            "schema": self_improvement_schema,
            "path": prompt_path / "self_improvement_prompt.md"
        },
        "thought_loop": {
            "name": "thought_loop_prompt",
            "content": thought_loop_prompt,
            "confidence": 0.0,
            "schema": None,
            "path": prompt_path / "thought_loop_prompt.md"
        },
        "xfer_long_term": {
            "name": "xfer_long_term_prompt",
            "content": xfer_long_term_prompt,
            "confidence": 0.0,
            "schema": xfer_long_term_schema,
            "path": prompt_path / "xfer_long_term_prompt.md"
        },
        "evaluator": {
            "name": "evaluator_prompt",
            "content": evaluator_prompt,
            "confidence": 0.0,
            "schema": None,
            "path": prompt_path / "evaluator_prompt.md"
        }
    }
