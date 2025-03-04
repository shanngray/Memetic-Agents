from src.log_config import log_event, log_error
from src.base_agent.type import Agent
from typing import List, Dict
import asyncio
from pathlib import Path
from filelock import FileLock
from datetime import datetime

async def save_reflection_to_disk_impl(agent: Agent, reflections: List[Dict], original_metadata: Dict) -> None:
    """Save learning reflections to disk for debugging/backup."""
    try:
        reflection_dump_path = agent.files_path / f"{agent.config.agent_name}_reflection_dump.md"
        lock = FileLock(f"{reflection_dump_path}.lock")
        
        async with asyncio.Lock():  # Use asyncio.Lock() for async context
            with lock:
                def write_reflections():
                    with open(reflection_dump_path, "a", encoding="utf-8") as f:
                        f.write(f"\n\n## Learning Reflections Generated {datetime.now().isoformat()}\n")
                        f.write("### Original Memory Metadata\n")
                        for key, value in original_metadata.items():
                            f.write(f"{key}: {value}\n")
                        
                        f.write("\n### Extracted Learnings\n")
                        for reflection in reflections:
                            category = reflection.get('category', 'Uncategorized')
                            importance = reflection.get('importance', 'N/A')
                            lesson = reflection.get('lesson', 'No lesson recorded')
                            thoughts = reflection.get('thoughts', 'No additional thoughts')
                            
                            f.write(f"\n#### {category.title()} (importance: {importance})\n")
                            f.write(f"Lesson: {lesson}\n")
                            f.write(f"Thoughts: {thoughts}\n")
                        f.write("\n---\n")
                
                # Properly await the thread execution
                await asyncio.to_thread(write_reflections)
        
        log_event(agent.logger, "reflection.dumped", 
                    f"Saved {len(reflections)} learning reflections to disk",
                    level="DEBUG")
    except Exception as e:
        log_error(agent.logger, 
                    f"Failed to save learning reflections to disk: {str(e)}")