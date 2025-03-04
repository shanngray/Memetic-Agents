from src.base_agent.type import Agent
from src.log_config import log_event, log_error
from filelock import FileLock
from typing import Dict

async def save_memory_to_disk_impl(agent: Agent, structured_info: Dict, metadata: Dict, memory_type: str) -> None:
    """Save structured memory information to disk for debugging/backup.
    
    Args:
        structured_info: Dictionary containing the structured information to save
        metadata: Dictionary containing metadata about the memory
        memory_type: Type of memory ('memory' or 'feedback')
    """
    try:
        # Use different files for different memory types
        file_suffix = "feedback" if memory_type == "feedback" else "long_term"
        memory_dump_path = agent.files_path / f"{agent.config.agent_name}_{file_suffix}_memory_dump.md"
        
        with FileLock(f"{memory_dump_path}.lock"):
            with open(memory_dump_path, "a", encoding="utf-8") as f:
                # Write header with timestamp
                f.write(f"\n\n## {memory_type.title()} Entry {metadata['timestamp']}\n")
                
                # Write metadata section
                f.write("### Metadata\n")
                for key, value in metadata.items():
                    if isinstance(value, list):
                        value = ", ".join(value)
                    f.write(f"{key}: {value}\n")
                
                # Write content section based on memory type
                if memory_type == "feedback":
                    f.write("\n### Feedback Insights\n")
                    if isinstance(structured_info, dict) and "insights" in structured_info:
                        for insight in structured_info["insights"]:
                            f.write(f"\n#### Insight\n")
                            f.write(f"Content: {insight['content']}\n")
                            f.write(f"Category: {insight['category']}\n")
                            f.write(f"Importance: {insight['importance']}\n")
                            f.write("Action Items:\n")
                            for item in insight['action_items']:
                                f.write(f"- {item}\n")
                            f.write(f"Tags: {', '.join(insight['tags'])}\n")
                    else:
                        # Handle case where structured_info is a string
                        f.write(f"\n{structured_info}\n")
                else:
                    # Handle regular memory entries
                    f.write("\n### Extracted Information\n")
                    if isinstance(structured_info, dict):
                        for category in ["facts", "decisions", "preferences", "patterns"]:
                            if structured_info.get(category):
                                f.write(f"\n#### {category.title()}\n")
                                for item in structured_info[category]:
                                    f.write(f"- {item}\n")
                    else:
                        # Handle case where structured_info is a string
                        f.write(f"\n{structured_info}\n")
                    
                    f.write("\n---\n")  # Add separator between entries

                log_event(agent.logger, f"{memory_type}.dumped", 
                            f"Saved {memory_type} {metadata.get('memory_id') or metadata.get('insight_id')} to disk")
                    
    except Exception as e:
        log_error(agent.logger, 
                    f"Failed to save {memory_type} {metadata.get('memory_id') or metadata.get('insight_id')} to disk: {str(e)}")