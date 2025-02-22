from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field

class PromptEntry(BaseModel):
    """Represents a single prompt entry with its metadata."""
    name: str
    content: Optional[str] = None
    confidence: float = Field(default=0.0, ge=0.0, le=10.0)
    schema: Optional[str] = None  # Schema content
    schema_path: Optional[Path] = None  # Path to schema file
    path: Optional[Path] = None
    category: Optional[str] = None
    implemented: bool = False

class PromptLibrary(BaseModel):
    """Container for all prompt-related information used by the agent."""
    system: PromptEntry = Field(default_factory=lambda: PromptEntry(
        name="system_prompt",
        category="system prompt",
        confidence=0.0,
        implemented=True
    ))
    reasoning: PromptEntry = Field(default_factory=lambda: PromptEntry(
        name="reasoning_prompt",
        category="reasoning",
        confidence=0.0,
        implemented=True
    ))
    give_feedback: PromptEntry = Field(default_factory=lambda: PromptEntry(
        name="give_feedback_prompt",
        category="give feedback",
        confidence=0.0,
        implemented=True
    ))
    reflect_feedback: PromptEntry = Field(default_factory=lambda: PromptEntry(
        name="reflect_feedback_prompt",
        confidence=0.0
    ))
    reflect_memories: PromptEntry = Field(default_factory=lambda: PromptEntry(
        name="reflect_memories_prompt",
        confidence=0.0
    ))
    self_improvement: PromptEntry = Field(default_factory=lambda: PromptEntry(
        name="self_improvement_prompt",
        confidence=0.0
    ))
    thought_loop: PromptEntry = Field(default_factory=lambda: PromptEntry(
        name="thought_loop_prompt",
        confidence=0.0
    ))
    xfer_long_term: PromptEntry = Field(default_factory=lambda: PromptEntry(
        name="xfer_long_term_prompt",
        confidence=0.0
    ))
    evaluator: PromptEntry = Field(default_factory=lambda: PromptEntry(
        name="evaluator_prompt",
        confidence=0.0
    ))
    xfer_feedback: PromptEntry = Field(default_factory=lambda: PromptEntry(
        name="xfer_feedback_prompt",
        confidence=0.0
    ))
    tools: PromptEntry = Field(default_factory=lambda: PromptEntry(
        name="tools_prompt",
        category="tools"
    ))
    agentic_structure: PromptEntry = Field(default_factory=lambda: PromptEntry(
        name="agentic_structure_prompt",
        category="agentic structure"
    ))
    insight: PromptEntry = Field(default_factory=lambda: PromptEntry(
        name="insight_prompt",
        category="insight"
    ))

    @classmethod
    def create(cls, prompt_path: Path,
               system_prompt: str,
               reasoning_prompt: str,
               give_feedback_prompt: str,
               reflect_feedback_prompt: str,
               reflect_memories_prompt: str,
               self_improvement_prompt: str,
               thought_loop_prompt: str,
               thought_loop_schema: str,
               xfer_long_term_prompt: str,
               evaluator_prompt: str,
               give_feedback_schema: str,
               reflect_memories_schema: str,
               self_improvement_schema: str,
               xfer_long_term_schema: str) -> "PromptLibrary":
        """Create a PromptLibrary instance with the provided prompts and schemas."""
        return cls(
            system=PromptEntry(
                name="system_prompt",
                content=system_prompt,
                category="system prompt",
                confidence=0.0,
                implemented=True
            ),
            reasoning=PromptEntry(
                name="reasoning_prompt",
                content=reasoning_prompt,
                path=prompt_path / "reasoning_prompt.md",
                category="reasoning",
                confidence=0.0,
                implemented=True
            ),
            give_feedback=PromptEntry(
                name="give_feedback_prompt",
                content=give_feedback_prompt,
                schema=give_feedback_schema,
                schema_path=prompt_path / "schemas" / "give_feedback_schema.json",
                path=prompt_path / "give_feedback_prompt.md",
                category="give feedback",
                confidence=0.0,
                implemented=True
            ),
            reflect_feedback=PromptEntry(
                name="reflect_feedback_prompt",
                content=reflect_feedback_prompt,
                confidence=0.0
            ),
            reflect_memories=PromptEntry(
                name="reflect_memories_prompt",
                content=reflect_memories_prompt,
                schema=reflect_memories_schema,
                schema_path=prompt_path / "schemas" / "reflect_memories_schema.json",
                path=prompt_path / "reflect_memories_prompt.md",
                confidence=0.0
            ),
            self_improvement=PromptEntry(
                name="self_improvement_prompt",
                content=self_improvement_prompt,
                schema=self_improvement_schema,
                schema_path=prompt_path / "schemas" / "self_improvement_schema.json",
                path=prompt_path / "self_improvement_prompt.md",
                confidence=0.0
            ),
            thought_loop=PromptEntry(
                name="thought_loop_prompt",
                content=thought_loop_prompt,
                schema=thought_loop_schema,
                schema_path=prompt_path / "schemas" / "thought_loop_schema.json",
                path=prompt_path / "thought_loop_prompt.md",
                confidence=0.0
            ),
            xfer_long_term=PromptEntry(
                name="xfer_long_term_prompt",
                content=xfer_long_term_prompt,
                schema=xfer_long_term_schema,
                schema_path=prompt_path / "schemas" / "xfer_long_term_schema.json",
                path=prompt_path / "xfer_long_term_prompt.md",
                confidence=0.0
            ),
            evaluator=PromptEntry(
                name="evaluator_prompt",
                content=evaluator_prompt,
                path=prompt_path / "evaluator_prompt.md",
                confidence=0.0
            ),
            xfer_feedback=PromptEntry(
                name="xfer_feedback_prompt",
                confidence=0.0
            ),
            tools=PromptEntry(
                name="tools_prompt",
                category="tools"
            ),
            agentic_structure=PromptEntry(
                name="agentic_structure_prompt",
                category="agentic structure"
            ),
            insight=PromptEntry(
                name="insight_prompt",
                category="insight"
            )
        )
