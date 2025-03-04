import datetime
import json
import uuid
from typing import Optional, Any, Dict
from src.log_config import log_event, log_error
from src.base_agent.type import Agent

async def receive_feedback_impl(
    agent: Agent,
    sender: str,
    conversation_id: str,
    score: int,
    feedback: str
) -> None:
    """Process and store received feedback from another agent.
    
    Args:
        agent: The agent receiving the feedback
        sender: Name of the agent providing feedback
        conversation_id: ID of the conversation being rated
        score: Numerical score (typically 0-10)
        feedback: Detailed feedback text
    """
    try:
        # Format feedback content
        formatted_feedback = (
            f"Feedback from {sender} regarding conversation {conversation_id}:\n"
            f"Score: {score}/10\n"
            f"Comments: {feedback}"
        )
        
        # Store in feedback collection with metadata
        metadata = {
            "sender": sender,
            "conversation_id": conversation_id,
            "score": score,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await agent.memory.store(
            content=formatted_feedback,
            collection_name="feedback",
            metadata=metadata
        )
        
        log_event(agent.logger, "feedback.received",
                    f"Received feedback from {sender} for conversation {conversation_id}")
                    
    except Exception as e:
        log_error(agent.logger, f"Failed to process received feedback: {str(e)}")

async def process_feedback_impl(agent: Agent, days_threshold: int = 0) -> None:
    """Process and transfer feedback to long-term memory.
    
    Args:
        agent: The agent processing the feedback
        days_threshold: Number of days worth of feedback to keep in feedback collection. 
                       Feedback older than this will be processed into long-term storage.
                       Default is 0 (process all feedback).
    """
    try:
        log_event(agent.logger, "agent.processing_feedback", 
                 f"Beginning feedback processing (threshold: {days_threshold} days)")
        
        # Retrieve feedback from feedback collection
        feedback_items = await agent.memory.retrieve(
            query="",  # Empty query to get all feedback
            collection_names=["feedback"],
            n_results=100
        )
        
        # Filter feedback based on threshold
        threshold_date = datetime.now() - timedelta(days=days_threshold)
        feedback_items = [
            feedback for feedback in feedback_items
            if datetime.fromisoformat(feedback["metadata"].get("timestamp", "")) < threshold_date
        ]
        
        if not feedback_items:
            log_event(agent.logger, "feedback.empty", 
                     f"No feedback older than {days_threshold} days found for processing")
            return

        # Process each feedback item
        for feedback in feedback_items:
            try:
                # Extract insights using LLM
                response = await agent.client.chat.completions.create(
                    model=agent.config.submodel,
                    messages=[
                        {"role": "system", "content": agent.prompt.xfer_feedback.content},
                        {"role": "user", "content": f"Feedback to analyze:\n{feedback['content']}"}
                    ],
                    response_format={ "type": "json_object" }
                )
                
                insights = json.loads(response.choices[0].message.content)
                
                # Store each extracted insight
                for insight in insights["insights"]:
                    # Format content with metadata
                    formatted_content = (
                        f"{insight['content']}\n\n"
                        f"Category: {insight['category']}\n"
                        f"Action Items:\n" + "\n".join(f"- {item}" for item in insight['action_items']) + "\n\n"
                        f"Tags: {', '.join(insight['tags'])}"
                    )
                    
                    metadata = {
                        "insight_id": str(uuid.uuid4()),
                        "original_feedback_id": feedback["metadata"].get("feedback_id"),
                        "category": insight["category"],
                        "importance": insight["importance"],
                        "source": "feedback_processing",
                        "timestamp": datetime.now().isoformat()
                    }

                    await agent.memory.store(
                        content=formatted_content,
                        collection_name="long_term",
                        metadata=metadata
                    )
                    
                    log_event(agent.logger, "feedback.memorised",
                                f"Processed feedback {metadata['insight_id']} into long-term storage")

                    # Save to disk for debugging/backup
                    await agent._save_memory_to_disk(formatted_content, metadata, "feedback")

            except Exception as e:
                log_error(agent.logger, f"Failed to process feedback: {str(e)}")
                continue


        await agent._cleanup_memories(days_threshold, "feedback")
        
        log_event(agent.logger, "memory.memorising.complete",
                    f"Completed memory consolidation for {len(feedback_items)} pieces of feedback")
                
    except Exception as e:
        log_error(agent.logger, "Failed to process feedback", exc_info=e)
    finally:
        if agent.status != AgentStatus.SHUTTING_DOWN:
            await agent.set_status(agent._previous_status, "transfer to long term - complete")

async def reflect_on_feedback_impl(
    agent: Agent,
    days_threshold: Optional[int] = None,
    min_score: Optional[int] = None,
    max_score: Optional[int] = None
) -> Dict[str, Any]:
    """Reflect on collected feedback to generate insights and improvements.
    
    Args:
        agent: The agent reflecting on feedback
        days_threshold: Optional number of days to look back
        min_score: Optional minimum score to consider
        max_score: Optional maximum score to consider
        
    Returns:
        Dictionary containing reflection results and metadata
    """
    try:
        # Build conversation for reflection
        messages = [
            Message(
                role="user" if agent.config.model == "o1-mini" else "developer" if agent.config.model == "o3-mini" else "system",
                content=agent.prompt.reflect_feedback.content
            )
        ]
        
        # Retrieve feedback
        feedback_items = await agent.memory.retrieve(
            query="",
            collection_names=["feedback"],
            n_results=100
        )
        
        # Apply filters
        if days_threshold:
            cutoff_date = datetime.now() - timedelta(days=days_threshold)
            feedback_items = [
                f for f in feedback_items
                if datetime.fromisoformat(f["metadata"].get("timestamp", "")) >= cutoff_date
            ]
            
        if min_score is not None:
            feedback_items = [
                f for f in feedback_items
                if f["metadata"].get("score", 0) >= min_score
            ]
            
        if max_score is not None:
            feedback_items = [
                f for f in feedback_items
                if f["metadata"].get("score", 10) <= max_score
            ]

        # Add feedback summary message
        feedback_summary = (
            f"Analyzing {len(feedback_items)} pieces of feedback\n\n" +
            "\n---\n".join(
                f"From: {item['metadata'].get('sender', 'Unknown')}\n"
                f"Score: {item['metadata'].get('score', 'N/A')}/10\n"
                f"Feedback: {item['content']}"
                for item in feedback_items
            )
        )
        
        messages.append(Message(
            role="user",
            content=feedback_summary
        ))
        
        # Get reflection from LLM
        response = await agent.client.chat.completions.create(
            model=agent.config.model,
            messages=[m.dict() for m in messages],
            temperature=0.7,
            response_format={ "type": "json_object" },
            **({"reasoning_effort": agent.config.reasoning_effort} if agent.config.model == "o3-mini" else {})
        )
        
        reflection = json.loads(response.choices[0].message.content)
        
        # Store reflection in long-term memory
        metadata = {
            "reflection_id": str(uuid.uuid4()),
            "type": "feedback_reflection",
            "feedback_count": len(feedback_items),
            "date_range": f"{days_threshold}_days" if days_threshold else "all_time",
            "timestamp": datetime.now().isoformat()
        }
        
        await agent.memory.store(
            content=json.dumps(reflection, indent=2),
            collection_name="long_term",
            metadata=metadata
        )
        
        log_event(agent.logger, "feedback.reflection",
                 f"Completed feedback reflection {metadata['reflection_id']}")
                 
        return {
            "reflection": reflection,
            "metadata": metadata,
            "feedback_analyzed": len(feedback_items)
        }
        
    except Exception as e:
        log_error(agent.logger, "Failed to reflect on feedback", exc_info=e)
        raise
