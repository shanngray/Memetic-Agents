# NOT IN USE
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
                        {"role": "system", "content": agent._xfer_feedback_prompt},
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
                    
                    log_event(agent.logger, "agent.memorising",
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