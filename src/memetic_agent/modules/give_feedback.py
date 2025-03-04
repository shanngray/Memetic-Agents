from typing import Any
from src.log_config import log_event, log_error
import httpx
from datetime import datetime
import json
from src.base_agent.type import Agent

async def evaluate_and_send_feedback_impl(
    agent: Agent,
    receiver: str,
    conversation_id: str,
    response_content: str
) -> None:
    """Evaluate response quality and send feedback to the agent."""
    try:
        # Get evaluation from LLM
        score, feedback = await evaluate_response_impl(agent, response_content)
        
        log_event(agent.logger, "feedback.evaluated",
                    f"Evaluated response by {receiver}. Score: {score}, Feedback: {feedback}", 
                    level="INFO")

        # Send feedback via API
        async with httpx.AsyncClient() as client:
            feedback_message = {
                "sender": agent.config.agent_name,
                "receiver": receiver,
                "conversation_id": conversation_id,
                "score": score,
                "feedback": feedback,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            await client.post(
                "http://localhost:8000/agent/feedback",
                json=feedback_message,
                timeout=300.0
            )
            
        log_event(agent.logger, "feedback.sent",
                    f"Sent feedback to {receiver} for conversation {conversation_id}")
                    
    except Exception as e:
        log_error(agent.logger, f"Failed to process/send feedback: {str(e)}")

async def evaluate_response_impl(agent: Agent, response_content: str) -> tuple[int, str]:
    """Evaluate response quality using LLM."""
    try:
        full_prompt = agent.prompt.give_feedback.content + "\n\nFormat your response as a JSON object with the following schema:\n" + agent.prompt.give_feedback.schema_content
        response = await agent.client.chat.completions.create(
            model=agent.config.model,
            messages=[
                {"role": "system", "content": full_prompt},
                {"role": "user", "content": f"Response to evaluate:\n{response_content}"}
            ],
            response_format={ "type": "json_object" },
            **({"reasoning_effort": agent.config.reasoning_effort} if agent.config.model == "o3-mini" else {})
        )
        
        result = json.loads(response.choices[0].message.content)
        
        log_event(agent.logger, "feedback.evaluated",
                    f"Evaluated response in JSON format: {result}", 
                    level="DEBUG")
        return result["score"], result["feedback"]
        
    except Exception as e:
        log_error(agent.logger, f"Failed to evaluate response: {str(e)}")
        return 5, "Error evaluating response"  # Default neutral score
