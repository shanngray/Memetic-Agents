from typing import Any
from src.log_config import log_event, log_error
import httpx
from datetime import datetime
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
        score, feedback = await agent._evaluate_response(response_content)
        
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
        response = await agent.client.chat.completions.create(
            model=agent.config.model,
            messages=[
                {"role": "system", "content": agent._give_feedback_prompt},
                {"role": "user", "content": f"Response to evaluate:\n{response_content}"}
            ],
            response_format={ "type": "json_object" }
        )
        
        print("\n|------------EVALUATE RESPONSE------------|\n")
        ic(f"Response: {response}")
        print("\n|--------------------------------|\n")
        
        result = json.loads(response.choices[0].message.content)
        return result["score"], result["feedback"]
        
    except Exception as e:
        log_error(agent.logger, f"Failed to evaluate response: {str(e)}")
        return 5, "Error evaluating response"  # Default neutral score
