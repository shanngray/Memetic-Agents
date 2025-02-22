from datetime import datetime
import httpx
import asyncio
from typing import Dict, Any
from src.base_agent.models import AgentStatus
from src.base_agent.type import Agent
from src.log_config import log_agent_message, log_event, log_error
from src.api_server.models.api_models import APIMessage, SocialMessage, PromptModel, PromptEvaluation
from random import choice
import uuid

async def start_socialising_impl(agent: Agent) -> Dict[str, Any]:
    """Start socialising with another agent."""
    socialising_agent = agent.config.agent_name

    async with httpx.AsyncClient(timeout=3000.0) as client:
        response = await client.get("http://localhost:8000/agent/status/all")
        statuses = response.json()
    
    # Print all agent statuses (to be removed after testing)
    print("statuses: ", statuses)

    agent_names = [agent_name for agent_name, status in statuses.items() 
                  if status["status"] == AgentStatus.SOCIALISING.value 
                  and agent_name != socialising_agent]  # Exclude self from the list

    if not agent_names:
        log_event(agent.logger, "social.lonely", 
                 "No other agents are currently socialising. Waiting for someone to join.", 
                 level="INFO")
        return {
            "role": "system",
            "content": "No other agents available for socialising",
            "sender": "system",
            "receiver": socialising_agent,
            "timestamp": datetime.now().isoformat()
        }

    random_agent_name = choice(agent_names)
    print("random_agent_name: ", random_agent_name)
    

    # Retrieve the parent conversation ID from the context variable
    new_conversation_id = uuid.uuid4()
  
    prompt_mapping = agent.get_prompt_mapping()
    
    # Get the prompt with the lowest confidence score
    low_conf_prompt_type = min(agent.prompt.confidence_scores.items(), key=lambda x: x[1])[0] + "_prompt"

    print("######################### low_conf_prompt_type: ", low_conf_prompt_type)
    # Get the prompt content
    prompt_content = prompt_mapping[low_conf_prompt_type]

    prompt = PromptModel(
        prompt=prompt_content,
        prompt_type=low_conf_prompt_type,
        uuid=str(uuid.uuid4()),
        timestamp=datetime.utcnow().isoformat(),
        owner_agent_name=socialising_agent,
        status="sender's initial prompt"
    )

    content = (f"Hello {random_agent_name}, I am {socialising_agent}. I'd like to share my {low_conf_prompt_type} "
               f"with you and get your feedback.")

    log_agent_message(agent.logger, "out", socialising_agent, random_agent_name, content)

    # Send the message to the directory service
    async with httpx.AsyncClient(timeout=300.0) as client:
        message = SocialMessage.create_with_prompt(
            sender=socialising_agent,
            receiver=random_agent_name,
            content=content,
            conversation_id=str(new_conversation_id),
            prompt=prompt,
            message_type="InitialPrompt"
        )
        
        try:
            log_event(agent.logger, "directory.route", 
                        f"Sending social message to directory service", level="DEBUG")
            log_event(agent.logger, "directory.route", 
                        f"Request payload: {message.dict()}", level="DEBUG")
            
            response = await client.post(
                "http://localhost:8000/agent/message",
                json=message.dict(),
                timeout=300.0
            )
            
            log_event(agent.logger, "directory.route", 
                        f"Received response: Status {response.status_code}", level="DEBUG")
            log_event(agent.logger, "directory.route", 
                        f"Response content: {response.text}", level="DEBUG")
            
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as e:
                log_error(agent.logger, f"HTTP error occurred in start_socialising_impl: {str(e)}")
                log_error(agent.logger, f"Response content: {response.text}")
                raise
            
            result = response.json()
            
            # Handle social message response
            if isinstance(message, SocialMessage):
                # Extract evaluation if present
                evaluation = None
                if "evaluation" in result:
                    evaluation = PromptEvaluation(**result["evaluation"])
                    
                return {
                    "role": "assistant",
                    "content": result.get("message", ""),
                    "sender": random_agent_name,
                    "receiver": socialising_agent,
                    "evaluation": evaluation.dict() if evaluation else None,
                    "timestamp": result.get("timestamp") or datetime.now().isoformat()
                }
            
            # Handle regular API message response
            return {
                "role": "assistant",
                "content": result.get("message", ""),
                "sender": random_agent_name,
                "receiver": socialising_agent,
                "timestamp": result.get("timestamp") or datetime.now().isoformat()
            }
            
        except httpx.TimeoutException as e:
            error_msg = f"Request timed out while sending social message to {random_agent_name}"
            log_error(agent.logger, error_msg, exc_info=e)
            return {
                "role": "error",
                "content": f"{error_msg}: {str(e)}",
                "sender": "system",
                "receiver": socialising_agent,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            log_error(agent.logger, f"Unexpected error in start_socialising", exc_info=e)
            raise
