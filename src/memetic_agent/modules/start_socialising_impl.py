from datetime import datetime
import httpx
import asyncio
from typing import Dict, Any
from src.base_agent.models import AgentStatus
from src.base_agent.type import Agent
from src.log_config import log_agent_message, log_event, log_error
from src.api_server.models.api_models import APIMessage, SocialMessage, PromptModel
from random import choice
import uuid

async def start_socialising_impl(agent: Agent) -> Dict[str, Any]:
    """Start socialising with another agent."""

    # 1) Lookup all agents in the directory service that are in 'socialising' status
    # 2) Select a random agent from the list
    # 3) Send a message to the selected agent
    # 4) Wait for a response from the selected agent
    # 5) If the response is a 'social message' then process it
    # 6) If the response is not a 'social message' then send a rote response

    socialising_agent = agent.config.agent_name

    async with httpx.AsyncClient(timeout=3000.0) as client:
        response = await client.get("http://localhost:8000/agent/status/all")
        statuses = response.json()
    
    # Print all agent statuses (to be removed after testing)
    print("statuses: ", statuses)

    agent_names = [agent_name for agent_name, status in statuses.items() if status["status"] == AgentStatus.SOCIALISING.value]

    random_agent_name = choice(agent_names)
    print("random_agent_name: ", random_agent_name)
    

    # Retrieve the parent conversation ID from the context variable
    new_conversation_id = uuid.uuid4()
  
    #The agent now needs to decide on which prompt it will send to the other agent.
    #For now, we will just choose a random prompt from the agent's prompt list.
    #Future state will involve the gaent choosing the promopt with the lowest confidence score.

    #TODO: Implement prompt selection logic

    prompt_mapping = agent.get_prompt_mapping()
    
    # Select a random prompt name and get its content
    random_prompt_name = choice(list(prompt_mapping.keys()))
    prompt_content = prompt_mapping[random_prompt_name]

    random_prompt = PromptModel(
        prompt=prompt_content,
        prompt_type=random_prompt_name,
        uuid=str(uuid.uuid4()),
        timestamp=datetime.utcnow().isoformat(),
        owner_agent_name=socialising_agent,
        status="sender's initial prompt"
    )

    content = (f"Hello {random_agent_name}, I am {socialising_agent}. I'd like to share my {random_prompt_name} "
               f"with you and get your feedback.")

    log_agent_message(agent.logger, "out", socialising_agent, random_agent_name, content)

    # Send the message to the directory service
    async with httpx.AsyncClient(timeout=3000.0) as client:
        message = SocialMessage.create(
            sender=socialising_agent,
            receiver=random_agent_name,
            content=content,
            conversation_id=str(new_conversation_id),
            prompt=random_prompt
        )
        
        try:
            log_event(agent.logger, "directory.route", 
                        f"Sending social message to directory service", level="DEBUG")
            log_event(agent.logger, "directory.route", 
                        f"Request payload: {message.dict()}", level="DEBUG")
            
            response = await client.post(
                "http://localhost:8000/agent/message",
                json=message.dict(),
                timeout=3000.0
            )
            
            log_event(agent.logger, "directory.route", 
                        f"Received response: Status {response.status_code}", level="DEBUG")
            log_event(agent.logger, "directory.route", 
                        f"Response content: {response.text}", level="DEBUG")
            
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as e:
                log_error(agent.logger, f"HTTP error occurred: {str(e)}")
                log_error(agent.logger, f"Response content: {response.text}")
                raise
            
            result = response.json()
            # Extract message content, handling both string and dict responses
            message_content = result.get("message", "")
            prompt_data = None
            
            if isinstance(message_content, dict):
                # Extract prompt data if it exists
                if "prompt" in message_content:
                    prompt_data = PromptModel(**message_content["prompt"])
                message_content = message_content.get("message", "")
            
            log_agent_message(agent.logger, "in", random_agent_name, socialising_agent, message_content)
                        
            return {
                "role": "assistant",
                "content": message_content,
                "sender": random_agent_name,
                "receiver": socialising_agent,
                "prompt": prompt_data.dict() if prompt_data else None,
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
        finally:
            
            if agent.status == AgentStatus.MESSAGE_PROCESSING or agent.status == AgentStatus.WAITING_RESPONSE:
                await agent.set_status(AgentStatus.SOCIALISING, "completed start_socialising")
