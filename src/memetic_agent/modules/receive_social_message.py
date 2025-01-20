from src.base_agent.base_agent import BaseAgent
from src.log_config import log_event
from src.api_server.models.api_models import SocialMessage, PromptModel
import uuid
from datetime import datetime

async def receive_social_message(agent: BaseAgent, message: SocialMessage) -> str:
    """Process incoming social messages containing prompts for review"""

    #TODO: We need to decide if the receiving agent will also potentially update its prompt or will it continue
    # Log the social interaction
    sender = message.sender
    content = message.content
    conversation_id = message.conversation_id
    
    log_event(agent.logger, "social.receive", 
             f"Received social message from {sender} about prompt: {message.prompt.prompt_type}")

    prompt_mapping = agent.get_prompt_mapping()

    received_prompt = message.prompt.prompt
    received_prompt_type = message.prompt.prompt_type
    received_message = message.content

    agents_version_of_prompt = prompt_mapping[received_prompt_type]

    combined_evalutor_prompt = (
        f"{agent._evaluator_prompt}\n\nType of prompt being evaluated: {received_prompt_type}"
        f"\n\n{agent.config.agent_name}'s version of prompt: {agents_version_of_prompt}"
    )

    combined_social_message = (
        f"Message received: {received_message}\n\n"
        f"{sender}'s version of prompt (to be evaluated): {received_prompt}"
    )

    evaluator_response = await agent.client.chat.completions.create(
        model=agent.config.model,
        messages=[
            {"role": "system", "content": combined_evalutor_prompt},
            {"role": "user", "content": combined_social_message}
        ]
    )

    evaluator_message = evaluator_response.choices[0].message.content.strip()

    # Need to add:
    # 1. Create response SocialMessage
    response_message = SocialMessage.create(
        sender=agent.config.agent_name,
        receiver=sender,
        content=evaluator_message,
        conversation_id=message.conversation_id,
        prompt=PromptModel(
            prompt=agents_version_of_prompt,
            prompt_type=received_prompt_type,
            uuid=str(uuid.uuid4()),
            timestamp=datetime.utcnow().isoformat(),
            owner_agent_name=agent.config.agent_name,
            status="receiver's response"
        )
    )

    # 2. Return formatted response for API
    return response_message.dict()