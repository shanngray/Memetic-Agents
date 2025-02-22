from src.base_agent.type import Agent
from src.base_agent.models import Message
from typing import List
from src.log_config import log_event, log_error


async def continue_or_stop_impl(agent: Agent, messages: List[Message]) -> str:
    """Evaluate message content to determine if we should continue or stop.
    
    Args:
        messages: List of conversation messages to evaluate
        
    Returns:
        "STOP" if conversation should end, or a rephrased continuation message
    """
    try:
        # Create prompt to evaluate conversation state
        system_prompt = Message(
            role="user" if agent.config.model == "o1-mini" else "developer" if agent.config.model == "o3-mini" else "system",
            content=f"{agent.prompt.thought_loop.content}\n\n{agent.prompt.thought_loop.schema}"
        )
        
        # Add thought loop prompt to the end of the message list
        messages.append(system_prompt)
        
        # Make LLM call to evaluate
        raw_response = await agent.client.chat.completions.create(
            model=agent.config.model,
            messages=[msg.dict() for msg in messages],
            **({"temperature": 0.7} if agent.config.model not in ["o1-mini", "o3-mini"] else {}),
            **({"reasoning_effort": agent.config.reasoning_effort} if agent.config.model == "o3-mini" else {})
        )
        
        response = raw_response.choices[0].message.content.strip()

        log_event(agent.logger, "conversation.evaluation", "Starting continue/stop evaluation", level="DEBUG")
        log_event(agent.logger, "conversation.agent", f"Agent: {agent.config.agent_name}", level="DEBUG")
        
        # Log messages in a condensed format
        for msg in messages:
            msg_details = {k: v for k, v in {
                'role': msg.role,
                'content': msg.content,
                'sender': msg.sender,
                'receiver': msg.receiver,
                'name': msg.name,
                'tool_calls': msg.tool_calls
            }.items() if v}  # Only include non-empty fields
            
            log_event(agent.logger, "conversation.message", f"Message details: {msg_details}", level="DEBUG")
        
        log_event(agent.logger, "conversation.response", f"LLM Response: {response}", level="DEBUG")       
        
        # Add human oversight
        print("\n|------------HUMAN OVERSIGHT------------|\n")
        print("\nProposed response:", response)
        human_input = input("\nAccept this response? (y/n, or type new response): ").strip()
        
        if human_input.lower() == 'y':
            pass  # Keep existing response
        elif human_input.lower() == 'n':
            response = "STOP"  # Force conversation to end
        else:
            # Use human's custom response
            response = human_input
        # Log the continuation decision
        if "STOP" in response:
            log_event(agent.logger, "conversation.complete", "Conversation marked as complete")
        else:
            log_event(agent.logger, "conversation.continue", f"Continuing conversation: {response}")
            
        return response
        
    except Exception as e:
        log_error(agent.logger, "Error evaluating conversation continuation", exc_info=e)
        return "STOP"  # Default to stopping on error

    return response