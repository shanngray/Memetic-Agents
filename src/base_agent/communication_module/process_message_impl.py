from typing import Any, Dict, List
from src.log_config import log_event, log_error, log_agent_message
from src.base_agent.models import Message, ToolCall, AgentStatus
from src.base_agent.tool_manager import ToolError
import asyncio
import uuid
from datetime import datetime
import json
import asyncio
from src.base_agent.type import Agent

async def process_message_impl(agent: Agent, content: str, sender: str, conversation_id: str) -> str:
    if agent.status == AgentStatus.SHUTTING_DOWN:
        raise ValueError("Agent is shutting down")

    try:
        log_event(agent.logger, "session.started", 
                    f"Processing message from {sender} in conversation {conversation_id}")
        log_event(agent.logger, "message.content", f"Content: {content}", level="DEBUG")

        # Set the current conversation ID in the context variable
        token = agent.current_conversation_id.set(conversation_id)  # Added to set context

        if conversation_id not in agent.conversations:
            agent.conversations[conversation_id] = [
                Message(
                    role="user" if agent.config.model == "o1-mini" else "developer" if agent.config.model == "o3-mini" else "system", 
                    content=agent._system_prompt
                    )
            ]
        
        messages = agent.conversations[conversation_id]
        
        # Add user message if not duplicate
        last_message = messages[-1] if messages else None
        if not last_message or last_message.content != content:
            user_message = Message(
                role="user",
                content=content,
                sender=sender,
                name=None,
                tool_calls=None,
                tool_call_id=None,
                receiver=agent.config.agent_name,
                timestamp=datetime.now().isoformat()
            )
            messages.append(user_message)
            log_event(agent.logger, "message.added", 
                        f"Added user message to conversation {conversation_id}")

        final_response = None
        iteration_count = 0
        max_iterations = 5
        while iteration_count < max_iterations:
            try:
                log_event(agent.logger, "openai.request", 
                            f"Sending request to OpenAI with {len(messages)} messages")
                
                response = await asyncio.wait_for(
                    agent.client.chat.completions.create(
                        model=agent.config.model,
                        **({"temperature": agent.config.temperature} if agent.config.model not in ["o1-mini", "o3-mini"] else {}),
                        messages=[m.dict() for m in messages],
                        tools=list(agent.tools.values()) if agent.tools else None,
                        timeout=3000,
                        **({"reasoning_effort": agent.config.reasoning_effort} if agent.config.model == "o3-mini" else {})
                    ),
                    timeout=3050
                )
                print("\n|---------------PROCESS MESSAGE-----------------|\n")
                print("ITERATION:", iteration_count)
                print("\nAGENT NAME:", agent.config.agent_name)
                # Print response details
                print("\nResponse Details:")
                print(f"  Model: {response.model}")
                print(f"  Created: {response.created}")
                
                # Print message details
                print("\nMessage Details:")
                message = response.choices[0].message
                print(f"  Role: {message.role}")
                print(f"  Content: {message.content}")
                
                # Print tool call details if present
                if message.tool_calls:
                    print("\nTool Calls:")
                    for tc in message.tool_calls:
                        print(f"\n  Tool Call ID: {tc.id}")
                        print(f"  Type: {tc.type}")
                        print(f"  Function Name: {tc.function.name}")
                        print(f"  Arguments: {tc.function.arguments}")
                print("\n|----------------------------------------------|\n")

                raw_message = response.choices[0].message
                message_content = raw_message.content or ""
                
                # Create and add assistant message
                assistant_message = Message(
                    role="assistant",
                    content=message_content,
                    tool_calls=[ToolCall(
                        id=tc.id,
                        type=tc.type,
                        function={
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    ) for tc in raw_message.tool_calls] if raw_message.tool_calls else None,
                    sender=agent.config.agent_name,
                    receiver=sender,
                    timestamp=datetime.now().isoformat(),
                    name=None,
                    tool_call_id=None
                )
                messages.append(assistant_message)
                log_event(agent.logger, "message.added", 
                            f"Added assistant message to conversation {conversation_id}")
                
                # Process tool calls if present
                if raw_message.tool_calls:
                    log_event(agent.logger, "tool.processing", 
                                f"Processing {len(raw_message.tool_calls)} tool calls")
                    
                    tool_calls = [ToolCall(
                        id=tc.id,
                        type=tc.type,
                        function={
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    ) for tc in raw_message.tool_calls]
                    
                    # Execute each tool and add results
                    tool_iteration_count = 0
                    for tool_call in tool_calls:
                        tool_iteration_count += 1
                        try:
                            tool_result = await agent.tool_mod.execute(tool_call)
                            tool_message = Message(
                                role="tool",
                                content=json.dumps(tool_result),
                                tool_call_id=tool_call.id,
                                timestamp=datetime.now().isoformat(),
                                tool_calls=None,
                                sender=tool_call.function['name'],
                                receiver=agent.config.agent_name
                            )
                            print("\n<<----------------TOOL CALL MESSAGE---------------->>\n")
                            print("ITERATION:", tool_iteration_count)
                            print(
                                f"\nTOOL MESSAGE: {tool_message.content}"
                                f"\nSENDER: {tool_message.sender}"
                                f"\nRECEIVER: {tool_message.receiver}"
                                f"\nTOOL CALL ID: {tool_message.tool_call_id}"
                                f"\nTIMESTAMP: {tool_message.timestamp}"
                            )
                            print("\n<<----------------------------------------------->>\n")
                            messages.append(tool_message)
                            log_event(agent.logger, "tool.result", 
                                        f"Added tool result for {tool_call.function['name']}")
                            
                        except ToolError as e:
                            error_message = Message(
                                role="tool",
                                content=json.dumps({"error": str(e)}),
                                tool_call_id=tool_call.id,
                                timestamp=datetime.now().isoformat(),
                                name=None,
                                tool_calls=None,
                                sender=agent.config.agent_name,
                                receiver=sender
                            )
                            messages.append(error_message)
                            log_error(agent.logger, 
                                        f"Tool error in {tool_call.function['name']}: {str(e)}")
                                            
                    continue  # Continue loop to process tool results
                
                # If we have a message without tool calls, evaluate to see if we should continue or stop
                if message_content:
                    continue_or_stop = await agent._continue_or_stop(messages)
                    if continue_or_stop == "STOP":
                        final_response = message_content
                        log_event(agent.logger, "message.final", 
                                    f"Final response generated for conversation {conversation_id}")
                        break
                    else:
                        # Wrap and append non-STOP message
                        messages.append(Message(
                            role="user",
                            content=continue_or_stop,
                            timestamp=datetime.now().isoformat(),
                            name=agent.config.agent_name,
                            tool_calls=None,
                            tool_call_id=None,
                            sender=agent.config.agent_name,
                            receiver=agent.config.agent_name
                        ))
                        log_event(agent.logger, "message.continue", 
                                    f"Added intermediate response for conversation {conversation_id}")
                
            except asyncio.TimeoutError as e:
                log_error(agent.logger, "OpenAI request timed out", exc_info=e)
                raise
            except Exception as e:
                log_error(agent.logger, "Error during message processing", exc_info=e)
                raise
            
            iteration_count += 1
        
        if iteration_count >= max_iterations:
            log_event(agent.logger, "message.max_iterations", 
                        f"Reached maximum iterations ({max_iterations}) for conversation {conversation_id}")
            final_response = f"I apologize, but I only got to here: {message_content}"
                    
        if final_response is None:
            error_msg = "Failed to generate a proper response"
            log_error(agent.logger, error_msg)
            return "I apologize, but I wasn't able to generate a proper response."
        
        return final_response

    finally:
        # Reset status and context
        agent.current_conversation_id.reset(token)
