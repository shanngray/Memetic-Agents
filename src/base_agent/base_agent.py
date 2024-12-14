import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Callable, Union, Tuple
from openai import AsyncOpenAI
import os
import sys
import importlib.util
from pathlib import Path
import httpx
import atexit
import signal
import traceback
import uuid
from fastapi import HTTPException
import inspect
from chromadb import PersistentClient
from filelock import FileLock
from icecream import ic
import contextvars

# Add the project root directory to the Python path
sys.path.append(str(Path(__file__).parents[2]))

from .models import Message, ToolCall, AgentStatus, Request
from .config import AgentConfig
from api_server.models.api_models import APIMessage
from src.log_config import setup_logger, log_tool_call, log_agent_message, log_error, log_event
from src.memory.memory_manager import MemoryManager
from src.base_agent.tool_manager import ToolManager, ToolError

class BaseAgent:
    def __init__(
        self,
        api_key: str,
        chroma_client: PersistentClient,
        config: Optional[AgentConfig] = None
    ):
        """Initialize the base agent with OpenAI client and configuration."""
        super().__init__()
        
        self.config = config or AgentConfig()
        
        # Setup folder for agent files
        self.files_path = Path("agent_files") / self.config.agent_name
        self.files_path.mkdir(parents=True, exist_ok=True)

        self.client = AsyncOpenAI(api_key=api_key)
        self.status = AgentStatus.IDLE
        self.pending_requests: Dict[str, asyncio.Future] = {}
        self.request_queue: asyncio.Queue = asyncio.Queue()
        self.waiting_for: Optional[str] = None
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.internal_tools: Dict[str, Callable] = {}
        self.old_conversation_list: Dict[str, str] = {}  # Dictionary where key is conversation_id and value is short/long
        # Example of a conversation entry:
        # self.conversation_list = {
        #     "conversation_id_1": "short_term",
        #     "conversation_id_2": "long_term"
        # }

        # Define a context variable to keep track of the current conversation ID
        self.current_conversation_id = contextvars.ContextVar('current_conversation_id', default=None)
        
        # Initialize logger before any other operations
        self.logger = setup_logger(
            name=self.config.agent_name,
            log_path=self.config.log_path,
            level=self.config.log_level,
            console_logging=self.config.console_logging
        )
        
        self._setup_logging()

        self.tool_mod = ToolManager(self)
        
        # Add send_message to internal tools
        self.internal_tools["send_message"] = self.send_message
        self.internal_tools["search_memory"] = self.search_memory

        # Register all internal tools AFTER adding them to tool_mod.internal
        for tool_name, tool_func in self.internal_tools.items():
            self.tool_mod.register(tool_func)
            
        # Load enabled external tools
        if self.config.enabled_tools:
            self._load_tool_definitions()

        #ToDO: check if this is used
        self.conversation_histories: Dict[str, List[Message]] = {}
        
        self.agent_directory = None  # Will be set during registration

        # Initialize memory manager with default collections
        self.memory = MemoryManager(
            agent_name=self.config.agent_name,
            logger=self.logger,
            chroma_client=chroma_client
        )
        


        # Initialize collections including feedback
        asyncio.create_task(self.memory.initialize(
            collection_names=["short_term", "long_term", "feedback"]
        ))
        

        self.conversations: Dict[str, List[Message]] = {}


        # Load existing memory if available
        self._load_memory()
        
        # Register shutdown handlers
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

        self._previous_status = None
        self._status_lock = asyncio.Lock()
        self._shutdown_event = asyncio.Event()

    def _setup_logging(self) -> None:
        """Configure logging if debug is enabled."""
        if self.config.debug:
            self.config.log_path.mkdir(exist_ok=True)
            
            # Only add handlers if none exist
            if not self.logger.handlers:
                # Always add file handler when debug is enabled
                file_handler = logging.FileHandler(
                    self.config.log_path / f'{self.config.agent_name.lower().replace(" ", "_")}_debug_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
                )
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)
                
                # Add console handler only if enabled
                if self.config.console_logging:
                    console_handler = logging.StreamHandler()
                    console_handler.setFormatter(formatter)
                    self.logger.addHandler(console_handler)
                    
                self.logger.setLevel(logging.DEBUG)
                
            self.logger.info("BaseAgent initialized")

        log_event(self.logger, "agent.registered", f"Initialized {self.config.agent_name}")

    def _load_tool_definitions(self) -> None:
        """Load tool definitions from JSON files."""
        for tool_name in self.config.enabled_tools:
            try:
                tool_path = self.config.tools_path / f"{tool_name}.json"
                if not tool_path.exists():
                    raise ToolError(f"Tool definition not found: {tool_path}")

                log_event(self.logger, "tool.loading", f"Loading tool definition: {tool_path}")
                with open(tool_path) as f:
                    tool_def = json.load(f)
                    self.tool_mod.register(tool_def)
                    
            except Exception as e:
                log_event(self.logger, "tool.error", f"Failed to load tool {tool_name}: {str(e)}")

    def register_tools(self, tools: List[Dict[str, Any]]) -> None:
        """Register multiple tools/functions that the agent can use.
        
        Args:
            tools: List of tool definitions to register
            
        Raises:
            ValueError: If any tool definition is invalid
        """
        for tool in tools:
            self.tool_mod.register(tool)
            

    def _trim_history(self, conversation_id: str) -> None:
        """Trim conversation history to max_history length."""
        if conversation_id not in self.conversations:
            return
            
        messages = self.conversations[conversation_id]
        if len(messages) > self.config.max_history:
            self.logger.debug(f"Trimming history from {len(messages)} messages")
            self.conversations[conversation_id] = [messages[0]] + messages[-(self.config.max_history-1):]

    async def process_message(self, content: str, sender: str, conversation_id: str) -> str:
        if self.status == AgentStatus.SHUTTING_DOWN:
            raise ValueError("Agent is shutting down")

        await self.set_status(AgentStatus.PROCESSING)

        try:
            log_event(self.logger, "session.started", 
                      f"Processing message from {sender} in conversation {conversation_id}")
            log_event(self.logger, "message.content", f"Content: {content}", level="DEBUG")

            # Set the current conversation ID in the context variable
            token = self.current_conversation_id.set(conversation_id)  # Added to set context

            if conversation_id not in self.conversations:
                self.conversations[conversation_id] = [
                    Message(role="system", content=self.config.system_prompt)
                ]
            
            messages = self.conversations[conversation_id]
            
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
                    receiver=self.config.agent_name,
                    timestamp=datetime.now().isoformat()
                )
                messages.append(user_message)
                log_event(self.logger, "message.added", 
                         f"Added user message to conversation {conversation_id}")

            final_response = None
            iteration_count = 0
            max_iterations = 5
            while iteration_count < max_iterations:
                try:
                    log_event(self.logger, "openai.request", 
                             f"Sending request to OpenAI with {len(messages)} messages")
                    
                    response = await asyncio.wait_for(
                        self.client.chat.completions.create(
                            model=self.config.model,
                            temperature=self.config.temperature,
                            messages=[m.dict() for m in messages],
                            tools=list(self.tools.values()) if self.tools else None,
                            timeout=3000
                        ),
                        timeout=3050
                    )
                    print("\n|---------------PROCESS MESSAGE-----------------|\n")
                    print("ITERATION:", iteration_count)
                    print("\nAGENT NAME:", self.config.agent_name)
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
                    print("\n|--------------------------------|\n")

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
                        sender=self.config.agent_name,
                        receiver=sender,
                        timestamp=datetime.now().isoformat(),
                        name=None,
                        tool_call_id=None
                    )
                    messages.append(assistant_message)
                    log_event(self.logger, "message.added", 
                             f"Added assistant message to conversation {conversation_id}")
                    
                    # Process tool calls if present
                    if raw_message.tool_calls:
                        log_event(self.logger, "tool.processing", 
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
                                await self.set_status(AgentStatus.WAITING) #change status outside of execute call
                                tool_result = await self.tool_mod.execute(tool_call)
                                if self.status != AgentStatus.SHUTTING_DOWN:
                                    await self.set_status(AgentStatus.PROCESSING)
                                tool_message = Message(
                                    role="tool",
                                    content=json.dumps(tool_result),
                                    tool_call_id=tool_call.id,
                                    timestamp=datetime.now().isoformat(),
                                    tool_calls=None,
                                    sender=tool_call.function['name'],
                                    receiver=self.config.agent_name
                                )
                                print("\n----------------TOOL CALL MESSAGE----------------\n")
                                print("ITERATION:", tool_iteration_count)
                                print(
                                    f"\nTOOL MESSAGE: {tool_message.content}"
                                    f"\nSENDER: {tool_message.sender}"
                                    f"\nRECEIVER: {tool_message.receiver}"
                                    f"\nTOOL CALL ID: {tool_message.tool_call_id}"
                                    f"\nTIMESTAMP: {tool_message.timestamp}"
                                )
                                print("\n-----------------------------------------------\n")
                                messages.append(tool_message)
                                log_event(self.logger, "tool.result", 
                                         f"Added tool result for {tool_call.function['name']}")
                                
                            except ToolError as e:
                                error_message = Message(
                                    role="tool",
                                    content=json.dumps({"error": str(e)}),
                                    tool_call_id=tool_call.id,
                                    timestamp=datetime.now().isoformat(),
                                    name=None,
                                    tool_calls=None,
                                    sender=self.config.agent_name,
                                    receiver=sender
                                )
                                messages.append(error_message)
                                log_error(self.logger, 
                                         f"Tool error in {tool_call.function['name']}: {str(e)}")
                                                
                        continue  # Continue loop to process tool results
                    
                    # If we have a message without tool calls, evaluate to see if we should continue or stop
                    if message_content:
                        continue_or_stop = await self._continue_or_stop(messages)
                        if continue_or_stop == "STOP":
                            final_response = message_content
                            log_event(self.logger, "message.final", 
                                     f"Final response generated for conversation {conversation_id}")
                            break
                        else:
                            # Wrap and append non-STOP message
                            messages.append(Message(
                                role="user",
                                content=continue_or_stop,
                                timestamp=datetime.now().isoformat(),
                                name=self.config.agent_name,
                                tool_calls=None,
                                tool_call_id=None,
                                sender=self.config.agent_name,
                                receiver=self.config.agent_name
                            ))
                            log_event(self.logger, "message.continue", 
                                     f"Added intermediate response for conversation {conversation_id}")
                    
                except asyncio.TimeoutError as e:
                    log_error(self.logger, "OpenAI request timed out", exc_info=e)
                    raise
                except Exception as e:
                    log_error(self.logger, "Error during message processing", exc_info=e)
                    raise
                
                iteration_count += 1
            
            if iteration_count >= max_iterations:
                log_event(self.logger, "message.max_iterations", 
                         f"Reached maximum iterations ({max_iterations}) for conversation {conversation_id}")
                final_response = f"I apologize, but I only got to here: {message_content}"
            
            # Trim history if needed
            self._trim_history(conversation_id)
            
            if final_response is None:
                error_msg = "Failed to generate a proper response"
                log_error(self.logger, error_msg)
                return "I apologize, but I wasn't able to generate a proper response."
            
            return final_response
                
        finally:
            # Reset the context variable after processing
            self.current_conversation_id.reset(token)

    async def _continue_or_stop(self, messages: List[Message]) -> str:
        """Evaluate message content to determine if we should continue or stop.
        
        Args:
            messages: List of conversation messages to evaluate
            
        Returns:
            "STOP" if conversation should end, or a rephrased continuation message
        """
        try:
            # Create prompt to evaluate conversation state
            system_prompt = Message(
                role="system",
                content=(
                    "Based on the conversation so far, determine if a complete response has been provided "
                    "or if the initial query has been fully addressed. If complete, respond with exactly 'STOP'. "
                    "Otherwise, respond with clear instructions on what steps to take next."
                )
            )
            messages = messages[1:]  # Remove first message
            messages.insert(0, system_prompt)  # Add new system prompt at start
            # Make LLM call to evaluate
            raw_response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=[msg.dict() for msg in messages],
                temperature=0.7
            )
            
            response = raw_response.choices[0].message.content.strip()

            print("\n|------------CONTINUE OR STOP------------|\n")
            # Print out all messages and their non-empty fields
            print("AGENT NAME:", self.config.agent_name)
            print("\nMessages:")
            for msg in messages:
                print("\nMessage:\n")
                if msg.role:
                    print(f"  role: {msg.role}\n")
                if msg.content:
                    print(f"  content: {msg.content}\n")
                if msg.sender:
                    print(f"  sender: {msg.sender}\n")
                if msg.receiver:
                    print(f"  receiver: {msg.receiver}\n")
                if msg.name:
                    print(f"  name: {msg.name}\n")
                if msg.tool_calls:
                    print(f"  tool_calls: {msg.tool_calls}\n")
            ic(f"Response: {response}")
            print("\n|--------------------------------|\n")            
            
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
                log_event(self.logger, "conversation.complete", "Conversation marked as complete")
            else:
                log_event(self.logger, "conversation.continue", f"Continuing conversation: {response}")
                
            return response
            
        except Exception as e:
            log_error(self.logger, "Error evaluating conversation continuation", exc_info=e)
            return "STOP"  # Default to stopping on error

        return response

    # TODO: We might want to get rid of this. Would need to update process_message and would need to make sure that all messages have the sender and receiver fields
    def get_conversation_history(self, agent_name: str) -> List[Message]:
        """Get messages related to a specific agent across all conversations."""
        related_messages = []
        for messages in self.conversations.values():
            related_messages.extend([
                msg for msg in messages 
                if (msg.role == "system") or  # Include system prompt
                   (hasattr(msg, 'sender') and msg.sender == agent_name) or
                   (hasattr(msg, 'receiver') and msg.receiver == agent_name)
            ])
        return related_messages

    async def receive_message(self, sender: str, content: str, conversation_id: str) -> str:
        """Process received message from another agent."""
        log_event(self.logger, "agent.message_received", 
                  f"Message from {sender} in conversation {conversation_id}")
        log_event(self.logger, "message.content", f"{content}", level="DEBUG")
        
        # Get current queue size before adding new message
        queue_size = self.request_queue.qsize()
        queue_position = queue_size + 1
        
        loaded_conversation = self.current_conversation_id.get()

        if conversation_id == loaded_conversation:
            # We are in the current conversation, so we can proceed with processing the message
            pass
        elif conversation_id in self.conversations:
            # We have a short term conversation, we need to update the current conversation_id
            self.current_conversation_id.set(conversation_id)
            pass
        elif conversation_id in self.old_conversation_list:
            # We have a long term conversation, so we need to load it into the conversation history and update the current conversation_id
            self.current_conversation_id.set(conversation_id)
            self.conversations[conversation_id].append(Message(role="system", content="This is an old conversation, you may have memories in longer_term memory that will help with context."))
            pass
        else:
            # This is a new conversation, we need to update the conversation list and the current conversation_id
            log_event(self.logger, "session.created", f"Creating new conversation: {conversation_id}")
            self.current_conversation_id.set(conversation_id)
            self.conversations[conversation_id] = [
                Message(role="system", content=self.config.system_prompt)
            ]
            pass

        # Create request ID and future
        request_id = str(uuid.uuid4())
        future = asyncio.Future()
        self.pending_requests[request_id] = future
        
        # Queue the request
        request = {
            "id": request_id,
            "sender": sender,
            "content": content,
            "conversation_id": conversation_id,
            "timestamp": datetime.now().isoformat(),
            "queue_position": queue_position
        }
        log_event(self.logger, "queue.added", 
                  f"Queuing request {request_id} from {sender} (position {queue_position} in queue)")
        await self.request_queue.put(request)
        
        try:
            # Wait for response with timeout
            log_event(self.logger, "queue.status", 
                     f"Request {request_id} waiting at position {queue_position}", level="DEBUG")
            response = await asyncio.wait_for(future, timeout=300.0)
            log_event(self.logger, "queue.completed", 
                     f"Request {request_id} completed successfully (was position {queue_position})")
            return response
        except asyncio.TimeoutError:
            log_event(self.logger, "session.timeout", 
                     f"Request {request_id} timed out after 30 seconds (was position {queue_position})", 
                     level="WARNING")
            raise
        finally:
            if request_id in self.pending_requests:
                del self.pending_requests[request_id]
                log_event(self.logger, "queue.status", 
                         f"Cleaned up request {request_id} (was position {queue_position})", 
                         level="DEBUG")

    async def send_message(self, receiver: str, content: str) -> Dict[str, Any]:
        """Send a message via API to another agent registered in the directory service."""
        await self.set_status(AgentStatus.WAITING)
        sender = self.config.agent_name

        if sender == receiver:
            log_event(self.logger, "agent.error", 
                      f"Cannot send message to self: {sender} -> {receiver}", 
                      level="ERROR")
            raise ValueError(f"Cannot send message to self: {sender} -> {receiver}")
        
        # Retrieve the parent conversation ID from the context variable
        conversation_id = self.current_conversation_id.get()  # Added to get parent ID from context




        log_agent_message(self.logger, "out", sender, receiver, content)
        
        async with httpx.AsyncClient(timeout=3000.0) as client:
            message = APIMessage.create(
                sender=sender,
                receiver=receiver,
                content=content,
                conversation_id=conversation_id
            )
            
            try:
                log_event(self.logger, "directory.route", 
                         f"Sending message to directory service", level="DEBUG")
                log_event(self.logger, "directory.route", 
                         f"Request payload: {message.dict()}", level="DEBUG")
                
                response = await client.post(
                    "http://localhost:8000/agent/message",
                    json=message.dict(),
                    timeout=3000.0
                )
                
                log_event(self.logger, "directory.route", 
                         f"Received response: Status {response.status_code}", level="DEBUG")
                log_event(self.logger, "directory.route", 
                         f"Response content: {response.text}", level="DEBUG")
                
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as e:
                    log_error(self.logger, f"HTTP error occurred: {str(e)}")
                    log_error(self.logger, f"Response content: {response.text}")
                    raise
                
                result = response.json()
                message_content = result.get("message", "")
                if isinstance(message_content, dict):
                    message_content = message_content.get("message", "")
                
                log_agent_message(self.logger, "in", receiver, sender, message_content)
                
                # Evaluate response quality in background
                asyncio.create_task(self._evaluate_and_send_feedback(
                    receiver=receiver,
                    conversation_id=conversation_id,
                    response_content=message_content
                ))
                
                return {
                    "role": "assistant",
                    "content": message_content,
                    "sender": receiver,
                    "receiver": sender,
                    "timestamp": result.get("timestamp") or datetime.now().isoformat()
                }
                
            except httpx.TimeoutException as e:
                error_msg = f"Request timed out while sending message to {receiver}"
                log_error(self.logger, error_msg, exc_info=e)
                return {
                    "role": "error",
                    "content": f"{error_msg}: {str(e)}",
                    "sender": "system",
                    "receiver": sender,
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                log_error(self.logger, f"Unexpected error in send_message", exc_info=e)
                raise
            finally:
                if self.status != AgentStatus.SHUTTING_DOWN:
                    await self.set_status(AgentStatus.PROCESSING)

    async def _evaluate_and_send_feedback(
        self,
        receiver: str,
        conversation_id: str,
        response_content: str
    ) -> None:
        """Evaluate response quality and send feedback to the agent."""
        try:
            # Get evaluation from LLM
            score, feedback = await self._evaluate_response(response_content)
            
            # Send feedback via API
            async with httpx.AsyncClient() as client:
                feedback_message = {
                    "sender": self.config.agent_name,
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
                
            log_event(self.logger, "feedback.sent",
                     f"Sent feedback to {receiver} for conversation {conversation_id}")
                     
        except Exception as e:
            log_error(self.logger, f"Failed to process/send feedback: {str(e)}")

    async def _evaluate_response(self, response_content: str) -> Tuple[int, str]:
        """Evaluate response quality using LLM."""
        try:
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": """Evaluate the quality of the following response. 
                    Provide:
                    1. A score from 0-10 (where 10 is excellent)
                    2. Brief, constructive feedback (Include WHAT is good/bad, WHY it matters, and HOW to improve)
                    
                    Format your response as a JSON object with 'score' and 'feedback' fields."""},
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
            log_error(self.logger, f"Failed to evaluate response: {str(e)}")
            return 5, "Error evaluating response"  # Default neutral score

    async def receive_feedback(
        self,
        sender: str,
        conversation_id: str,
        score: int,
        feedback: str
    ) -> None:
        """Process and store received feedback from another agent.
        
        Args:
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
            
            await self.memory.store(
                content=formatted_feedback,
                collection_name="feedback",
                metadata=metadata
            )
            
            log_event(self.logger, "feedback.received",
                     f"Received feedback from {sender} for conversation {conversation_id}")
                     
        except Exception as e:
            log_error(self.logger, f"Failed to process received feedback: {str(e)}")

    def _load_memory(self) -> None:
        """Load short term memories into conversations and list of long term memories into old_conversation_list."""
        try:
            # Initialize empty containers
            self.conversations = {}
            self.old_conversation_list = {}
            
            # Check if short-term collection exists
            if "short_term" not in self.memory.collections:
                log_event(self.logger, "memory.init", 
                         "No short-term memory collection found - starting fresh")
                return
            
            # Part 1: Load conversations from short-term memory
            short_term_collection = self.memory._get_collection("short_term")
            results = short_term_collection.get()
            
            if not results["documents"]:
                log_event(self.logger, "memory.init", 
                         "Short-term memory collection is empty - starting fresh")
                return
            
            # Group results by conversation_id
            conversation_groups = {}
            for doc, metadata in zip(results["documents"], results["metadatas"]):
                conv_id = metadata.get("conversation_id")
                if conv_id:
                    if conv_id not in conversation_groups:
                        conversation_groups[conv_id] = {
                            "content": [],
                            "participants": set(),
                            "timestamps": []
                        }
                    conversation_groups[conv_id]["content"].append(doc)
                    if "participants" in metadata:
                        conversation_groups[conv_id]["participants"].update(
                            metadata["participants"].split(",")
                        )
                    if "timestamp" in metadata:
                        conversation_groups[conv_id]["timestamps"].append(metadata["timestamp"])

            # Convert grouped content into conversations
            for conv_id, group in conversation_groups.items():
                # Start with system prompt
                messages = [Message(role="system", content=self.config.system_prompt)]
                
                # Combine all content for this conversation
                combined_content = "\n".join(group["content"])
                
                # Parse the combined content into messages
                try:
                    # Split content into message chunks and convert to Message objects
                    message_chunks = combined_content.split("\n")
                    for chunk in message_chunks:
                        if chunk.strip():
                            # Try to parse role and content from chunk
                            if ": " in chunk:
                                role, content = chunk.split(": ", 1)
                                # Convert role to standard format
                                role = role.lower()
                                if role not in ["system", "user", "assistant", "tool"]:
                                    role = "user"
                            else:
                                # Default to user role if format is unclear
                                role = "user"
                                content = chunk
                                
                            messages.append(Message(
                                role=role,
                                content=content,
                                timestamp=min(group["timestamps"]) if group["timestamps"] else datetime.now().isoformat()
                            ))
                
                    self.conversations[conv_id] = messages
                    log_event(self.logger, "memory.loaded", 
                             f"Loaded conversation {conv_id} with {len(messages)} messages")
                    
                except Exception as e:
                    log_error(self.logger, 
                             f"Error parsing conversation {conv_id}: {str(e)}")
                    continue

            # Part 2: Load old conversations list
            old_conversations_file = self.files_path / "old_conversations.json"
            if old_conversations_file.exists():
                try:
                    with open(old_conversations_file, "r") as f:
                        self.old_conversation_list = json.load(f)
                    log_event(self.logger, "memory.loaded", 
                             f"Loaded {len(self.old_conversation_list)} old conversations")
                except json.JSONDecodeError as e:
                    log_error(self.logger, 
                             f"Error loading old conversations: {str(e)}")
                    # Initialize empty if file is corrupted
                    self.old_conversation_list = {}
            else:
                # Initialize empty if file doesn't exist
                self.old_conversation_list = {}

        except Exception as e:
            log_error(self.logger, "Failed to load memory", exc_info=e)
            # Initialize empty containers on error
            self.conversations = {}
            self.old_conversation_list = {}

    async def _save_memory(self) -> None:
        """Save new messages from conversations to memory store."""
        try:
            # Get the latest timestamp for each conversation from short-term memory
            short_term = self.memory._get_collection("short_term")
            latest_timestamps = {}
            
            for conversation_id in self.conversations:
                # Changed from get() with order_by to get() with where clause
                results = short_term.get(
                    where={"conversation_id": conversation_id}
                )
                
                # Manually find the latest timestamp from results
                if results["metadatas"]:
                    timestamps = [
                        metadata.get("timestamp") 
                        for metadata in results["metadatas"] 
                        if metadata.get("timestamp")
                    ]
                    if timestamps:
                        latest_timestamps[conversation_id] = max(timestamps)
                    else:
                        latest_timestamps[conversation_id] = None
                else:
                    latest_timestamps[conversation_id] = None

            # Save only new messages for each conversation
            for conversation_id, conversation in self.conversations.items():
                latest_timestamp = latest_timestamps.get(conversation_id)
                
                # Filter messages that are newer than the latest saved timestamp
                new_messages = []
                if latest_timestamp:
                    new_messages = [
                        msg for msg in conversation 
                        if msg.role != "system" and msg.timestamp > latest_timestamp
                    ]
                else:
                    # If no previous messages, save all except system message
                    new_messages = [msg for msg in conversation if msg.role != "system"]

                if not new_messages:
                    continue

                # Format new messages for storage
                content = "\n".join([
                    f"{msg.role}: {msg.content}" 
                    for msg in new_messages
                ])
                
                # Get all participants from new messages
                participants = set(
                    getattr(msg, 'sender', None) or msg.role 
                    for msg in new_messages
                )
                participants_str = ",".join(sorted(participants))
                
                metadata = {
                    "conversation_id": conversation_id,
                    "timestamp": datetime.now().isoformat(),
                    "participants": participants_str
                }
                
                try:
                    await self.memory.store(
                        content=content,
                        collection_name="short_term",
                        metadata=metadata
                    )
                    log_event(self.logger, "memory.saved", 
                             f"Saved {len(new_messages)} new messages for conversation {conversation_id}")
                    
                    # Save memory dump to disk
                    memory_dump_path = self.files_path / f"{self.config.agent_name}_short_term_memory_dump.md"
                    try:
                        with FileLock(f"{memory_dump_path}.lock"):
                            with open(memory_dump_path, "a", encoding="utf-8") as f:
                                # Add timestamp header
                                f.write(f"\n\n## Memory Entry {datetime.now().isoformat()}\n")
                                for key, value in metadata.items():
                                    f.write(f"{key}: {value}\n")
                                f.write("\n")
                                f.write(content)
                                f.write("\n---\n")
                        log_event(self.logger, "memory.dumped", 
                                f"Saved new messages for conversation {conversation_id} to disk")
                    except Exception as e:
                        log_error(self.logger,
                                f"Failed to save conversation {conversation_id} to disk: {str(e)}")
                    
                except Exception as e:
                    log_error(self.logger, 
                             f"Failed to save conversation {conversation_id}: {str(e)}")
                    
        except Exception as e:
            log_error(self.logger, "Failed to save memory", exc_info=e)

    def _handle_shutdown(self, signum, frame) -> None:
        """Handle shutdown signals gracefully by creating async task."""
        self.logger.info("Shutdown signal received")
        # Create async task for shutdown which includes memory saving
        asyncio.create_task(self.shutdown())

    async def shutdown(self) -> None:
        """Gracefully shutdown the agent."""
        await self.set_status(AgentStatus.SHUTTING_DOWN)
        self._shutdown_event.set()
        
        self.logger.info(f"Shutting down {self.config.agent_name}")
        # Save memory as part of shutdown
        await self._save_memory()
        
        # Cancel any pending requests
        for future in self.pending_requests.values():
            if not future.done():
                future.cancel()
        
        # Clear queues
        while not self.request_queue.empty():
            try:
                self.request_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def start(self):
        """Start the agent's main processing loop and initialise memories"""
        self.logger.info(f"Starting {self.config.agent_name} processing loop")
        
        # Initialize memory collections
        await self.memory.initialize()

        while not self._shutdown_event.is_set():
            try:
                await self.process_queue()
                await asyncio.sleep(0.1)  # Prevent CPU spinning
            except Exception as e:
                self.logger.error(f"Error in processing loop: {str(e)}")
                if self.status != AgentStatus.SHUTTING_DOWN:
                    await self.set_status(AgentStatus.IDLE)

    async def process_queue(self):
        """Process any pending requests in the queue"""
        if self.status == AgentStatus.SHUTTING_DOWN:
            return

        if not self.request_queue.empty():
            current_queue_size = self.request_queue.qsize()
            request = await self.request_queue.get()
            queue_position = request.get('queue_position', 'unknown')
            
            await self.set_status(AgentStatus.PROCESSING)
            try:
                log_event(self.logger, "queue.dequeued", 
                         f"Processing request {request['id']} from {request['sender']} "
                         f"(position {queue_position}/{current_queue_size} in queue)")
                
                log_event(self.logger, "queue.processing", 
                         f"Processing message content: {request['content']} "
                         f"(was position {queue_position})", level="DEBUG")
                response = await self.process_message(
                    content=request["content"],
                    sender=request["sender"],
                    conversation_id=request["conversation_id"]
                )
                
                # Complete the future with the response
                if request["id"] in self.pending_requests:
                    self.pending_requests[request["id"]].set_result(response)
                    log_event(self.logger, "queue.completed", 
                             f"Request {request['id']} processed successfully "
                             f"(was position {queue_position}/{current_queue_size})")
                    del self.pending_requests[request["id"]]
                    
            except Exception as e:
                log_error(self.logger, 
                         f"Error processing request {request['id']} "
                         f"(was position {queue_position}/{current_queue_size}): {str(e)}", 
                         exc_info=e)
                if request["id"] in self.pending_requests:
                    self.pending_requests[request["id"]].set_exception(e)
                    del self.pending_requests[request["id"]]
            finally:
                if self.status != AgentStatus.SHUTTING_DOWN:
                    await self.set_status(AgentStatus.IDLE)

    async def set_status(self, new_status: AgentStatus) -> None:
        """Update agent status with validation and logging."""
        async with self._status_lock:
            if new_status == self.status:
                return

            valid_transitions = AgentStatus.get_valid_transitions(self.status)
            if new_status not in valid_transitions:
                raise ValueError(
                    f"Invalid status transition from {self.status} to {new_status}"
                )

            self._previous_status = self.status
            self.status = new_status
            log_event(
                self.logger, 
                f"agent.{self.status}", 
                f"Status changed from {self._previous_status} to {self.status}"
            )
            
            # Trigger memory consolidation when entering MEMORISING state
            if new_status == AgentStatus.MEMORISING:
                asyncio.create_task(self._transfer_to_long_term())

    async def _transfer_to_long_term(self, days_threshold: int = 0) -> None:
        """Transfer short-term memories into long-term storage using simplified SPO format.
        
        Args:
            days_threshold: Number of days worth of memories to keep in short-term. 
                           Memories older than this will be processed into long-term storage.
                           Default is 0 (process all memories).
        """
        try:
            log_event(self.logger, "agent.memorising", 
                     f"Beginning memory consolidation process (threshold: {days_threshold} days)")
            
            # Retrieve recent memories from short-term storage
            short_term_memories = await self.memory.retrieve(
                query="",  # Empty query to get all memories
                collection_names=["short_term"],
                n_results=100
            )
            
            # Filter memories based on threshold
            threshold_date = datetime.now() - timedelta(days=days_threshold)
            short_term_memories = [
                memory for memory in short_term_memories
                if datetime.fromisoformat(memory["metadata"].get("timestamp", "")) < threshold_date
            ]
            
            if not short_term_memories:
                log_event(self.logger, "memory.error", 
                         f"No memories older than {days_threshold} days found for consolidation")
                await self.set_status(self._previous_status)
                return
            
            # Process each memory individually
            for memory in short_term_memories:
                try:
                    # Extract structured information using LLM
                    response = await self.client.chat.completions.create(
                        model=self.config.model,
                        messages=[
                            {"role": "system", "content": """Analyze this memory and extract information in SPO (Subject-Predicate-Object) format.
                            Format your response as a JSON object with this structure:
                            {
                                "memories": [
                                    {
                                        "content": "SPO statement in 1-3 sentences",
                                        "type": "fact|decision|preference|pattern",
                                        "tags": ["#tag1", "#tag2", ...],
                                        "importance": 1-10
                                    }
                                ]
                            }
                            
                            Include type as a tag and add other relevant topic/context tags.
                            Keep statements clear and concise.
                            
                            Example Input: 'During the community meeting last Tuesday, Sarah Johnson presented a detailed proposal for a new youth center 
                            downtown. The proposal included a $500,000 budget plan and received strong support from local business owners. Several attendees 
                            raised concerns about parking availability, but Sarah addressed these by suggesting a partnership with the nearby church for 
                            additional parking spaces during peak hours.'

                            Example Output:
                            {
                                "memories": [
                                    {
                                        "content": "Sarah Johnson (S) presented (P) a youth center proposal at the community meeting (O). The proposal 
                                        outlined a $500,000 budget and garnered support from local businesses.",
                                        "type": "fact",
                                        "tags": ["#community_development", "#youth_services", "#proposal", "#budget"],
                                        "importance": 8
                                    }
                                ]
                            }
                            """},
                            {"role": "user", "content": f"Memory to analyze:\n{memory['content']}"}
                        ],
                        response_format={ "type": "json_object" }
                    )
                    
                    structured_info = json.loads(response.choices[0].message.content)
                    
                    # Store each extracted memory
                    for mem in structured_info["memories"]:
                        # Format content with tags section
                        formatted_content = f"{mem['content']}\n\nMetaTags: {', '.join(mem['tags'])}"
                        
                        # Simplified metadata structure with only scalar values
                        metadata = {
                            "memory_id": str(uuid.uuid4()),
                            "original_timestamp": memory["metadata"].get("timestamp"),
                            "memory_type": mem["type"],
                            "importance": mem["importance"],
                            "primary_tag": mem["tags"][0] if mem["tags"] else "#untagged",
                            "timestamp": datetime.now().isoformat()
                        }

                        await self.memory.store(
                            content=formatted_content,
                            collection_name="long_term",
                            metadata=metadata
                        )
                        
                        log_event(self.logger, "agent.memorising",
                                 f"Processed memory {metadata['memory_id']} into long-term storage")

                        # Save to disk for debugging/backup
                        await self._save_memory_to_disk(formatted_content, metadata)

                except Exception as e:
                    log_error(self.logger, f"Failed to process memory: {str(e)}", exc_info=e)
                    continue
                
            await self._cleanup_short_term_memories(days_threshold)
            
            log_event(self.logger, "memory.memorising.complete",
                     f"Completed memory consolidation for {len(short_term_memories)} memories")
                 
        except Exception as e:
            log_error(self.logger, "Failed to consolidate memories", exc_info=e)
        finally:
            if self.status != AgentStatus.SHUTTING_DOWN:
                await self.set_status(self._previous_status)

    async def _save_memory_to_disk(self, structured_info: Dict, metadata: Dict) -> None:
        """Save structured memory information to disk for debugging/backup."""
        try:
            memory_dump_path = self.files_path / f"{self.config.agent_name}_long_term_memory_dump.md"
            with FileLock(f"{memory_dump_path}.lock"):
                with open(memory_dump_path, "a", encoding="utf-8") as f:
                    f.write(f"\n\n## Structured Memory Entry {metadata['timestamp']}\n")
                    f.write("### Metadata\n")
                    for key, value in metadata.items():
                        if isinstance(value, list):
                            value = ", ".join(value)
                        f.write(f"{key}: {value}\n")
                    
                    f.write("\n### Extracted Information\n")
                    # Only write structured info if it's a dictionary with the expected format
                    if isinstance(structured_info, dict):
                        for category in ["facts", "decisions", "preferences", "patterns"]:
                            if structured_info.get(category):
                                f.write(f"\n#### {category.title()}\n")
                                for item in structured_info[category]:
                                    f.write(f"- {item}\n")
                    f.write("\n")

            log_event(self.logger, "memory.dumped", 
                     f"Saved structured memory {metadata['memory_id']} to disk")
        except Exception as e:
            log_error(self.logger, 
                     f"Failed to save memory {metadata['memory_id']} to disk: {str(e)}")

    async def _cleanup_short_term_memories(self, days_threshold: int = 0) -> None:
        """Clean up old short-term memories after consolidation.
        
        Args:
            days_threshold: Number of days worth of memories to keep.
                           Memories older than this will be deleted.
                           Default is 0 (clean up all processed memories).
        """
        try:
            log_event(self.logger, "memory.cleanup.start", 
                     f"Starting cleanup of memories older than {days_threshold} days")
            
            # Retrieve all short-term memories
            old_memories = await self.memory.retrieve(
                query="",
                collection_names=["short_term"],
                n_results=1000
            )
            
            # Filter based on threshold
            cutoff_time = datetime.now() - timedelta(days=days_threshold)
            old_memories = [
                memory for memory in old_memories
                if datetime.fromisoformat(memory["metadata"].get("timestamp", "")) < cutoff_time
            ]
            
            # Delete old memories using the ChromaDB ID from metadata
            deleted_count = 0
            for memory in old_memories:
                chroma_id = memory["metadata"].get("chroma_id")  # Use the new chroma_id field
                conversation_id = memory["metadata"].get("conversation_id")
                self.old_conversation_list[conversation_id] = "placeholder_name"
                if chroma_id:
                    await self.memory.delete(chroma_id, "short_term")
                    deleted_count += 1
                
            log_event(self.logger, "memory.cleanup", 
                     f"Cleaned up {deleted_count} memories older than {days_threshold} days")
             
        except Exception as e:
            log_error(self.logger, "Failed to cleanup short-term memories", exc_info=e)

    #TODO: Need to fix internal tools so that the descriptions and tool definitions work properly.
    async def search_memory(self, query: str, keywords: Optional[List[str]] = None) -> Dict[str, Any]:
        """Use to search your memory using a 'query' and (optional) 'keywords'.
        
        Args:
            query: The search query string
            keywords: Optional list of keywords to help filter results
            
        Returns:
            Dict containing search results and metadata
        """
        try:
            # Create metadata filter if keywords provided
            metadata_filter = None
            if keywords:
                # Use $in operator instead of $contains for keyword matching
                # This will match if any of the keywords exactly match the content
                metadata_filter = {
                    "content": {"$in": keywords}
                }

            # Search both short and long term memory
            memories = await self.memory.retrieve(
                query=query,
                collection_names=["short_term", "long_term"],
                n_results=5,
                metadata_filter=metadata_filter
            )

            log_event(self.logger, "memory.searched", 
                     f"Memory search for '{query}' returned {len(memories)} results")

            return {
                "query": query,
                "results": memories,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            error_msg = f"Failed to search memory: {str(e)}"
            log_error(self.logger, error_msg, exc_info=e)
            return {
                "query": query,
                "error": error_msg,
                "timestamp": datetime.now().isoformat()
            }
        