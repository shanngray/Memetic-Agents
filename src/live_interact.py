import asyncio
import httpx
import re
import uuid
import logging
from datetime import datetime
from pathlib import Path
from api_server.models.api_models import APIMessage
from base_agent.models import AgentStatus
from icecream import ic
import os
from conversation_store import ConversationStore
from log_config import setup_logger, log_event

# Setup logging with environment variable
server_log_level = os.getenv("LIVE_INTERACT_LOG_LEVEL", "INFO")

# Initialize logger
logger = setup_logger(
    name="LiveInteract",
    log_path=Path("logs"),
    level=os.getenv("LIVE_INTERACT_LOG_LEVEL", "INFO"),
    console_logging=True
)

#TODO: The conversation store never prunes. old conversations need to be deleted. 
# Initialize conversation store
CONVERSATION_STORE = ConversationStore(Path("agent_files/conversations.json"))

async def handle_lookup():
    """Query the directory service for all registered agents."""
    async with httpx.AsyncClient() as client:
        try:
            log_event(logger, "directory.lookup", "Querying directory service for registered agents", level="DEBUG")
            response = await client.get("http://localhost:8000/agent/lookup")
            agents_dict = response.json()
            print("\nRegistered Agents:")
            for name, agent_info in agents_dict.items():
                print(f"\n- {name} (Port: {agent_info['port']})")
                print(f"  Description: {agent_info['description']}")
                print(f"  Tools: {', '.join(agent_info['tools'])}")
            log_event(logger, "directory.lookup", f"Found {len(agents_dict)} registered agents", level="DEBUG")
        except Exception as e:
            log_event(logger, "directory.lookup", f"Error looking up agents: {str(e)}", level="ERROR")
            print(f"\nError looking up agents: {str(e)}")

async def handle_direct_message(agent_name: str, message: str, conversation_id: str):
    """Send a message directly to a specific agent."""
    if not conversation_id:
        log_event(logger, "message.content", "No active conversation for direct message", level="DEBUG")
        print("\nNo active conversation. Use /new to start one or /resume <id> to continue an existing one.")
        return
        
    log_event(logger, "message.content", f"Sending message to {agent_name} in conversation {conversation_id[:8]}...", level="DEBUG")
    
    # Update conversation participants if needed
    conversation = CONVERSATION_STORE.get_conversation(conversation_id)
    if conversation and agent_name not in conversation["participants"]:
        participants = conversation["participants"] + [agent_name]
        CONVERSATION_STORE.update_conversation(conversation_id, participants=participants)
        log_event(logger, "directory.agent_registered", f"Added {agent_name} to conversation {conversation_id[:8]}", level="DEBUG")
    
    api_message = APIMessage(
        sender="User",
        receiver=agent_name,
        content=message,
        conversation_id=conversation_id,
        timestamp=datetime.now().isoformat()
    )
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"http://localhost:8000/agent/message",
                json=api_message.dict(),
                timeout=6000.0  # Needs to be longer than server timeout as there may be multiple server hops
            )
            response.raise_for_status()
            response_data = response.json()
            
            # Handle AgentResponse structure
            if isinstance(response_data, dict):
                if response_data.get('success'):
                    message_content = response_data.get('message', '')
                    if isinstance(message_content, dict):
                        message_content = message_content.get('message', '')
                    print(f"\n{agent_name}: {message_content}")
                else:
                    print(f"\nError: {response_data.get('message', 'Unknown error')}")
            else:
                print(f"\n{agent_name}: {response_data}")
                    
        except httpx.ReadTimeout:
            print(f"\nError: Request to {agent_name} timed out after 300 seconds")
        except Exception as e:
            print(f"\nError: {str(e)}")

async def handle_status_update(agent_name: str, new_status: str, conversation_id: str):
    """Update an agent's status."""
    # Format conversation ID for logging, handling None case
    conv_id_str = f"{conversation_id[:8]}" if conversation_id else "no active conversation"
    log_event(logger, "status.update.request", 
             f"Updating status for {agent_name} to {new_status} for {conv_id_str}", level="DEBUG")
    try:
        # Allow both name and value input
        try:
            status_int = int(new_status)
            status = AgentStatus(status_int)
        except ValueError:
            status = AgentStatus[new_status.upper()]
            
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"http://localhost:8000/agent/{agent_name}/status",
                params={"status": str(status.value)}
            )
            # Log the raw response for debugging
            log_event(logger, "status.update.debug", 
                     f"Raw response: Status {response.status_code}, Content: {response.text}", level="DEBUG")
            
            response.raise_for_status()
            result = response.json()
            log_event(logger, "status.update.success", 
                     f"Status updated for {agent_name}: /{result['previous_status']} → /{result['current_status']}", level="DEBUG")
            print(f"\nStatus updated: /{result['previous_status']} → /{result['current_status']}")
    except KeyError:
        valid_statuses = [s.name.lower() for s in AgentStatus]
        error_msg = f"Invalid status '{new_status}'. Must be one of: {', '.join(valid_statuses)}"
        log_event(logger, "status.update.error", error_msg, level="ERROR")
        print(f"\nError: {error_msg}")
        return
    except httpx.HTTPError as e:
        error_msg = f"HTTP error occurred: {str(e)}"
        if hasattr(e, 'response'):
            error_msg += f"\nResponse status: {e.response.status_code}"
            error_msg += f"\nResponse content: {e.response.text}"
        log_event(logger, "status.update.error", error_msg, level="ERROR")
        print(f"\nError from server while updating status of agent {agent_name} to {new_status}.")
        print(f"Error details: {error_msg}")
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}, Type: {type(e).__name__}"
        log_event(logger, "status.update.error", error_msg, level="ERROR")
        print(f"\nError from server while updating status of agent {agent_name} to {new_status}.")
        print(f"Error details: {error_msg}")

async def handle_show_collection(collection_name: str):
    """Show documents in the specified collection."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"http://localhost:8000/collections/{collection_name}")
            documents = response.json()
            print(f"\nDocuments in {collection_name}:")
            for doc in documents:
                # Handle both dictionary and string document formats
                if isinstance(doc, dict):
                    print(f"\n- ID: {doc.get('_id', 'N/A')}")
                    for key, value in doc.items():
                        if key != '_id':
                            print(f"  {key}: {value}")
                else:
                    print(f"\n- {doc}")
        except Exception as e:
            print(f"\nError showing collection: {str(e)}")

async def handle_list_collections():
    """List all available collections."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get("http://localhost:8000/collections")
            collections = response.json()
            print("\nAvailable Collections:")
            for collection in collections:
                print(f"- {collection}")
        except Exception as e:
            print(f"\nError listing collections: {str(e)}")

async def handle_list_status():
    """List status of all agents."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get("http://localhost:8000/agent/status/all")
            statuses = response.json()
            print("\nAgent Statuses:")
            for agent, status_info in statuses.items():
                # Get the status value and convert to enum name
                status_value = status_info.get('status', 'UNKNOWN')
                try:
                    # Convert integer status to enum name
                    status_name = AgentStatus(status_value).name
                    category = status_info.get('category', 'unknown')
                    print(f"- {agent}: {status_name} ({category})")
                    
                    # If there's an error, show it
                    if 'error' in status_info:
                        print(f"  Error: {status_info['error']}")
                except ValueError:
                    # If we can't convert to enum, show raw status
                    print(f"- {agent}: {status_value}")
                    
        except Exception as e:
            print(f"\nError listing statuses: {str(e)}")

def handle_clear():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

async def handle_new_conversation(new_name="New Conversation"):
    """Start a new conversation."""
    conversation_id = str(uuid.uuid4())
    CONVERSATION_STORE.add_conversation(
        conversation_id=conversation_id,
        name=new_name,
        participants=["User"]  # Initial participant
    )
    log_event(logger, "social.conversation.new", 
              f"Created new conversation {conversation_id[:8]} with name: {new_name}", level="DEBUG")
    return conversation_id

async def handle_list_conversations():
    """List all available conversations."""
    conversations = CONVERSATION_STORE.list_conversations()
    if not conversations:
        print("\nNo conversations found.")
        return
    
    print("\nAvailable Conversations:")
    for conv in conversations:
        print(f"\n- ID: {conv['id']}")
        print(f"  Name: {conv['name']}")
        print(f"  Participants: {', '.join(conv['participants'])}")
        print(f"  Messages: {conv['message_count']}")
        print(f"  Last Active: {conv['last_active']}")

async def handle_rename_conversation(conversation_id: str, new_name: str):
    """Rename an existing conversation."""
    CONVERSATION_STORE.update_conversation(conversation_id, name=new_name)
    print(f"\nRenamed conversation to: {new_name}")

async def live_interact():
    log_event(logger, "server.startup", "Starting live interaction interface", level="DEBUG")
    EXIT_COMMANDS = {"exit", "quit", "bye", "goodbye", "/exit", "/quit"}
    CONVERSATION_COMMANDS = {"/new", "/list", "/resume", "/exit_conv", "/rename"}
    
    print("Starting conversation with Agentic Community (type 'exit' to quit)\n")
    print("Available commands:")
    print("  /new <name> - Start a new conversation")
    print("  /list - List available conversations")
    print("  /resume <id> - Resume existing conversation")
    print("  /exit_conv - Exit current conversation")
    print("  /rename <name> - Rename current conversation")
    print("  /lookup - List all available agents")
    print("  @agent_name message - Send direct message to agent")
    print("  @agent_name/status/new_status - Update agent status")
    print("  /show <collection_name> - Show documents in collection")
    print("  /list_col - List all collections")
    print("  /list_status - List all agent statuses")
    print("  /clear - Clear the screen")
    print("  /help - Show this help message\n")

    current_conversation_id = None
    
    while True:
        # Show conversation context in prompt if in a conversation
        prompt = f"\nConversation {current_conversation_id[:8]}... > " if current_conversation_id else "\nNo active conversation > "
        user_input = input(prompt).strip()
        
        # Parse input for different command patterns
        status_pattern = re.match(r"@(\w+)/status/(\w+)", user_input)
        direct_msg_pattern = re.match(r"@(\w+)\s+(.+)", user_input)
        rename_pattern = re.match(r"/rename\s+(.+)", user_input)
        resume_pattern = re.match(r"/resume\s+(.+)", user_input)
        new_pattern = re.match(r"/new(?:\s+(.+))?", user_input)
        
        match user_input:
            case cmd if cmd.lower() in EXIT_COMMANDS:
                log_event(logger, "server.shutdown", "Ending live interaction session")
                print("Ending conversation. Goodbye!")
                break
                
            case cmd if new_pattern:
                new_name = new_pattern.group(1)  # Will be None if no name provided
                current_conversation_id = await handle_new_conversation(new_name)
                print(f"\nStarted new conversation with ID: {current_conversation_id}")
                
            case "/list":
                await handle_list_conversations()
                
            case cmd if resume_pattern:
                conv_id = resume_pattern.group(1)
                if CONVERSATION_STORE.get_conversation(conv_id):
                    current_conversation_id = conv_id
                    print(f"\nResumed conversation: {conv_id}")
                else:
                    print(f"\nConversation not found: {conv_id}")
                
            case "/exit_conv":
                if current_conversation_id:
                    current_conversation_id = None
                    print("\nExited conversation")
                else:
                    print("\nNo active conversation")
                
            case cmd if rename_pattern:
                if current_conversation_id:
                    new_name = rename_pattern.group(1)
                    await handle_rename_conversation(current_conversation_id, new_name)
                else:
                    print("\nNo active conversation to rename")
                
            case "/lookup":
                await handle_lookup()
                
            case cmd if status_pattern:
                agent_name, new_status = status_pattern.groups()
                await handle_status_update(agent_name, new_status, current_conversation_id)
                
            case cmd if direct_msg_pattern:
                agent_name, message = direct_msg_pattern.groups()
                await handle_direct_message(agent_name, message, current_conversation_id)
                
            case cmd if cmd.startswith("/show "):
                collection_name = cmd.split()[1]
                await handle_show_collection(collection_name)
                
            case "/list_col":
                await handle_list_collections()
                
            case "/list_status":
                await handle_list_status()
                
            case "/clear":
                handle_clear()
                
            case "/help":
                print("\nAvailable commands:")
                print("  /new - Start a new conversation")
                print("  /list - List available conversations")
                print("  /resume <id> - Resume existing conversation")
                print("  /exit_conv - Exit current conversation")
                print("  /rename <name> - Rename current conversation")
                print("  /lookup - List all available agents")
                print("  @agent_name message - Send direct message to agent")
                print("  @agent_name/status/new_status - Update agent status")
                print("  /show <collection_name> - Show documents in collection")
                print("  /list_col - List all collections")
                print("  /list_status - List all agent statuses")
                print("  /clear - Clear the screen")
                print("  /help - Show this help message")
                
            case _:
                if current_conversation_id is None:
                    print("\nNo active conversation. Use /new to start one or /resume <id> to continue an existing one.")
                else:
                    #TODO: need to change this to just continue the conversation if there is an id
                    # Update conversation state before sending message
                    CONVERSATION_STORE.increment_message_count(current_conversation_id)
                    # Send to default agent
                    await handle_direct_message("Aithor", user_input, current_conversation_id)

if __name__ == "__main__":
    asyncio.run(live_interact())
