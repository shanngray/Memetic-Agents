import asyncio
import logging
from datetime import datetime
from pathlib import Path
import uvicorn
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import httpx
import os
from base_agent.base_agent import BaseAgent
from base_agent.config import AgentConfig
from base_agent.models import AgentStatus
from api_server.services.agent_directory import AgentDirectory, AgentDirectoryService
from api_server.models.api_models import APIMessage, AgentResponse, FeedbackMessage
from src.log_config import setup_logger, log_event
from typing import Tuple, List, Optional
import uuid
import traceback
from database.chroma_database import get_chroma_client
from pydantic import BaseModel, Field

# Setup logging
log_path = Path("logs")
log_path.mkdir(exist_ok=True)
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_path / f'server_debug_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Setup logging with environment variable
server_log_level = os.getenv("SERVER_LOG_LEVEL", "INFO")
logger = setup_logger(
    name="DirectoryService",
    log_path=Path("logs"),
    level=os.getenv("SERVER_LOG_LEVEL", "INFO"),
    console_logging=True
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application lifespan events."""
    log_event(logger, "directory.startup", "Starting directory service")
    yield
    log_event(logger, "directory.shutdown", "Shutting down directory service")

async def start_directory_service():
    """
    Initialize and start the agent directory service.
    
    Creates a FastAPI application with the following endpoints:
    - /agent/register: Register new agents
    - /agent/lookup: Look up registered agents
    - /agent/message: Route messages between agents
    - /agent/feedback: Route feedback between agents
    - /health: Service health check
    - /collections: Manage vector store collections
    
    Returns:
        Tuple[asyncio.Task, FastAPI]: Server task and FastAPI application instance
    """
    app = FastAPI(lifespan=lifespan)
    directory_service = AgentDirectoryService()
    
    # Get log level from environment or default to INFO
    server_log_level = os.getenv("SERVER_LOG_LEVEL", "INFO").upper()
    
    # Register routes
    @app.post("/agent/register")
    async def register_agent(agent: AgentDirectory):
        """
        Register a new agent with the directory service.
        
        Args:
            agent: AgentDirectory instance containing registration details
            
        Returns:
            AgentResponse: Success confirmation with registration message
            
        Raises:
            HTTPException: If registration fails
        """
        log_event(logger, "directory.agent_registered", f"Registering agent: {agent.name}")
        directory_service.register_agent(agent)
        return AgentResponse(success=True, message=f"Agent {agent.name} registered")

    @app.get("/agent/lookup")
    async def lookup_agents(agent_name: str = None):
        """Lookup registered agents."""
        log_event(logger, "directory.agent_lookup", f"Looking up agents. Specific agent: {agent_name}")
        return directory_service.lookup_agents(agent_name)

    @app.post("/agent/message")
    async def route_message(message: APIMessage):
        log_event(
            logger, 
            "directory.message_route", 
            f"Routing message: {message.sender} → {message.receiver} ({message.conversation_id})"
        )
        return await directory_service.route_message(message)

    @app.get("/health")
    async def health_check():
        log_event(logger, "health.check", "Health check requested")
        return {"status": "healthy"}

    @app.get("/collections")
    async def list_collections():
        """List all collections and their document counts."""
        try:
            chroma_client = await get_chroma_client()
            collections = chroma_client.list_collections()
            result = []
            for collection in collections:
                count = collection.count()
                result.append({
                    "name": collection.name,
                    "document_count": count
                })
            return result
        except Exception as e:
            log_event(logger, "collections.list.error", f"Error listing collections: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/collections/{collection_name}")
    async def get_collection_documents(collection_name: str):
        """Get all documents and metadata from a specific collection."""
        try:
            chroma_client = await get_chroma_client()
            collection = chroma_client.get_collection(collection_name)
            if collection is None:
                raise HTTPException(status_code=404, detail=f"Collection {collection_name} not found")
            
            result = collection.get()
            return {
                "collection_name": collection_name,
                "documents": result["documents"] if "documents" in result else [],
                "metadatas": result["metadatas"] if "metadatas" in result else [],
                "ids": result["ids"] if "ids" in result else []
            }
        except Exception as e:
            log_event(logger, "collections.get.error", f"Error getting collection {collection_name}: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/agent/status/all")
    async def get_all_agent_statuses():
        """
        Retrieve current status of all registered agents.
        
        Performs health checks on each registered agent and returns their status.
        Possible status values:
        - healthy: Agent is responsive and operational
        - unreachable: Agent cannot be contacted
        - error: Agent returned an error state
        
        Returns:
            Dict[str, Dict]: Status information for each agent
            
        Raises:
            HTTPException: If status collection fails
        """
        try:
            agents_dict = directory_service.lookup_agents()
            statuses = {}
            for agent_name, agent_info in agents_dict.items():
                try:
                    async with httpx.AsyncClient() as client:
                        response = await client.get(
                            f"http://{agent_info['address']}:{agent_info['port']}/health"
                        )
                        if response.status_code == 200:
                            statuses[agent_name] = response.json()
                        else:
                            statuses[agent_name] = {
                                "status": "unreachable", 
                                "error": response.text
                            }
                except Exception as e:
                    statuses[agent_name] = {
                        "status": "error", 
                        "error": str(e)
                    }
            
            return statuses
        except Exception as e:
            log_event(logger, "agent.status.all.error", f"Error getting agent statuses: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/agent/feedback")
    async def route_feedback(feedback: FeedbackMessage):
        """Route feedback from one agent to another."""
        log_event(
            logger, 
            "directory.feedback_route", 
            f"Routing feedback: {feedback.sender} → {feedback.receiver}"
        )
        
        for attempt in range(3):  # Add retry logic
            try:
                # Get receiver info from directory
                receiver_info = directory_service.lookup_agents(feedback.receiver)
                if not receiver_info:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Receiver agent {feedback.receiver} not found"
                    )
                
                receiver = receiver_info[feedback.receiver]
                
                # Send feedback to receiver's feedback endpoint
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"http://{receiver['address']}:{receiver['port']}/feedback",
                        json=feedback.dict(),
                        timeout=30.0
                    )
                    response.raise_for_status()
                
                return {"success": True, "message": "Feedback delivered"}
                
            except Exception as e:
                if attempt == 2:  # Last attempt
                    raise
                await asyncio.sleep(1)  # Wait before retry

    @app.post("/agent/{agent_name}/status")
    async def route_status_update(agent_name: str, status: str):
        """Route status update to the appropriate agent."""
        try:
            log_event(logger, "status.update.request", 
                     f"Received status update request for {agent_name}: {status}")
            
            # Get agent info from directory
            agents_dict = directory_service.lookup_agents(agent_name)
            if not agents_dict:
                log_event(logger, "status.update.error", 
                         f"Agent {agent_name} not found", level="ERROR")
                raise HTTPException(status_code=404, detail=f"Agent {agent_name} not found")
            
            agent_info = agents_dict[agent_name]
            
            # Forward status update to agent's status endpoint with query parameters
            async with httpx.AsyncClient() as client:
                log_event(logger, "status.update.forward", 
                         f"Forwarding status update to {agent_name} at {agent_info['address']}:{agent_info['port']}")
                response = await client.post(
                    f"http://{agent_info['address']}:{agent_info['port']}/status",
                    params={"status": status}  # Changed from json to params
                )
                response.raise_for_status()
                return response.json()
                
        except HTTPException:
            raise
        except Exception as e:
            log_event(logger, "status.update.error", 
                     f"Error updating agent status: {str(e)}\n{traceback.format_exc()}", 
                     level="ERROR")
            raise HTTPException(status_code=500, detail=str(e))

    # Start the directory service server
    config = uvicorn.Config(
        app=app,
        host="localhost",
        port=8000,
        loop="asyncio",
        log_level=server_log_level.lower(),  # Uvicorn expects lowercase
        log_config={
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s - %(levelname)s - %(message)s",
                }
            },
            "handlers": {
                "default": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                },
            },
            "loggers": {
                "uvicorn": {"handlers": ["default"], "level": server_log_level},
                "uvicorn.access": {"handlers": ["default"], "level": server_log_level},
                "uvicorn.error": {"handlers": ["default"], "level": server_log_level},
            },
        },
    )
    server = uvicorn.Server(config)
    server_task = asyncio.create_task(server.serve())
    
    return server_task, app

async def setup_agent_server(agent: BaseAgent) -> asyncio.Task:
    """
    Configure and start an individual agent's HTTP server.
    
    Sets up the following for each agent:
    - FastAPI application with agent-specific endpoints
    - Logging configuration
    - Processing loop
    - Directory service registration
    
    Args:
        agent: BaseAgent instance to set up
        
    Returns:
        Tuple[BaseAgent, List[asyncio.Task]]: Configured agent and its associated tasks
        
    Raises:
        Exception: If agent registration fails
    """
    agent_logger = setup_logger(
        name=agent.config.agent_name,
        log_path=agent.config.log_path,
        level=agent.config.log_level,
        console_logging=agent.config.console_logging
    )
    agent_logger.info(f"Setting up agent: {agent.config.agent_name}")
    
    # Create FastAPI app for the agent
    agent_app = FastAPI()
    
    # Get log level from environment or default to INFO
    server_log_level = os.getenv("SERVER_LOG_LEVEL", "INFO").upper()
    
    # Start the agent's processing loop
    processing_task = asyncio.create_task(agent.start())
    server_task = asyncio.create_task(start_agent_server(agent, agent.config.api_port))

    # Register agent with directory service
    try:
        async with httpx.AsyncClient() as client:
            agent_logger.info(f"Registering {agent.config.agent_name} with directory service")
            agent_directory = AgentDirectory(
                name=agent.config.agent_name,
                address="localhost",
                port=agent.config.api_port,
                agent_type="specialized",
                status="active",
                description=agent.config.description,
                tools=agent.config.enabled_tools,
                agent_instance=agent
            )
            await client.post(
                "http://localhost:8000/agent/register",
                json=agent_directory.dict(exclude={'agent_instance'})
            )
            
    except Exception as e:
        agent_logger.error(f"Failed to register agent {agent.config.agent_name}: {str(e)}")
        server_task.cancel()
        raise
    
    return agent, [processing_task, server_task]

# NO LONGER IN USE
async def setup_agent(name: str, port: int, system_prompt: str, tools: list, description: str) -> Tuple[BaseAgent, List[asyncio.Task]]:
    """Setup and register an agent with the directory service.
    
    Args:
        name: Agent name
        port: Port number
        system_prompt: System prompt
        tools: List of tool names
        description: Short description of agent capabilities
    """
    config = AgentConfig(
        debug=os.getenv("AGENT_DEBUG", "false").lower() == "true",
        console_logging=True,
        model="gpt-4o-mini",
        system_prompt=system_prompt,
        agent_name=name,
        enabled_tools=tools,
        api_port=port,
        log_path=Path("logs"),
        log_level=os.getenv("AGENT_LOG_LEVEL", "DEBUG")
    )
    
    agent_logger = setup_logger(
        name=name,
        log_path=config.log_path,
        level=config.log_level,
        console_logging=config.console_logging
    )
    agent_logger.info(f"Setting up agent: {name}")
    
    # Create FastAPI app for the agent
    agent_app = FastAPI()
    
    agent = BaseAgent(
        api_key=os.getenv("OPENAI_API_KEY"),
        config=config
    )

    # Get log level from environment or default to INFO
    server_log_level = os.getenv("SERVER_LOG_LEVEL", "INFO").upper()
    
    # Start the agent's processing loop
    processing_task = asyncio.create_task(agent.start())
    server_task = asyncio.create_task(start_agent_server(agent, port))
    
    # Register agent with directory service
    try:
        async with httpx.AsyncClient() as client:
            agent_logger.info(f"Registering {name} with directory service")
            agent_directory = AgentDirectory(
                name=name,
                address="localhost",
                port=port,
                agent_type="specialized",
                status="active",
                description=description,
                tools=tools,
                agent_instance=agent
            )
            await client.post(
                "http://localhost:8000/agent/register",
                json=agent_directory.dict(exclude={'agent_instance'})
            )
            
    except Exception as e:
        agent_logger.error(f"Failed to register agent {name}: {str(e)}")
        server_task.cancel()
        raise
    
    return agent, [processing_task, server_task]

async def start_agent_server(agent: BaseAgent, port: int) -> asyncio.Task:
    """
    Start the HTTP server for an individual agent.
    
    Configures the following endpoints:
    - /receive: Handle incoming messages
    - /feedback: Handle received feedback
    - /status: Update agent status
    - /health: Agent health check
    
    Args:
        agent: BaseAgent instance to serve
        port: Port number to listen on
        
    Returns:
        asyncio.Task: Server task
        
    Configuration:
        - Customizable timeouts
        - Lifespan management
        - Comprehensive logging
        - Error handling with HTTP status codes
    """
    agent_app = FastAPI()
    
    # Add lifespan management
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Lifespan manager for the agent server."""
        agent.logger.info(f"Starting {agent.config.agent_name} server")
        yield
        agent.logger.info(f"Shutting down {agent.config.agent_name} server")

    agent_app.router.lifespan_context = lifespan
    
    # Add routes for the agent's API
    @agent_app.post("/receive")
    async def receive_endpoint(message: APIMessage):
        """Handle incoming messages for the agent."""
        try:
            response = await agent.receive_message(
                sender=message.sender,
                content=message.content,
                conversation_id=message.conversation_id
            )
            return AgentResponse(success=True, message=response)
        except asyncio.TimeoutError:
            raise HTTPException(status_code=408, detail="Request timeout")
        except Exception as e:
            agent.logger.error(f"Error processing message: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @agent_app.post("/feedback")
    async def receive_feedback(feedback: FeedbackMessage):
        """Handle received feedback from another agent."""
        try:
            await agent.receive_feedback(
                sender=feedback.sender,
                conversation_id=feedback.conversation_id,
                score=feedback.score,
                feedback=feedback.feedback
            )
            return {"success": True, "message": "Feedback received"}
        except Exception as e:
            agent.logger.error(f"Error processing feedback: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @agent_app.post("/status")
    async def update_status(status: str):
        """Handle incoming commands to update agent's status."""
        try:
            agent_status = AgentStatus(status.lower())
            # Check if transition is valid
            valid_transitions = AgentStatus.get_valid_transitions(agent.status)
            if agent_status not in valid_transitions:
                error_msg = f"Invalid status transition from {agent.status} to {status}"
                agent.logger.error(error_msg)
                raise HTTPException(status_code=400, detail=error_msg)
            
            # Update the status
            previous_status = agent.status
            await agent.set_status(agent_status)
            
            agent.logger.info(f"Status updated successfully: {previous_status} → {agent_status}")
            return {
                "success": True,
                "previous_status": previous_status.value,
                "current_status": agent_status.value
            }
            
        except ValueError as e:
            error_msg = f"Status update failed: {str(e)}"
            agent.logger.error(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)
        except Exception as e:
            error_msg = f"Unexpected error updating status: {str(e)}"
            agent.logger.error(error_msg, exc_info=True)
            raise HTTPException(status_code=500, detail=error_msg)

    # Add health check endpoint for agent server
    @agent_app.get("/health")
    async def health_check():
        """Check if agent is running and responsive."""
        try:
            return {
                "status": "healthy",
                "agent": agent.config.agent_name,
                "state": agent.status.value
            }
        except Exception as e:
            agent.logger.error(f"Health check failed: {str(e)}")
            raise HTTPException(status_code=503, detail="Agent unhealthy")

    # Configure and start the server
    config = uvicorn.Config(
        app=agent_app,
        host="localhost",
        port=port,
        loop="asyncio",
        log_level=agent.config.log_level.lower(),
        lifespan="on",
        timeout_keep_alive=3000,
        timeout_notify=3000,
        log_config={
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s - %(levelname)s - %(message)s",
                }
            },
            "handlers": {
                "default": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                },
            },
            "loggers": {
                "uvicorn": {"handlers": ["default"], "level": agent.config.log_level},
                "uvicorn.access": {"handlers": ["default"], "level": agent.config.log_level},
                "uvicorn.error": {"handlers": ["default"], "level": agent.config.log_level},
            },
        },
    )
    
    server = uvicorn.Server(config)
    return asyncio.create_task(server.serve())

# Start the directory service
async def start_server():
    directory_task, app = await start_directory_service()
    return directory_task, app

class StatusUpdate(BaseModel):
    status: str = Field(..., description="New status for the agent")


