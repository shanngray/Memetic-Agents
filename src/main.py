import asyncio
import os
from pathlib import Path
import sys
import signal

# Add the project root directory to the Python path
sys.path.append(str(Path(__file__).parents[1]))

from src.api_server.server import start_server, setup_agent_server
from src.log_config import setup_logger, log_event, log_error
from src.database.chroma_database import get_chroma_client

# Setup logging
logger = setup_logger(
    name="MainProcess",
    level=os.getenv("SERVER_LOG_LEVEL", "INFO"),
    console_logging=os.getenv("CONSOLE_LOGGING", "True").lower() == "true"
)

async def main():
    log_event(logger, "main.start", "Starting main application")
    
    # Create lists to track tasks and servers
    all_tasks = []
    running_servers = []
    running_agents = []
    
    try:
        # Initialize single ChromaDB client
        chroma_client = await get_chroma_client()
        log_event(logger, "main.start", "Initialized ChromaDB client")
        
        # Start the directory service
        directory_task, directory_server = await start_server()
        all_tasks.append(directory_task)
        running_servers.append(directory_server)
        
        # Wait for directory service to start
        await asyncio.sleep(2)
        
        # Get all agent creation files from the agents directory
        agents_dir = Path(__file__).parent / "agents"
        agent_files = [f for f in agents_dir.glob("create_*_agent.py")]
        
        for agent_file in agent_files:
            agent_name = agent_file.stem.split('_')[1]
            
            try:
                # Import and create the agent
                module_name = f"src.agents.{agent_file.stem}"
                function_name = f"create_{agent_name}_agent"
                
                log_event(logger, "agent.import", f"Importing agent module: {module_name}")
                module = __import__(module_name, fromlist=[function_name])
                create_func = getattr(module, function_name)
                
                log_event(logger, "agent.create", f"Loading agent:      {agent_name}")
                agent = create_func(chroma_client=chroma_client)
                
                # For memetic agents, ensure initialization is complete
                if hasattr(agent, 'initialize'):
                    log_event(logger, "agent.init", f"Initializing agent:       {agent_name}")
                    try:
                        await agent.initialize()
                    except Exception as e:
                        log_error(logger, f"Failed to initialize {agent_name} agent: {str(e)}")
                        continue
                
                # Setup agent and get tasks
                log_event(logger, "agent.setup", f"Setting up {agent_name} agent server")
                agent, tasks = await setup_agent_server(agent)
                running_agents.append(agent)
                all_tasks.extend(tasks)
                
                log_event(logger, "agent.ready", f"{agent_name} agent ready")
                
            except ImportError as e:
                log_error(logger, f"Failed to import {agent_name} agent module: {str(e)}")
                continue
            except AttributeError as e:
                log_error(logger, f"Failed to find create function for {agent_name} agent: {str(e)}")
                continue
            except Exception as e:
                log_error(logger, f"Unexpected error setting up {agent_name} agent: {str(e)}")
                continue
        
        if not running_agents:
            log_error(logger, "No agents were successfully started")
            return
            
        log_event(logger, "main.agents.ready", f"Successfully started {len(running_agents)} agents")
        
        # Wait for shutdown signal
        shutdown_event = asyncio.Event()
        try:
            await shutdown_event.wait()
        except asyncio.CancelledError:
            log_event(logger, "main.shutdown", "Received shutdown signal")
            
    except asyncio.CancelledError:
        logger.info("Shutdown initiated")
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
    finally:
        logger.info("Starting shutdown sequence")
        
        # First, shutdown all agents gracefully
        shutdown_tasks = []
        for agent in running_agents:
            shutdown_tasks.append(asyncio.create_task(agent.shutdown()))
        
        if shutdown_tasks:
            try:
                await asyncio.gather(*shutdown_tasks, return_exceptions=True)
            except Exception as e:
                logger.error(f"Error during agent shutdown: {str(e)}")
        
        # Cancel all running tasks
        for task in all_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete with timeout
        if all_tasks:
            try:
                done, pending = await asyncio.wait(all_tasks, timeout=5.0)
                for task in pending:
                    task.cancel()
                if pending:  # Only wait if there are pending tasks
                    await asyncio.wait(pending, timeout=1.0)
            except Exception as e:
                if "Set of Tasks/Futures is empty" not in str(e):  # Ignore empty set error
                    logger.error(f"Error during task shutdown: {str(e)}")
        
        # Shutdown all Uvicorn servers
        for server in running_servers:
            try:
                # Uvicorn servers should be stopped using their .should_exit property
                if hasattr(server, 'should_exit'):
                    server.should_exit = True
                    await asyncio.sleep(0.1)  # Give server time to process shutdown
                elif hasattr(server, 'shutdown'):
                    await server.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down server: {str(e)}")
        
        logger.info("Shutdown complete")

if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    main_task = None
    
    def handle_interrupt():
        logger.info("Received keyboard interrupt")
        if main_task:
            main_task.cancel()
    
    try:
        # Register signal handlers
        loop.add_signal_handler(signal.SIGINT, handle_interrupt)
        loop.add_signal_handler(signal.SIGTERM, handle_interrupt)
        
        main_task = loop.create_task(main())
        loop.run_until_complete(main_task)
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, initiating shutdown...")
        if main_task:
            main_task.cancel()
            try:
                loop.run_until_complete(main_task)
            except asyncio.CancelledError:
                pass
    finally:
        try:
            tasks = asyncio.all_tasks(loop)
            for task in tasks:
                task.cancel()
            
            loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()
            
            logger.info("Clean shutdown completed")
        except Exception as e:
            logger.error(f"Error during final cleanup: {str(e)}")
        sys.exit(0)
