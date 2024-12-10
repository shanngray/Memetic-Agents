import asyncio
import os
import sys
from pathlib import Path

# Add the project root directory to the Python path
sys.path.append(str(Path(__file__).parents[2]))

from api_server.server import setup_agent, start_server
from src.log_config import setup_logger

# Setup logging with environment variable
logger = setup_logger(
    name="ServerRun",
    level=os.getenv("SERVER_LOG_LEVEL", "INFO"),
    console_logging=os.getenv("CONSOLE_LOGGING", "True").lower() == "true"
)

# TODO: Currently the agents aren't really having a conversation, they are just sending messages to each other
#       Need to give the agents each others' addresses so that they can send messages to each other 
#       need to create agents that need to ustilise each others tools to solve a problem.
#       Need to work out where debugging options are
#       Need to make conversation more readable
#       Need to add a tool to create agents
#       Need to make conversation history persistent
#       Need to add memories
#       Need to add franken_agent code back in

async def test_concurrent_communication(math_agent, time_agent):
    """Test concurrent communication between agents."""
    logger.info("Starting concurrent communication test")
    
    # Create multiple concurrent requests
    tasks = [
        # Math agent asks time agent for current time
        math_agent.process_message(
            "I need to know the current time in New York and London. Can you find an agent to help?", 
            "User",
            "New"
        ),
        # Time agent asks math agent to calculate time difference
        time_agent.process_message(
            "I need to calculate the time difference between UTC+1 and UTC-5. Can you help me find a math agent?",
            "User",
            "New"
        ),
        # Math agent requests another calculation while waiting
        math_agent.process_message(
            "While we wait for the time information, can you calculate 15 * 24?",
            "User",
            "New"
        )
    ]
    
    # Wait for all tasks to complete
    responses = await asyncio.gather(*tasks)
    
    # Log responses
    for i, response in enumerate(responses):
        logger.info(f"Response {i + 1}: {response}")
    
    return responses

async def main():
    logger.info("Starting main application")
    # Start the directory service
    directory_task, app = await start_server()
    
    # Wait a moment for the service to start
    await asyncio.sleep(2)
    
    server_tasks = []
    try:
        # Create math agent with lookup_agent tool
        logger.info("Creating math agent")
        math_agent, math_tasks = await setup_agent(
            name="MathAgent",
            port=8010,
            system_prompt="""You are a mathematical expert that can perform calculations and solve math problems.
            When you need help with non-mathematical queries, you can use the lookup_agent tool to find other agents 
            that specialize in those areas. After finding a suitable agent, you can use the send_message tool to 
            communicate with them. Always explain your reasoning and process.""",
            tools=["lookup_agent", "calc", "send_message"],
            description="Mathematical expert that can perform calculations and solve math problems"
        )
        server_tasks.extend(math_tasks)
        
        # Create time agent
        logger.info("Creating time agent")
        time_agent, time_tasks = await setup_agent(
            name="TimeAgent",
            port=8011,
            system_prompt="""You are a time and timezone expert. Help users with time-related queries using 
            the get_current_time tool. You can also communicate with other agents using the send_message tool 
            when needed. Explain time differences and conversions clearly.""",
            tools=["get_current_time", "send_message", "lookup_agent"],
            description="Time and timezone expert that can provide current time in different timezones"
        )
        server_tasks.extend(time_tasks)
        
        # Wait for a moment to ensure all servers are running
        await asyncio.sleep(2)
        
        # Run concurrent communication test
        logger.info("Starting concurrent communication test")
        test_responses = await test_concurrent_communication(math_agent, time_agent)
        
        # Log test results
        logger.info("Concurrent communication test completed")
        logger.info("Test responses:")
        for i, response in enumerate(test_responses):
            logger.info(f"Test {i + 1} response: {response}")
            
    except Exception as e:
        logger.error(f"Error during agent setup: {str(e)}")
        print(f"Error during agent setup: {e}")
    finally:
        # Gracefully shutdown all servers
        logger.info("Shutting down servers")
        try:
            for task in server_tasks:
                task.cancel()
            directory_task.cancel()
            
            # Wait for all tasks to complete with a timeout
            await asyncio.wait(
                [directory_task] + server_tasks, 
                timeout=5.0,
                return_when=asyncio.ALL_COMPLETED
            )
        except Exception as e:
            logger.error(f"Error during shutdown: {str(e)}")
        finally:
            logger.info("All servers shutdown complete")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
