# Memetic Agents: A Framework for AI Agents That Learn Through Social Interaction

Memetic Agents is an innovative framework for creating AI agents capable of social learning and knowledge evolution. The framework implements a multi-tiered memory architecture (working, short-term, long-term), dynamic prompt modification, and inter-agent communication protocols. The system enables agents to maintain persistent knowledge, engage in meaningful interactions, and evolve their capabilities through social learning mechanisms inspired by memetic theory. Agents can reflect on their experiences, consolidate memories, and share knowledge with other agents, creating a network of continuously improving AI entities.

## Features

- **Multi-Agent Architecture**: Multiple specialized agents working together
- **Directory Service**: Central service for agent discovery and message routing
- **Persistent Memory**: ChromaDB-based memory system for agents
- **Interactive Interface**: Command-line interface for direct agent interaction
- **Tool Integration**: Extensible tool system for agent capabilities
- **Health Monitoring**: Built-in health checks and status monitoring
- **Logging System**: Comprehensive logging with configurable levels

## Prerequisites

- Python 3.x
- Poetry (Python dependency management)
- Required API keys:
  - OpenAI API key
  - Tavily API key (for web search capabilities)

## Setup

1. Clone this repository

2. Install dependencies using Poetry:
   ```bash
   poetry install
   ```

3. Create a `.env` file based on `.env.sample`:
   ```
   OPENAI_API_KEY=your_openai_key_here
   TAVILY_API_KEY=your_tavily_key_here
   ```

4. Create a `agent_files` directory:
   ```bash
   mkdir agent_files
   ```

## Creating New Agents

The system provides an interactive agent factory to create new agents. To create a new agent:

1. Run the agent factory script:
   ```bash
   poetry run python src/utils/new_agent_factory.py
   ```

2. Follow the interactive prompts to configure your agent:
   - Agent name
   - Description of agent's capabilities
   - Model selection (default: gpt-4-mini)
   - Submodel selection (default: gpt-4o-mini)
   - Prompt template selection (default: placeholder)
   - Console logging preference
   - API port number (must be unique)
   - Temperature setting (0.0-2.0)
   - Enabled tools selection

The factory will:

- Create a folder structure in `agent_files/<agent_name>/`
- Set up prompt modules and configuration files
- Generate an agent creation script in `src/agents/`

Example agent creation:

```bash
$ poetry run python src/utils/new_agent_factory.py
? What is the name of your agent? ResearchAgent
? Provide a brief description of your agent: Specialized in academic research and citation
? Which model should the agent use? gpt-4-mini
? Which submodel should the agent use? gpt-4o-mini
? Which prompt templates should the agent use? placeholder
? Enable console logging? True
? Which API port should the agent use? 8085
? Set the temperature (0.0-2.0) 0.7
? Select enabled tools (use space to select) ◉ agent_search ◉ list_agents ◉ web_search
```

## Running the System

The system requires two terminals to run:

1. Start the main system (directory service and agents):

   ```bash
   poetry run python src/main.py
   ```

2. Launch the interaction interface:

   ```bash
   poetry run python src/live_interact.py
   ```

## Interactive Commands

- `/lookup` - List all available agents
- `@agent_name message` - Send direct message to agent
- `@agent_name/status/new_status` - Update agent status
- `/show <collection_name>` - Show documents in collection
- `/list_col` - List all collections
- `/list_status` - List all agent statuses
- `/clear` - Clear the screen
- `/help` - Show help message

## API Documentation

- API specifications and OpenAPI documentation available at:
  - Directory Service: `http://127.0.0.1:8000/docs`
  - Individual Agent APIs: Available at their respective ports

## Memory Management

The system uses ChromaDB for persistent storage with two types of memory:

- Short-term memory: Recent interactions and temporary data
- Long-term memory: Consolidated and processed information
- Feedback memory: Feedback from other agents
- Reflections: Internal thoughts and evaluations

## Development

### Environment Variables

- `AGENT_DEBUG`: Enable debug mode (default: false)
- `AGENT_LOG_LEVEL`: Set logging level (default: INFO)
- `SERVER_LOG_LEVEL`: Set server logging level (default: INFO)
- `CONSOLE_LOGGING`: Enable console logging (default: True)

### Adding New Agents

Create new agent files in `src/agents/` following the existing patterns:

'''python
def create_new_agent(chroma_client: PersistentClient) -> BaseAgent:
config = AgentConfig(
agent_name="NewAgent",
description="Description of the agent's capabilities",
enabled_tools=["list_of", "enabled", "tools"],
api_port=unique_port_number
)
return BaseAgent(api_key=os.getenv("OPENAI_API_KEY"),
chroma_client=chroma_client,
config=config)
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
