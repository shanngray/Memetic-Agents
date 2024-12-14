from typing import Dict, Any, Callable, Union
import inspect
from src.log_config import setup_logger, log_tool_call, log_agent_message, log_error, log_event
from src.base_agent.models import ToolCall
from src.base_agent.config import AgentConfig
from logging import Logger
import json
import sys
import importlib

class ToolError(Exception):
    """Custom exception for tool-related errors."""
    pass

class ToolManager:
    """Manages tool registration, loading, and execution."""
    def __init__(self, agent: Any):
        self.agent = agent
        self.tools: Dict[str, Dict[str, Any]] = agent.tools
        self.internal_tools: Dict[str, Callable] = agent.internal_tools
        log_event(self.agent.logger, "tool.installed", 
                 f"Installed tool module for {self.agent.config.agent_name}")

    def register(self, tool: Union[Dict[str, Any], Callable]) -> None:
        """Register a new tool/function that the agent can use.
        
        Args:
            tool: Either a tool definition dictionary or a callable method
        """
        if callable(tool):
            # Handle internal tool (method)
            tool_name = tool.__name__
            if tool_name in self.tools:
                self.agent.logger.warning(f"External tool {tool_name} will be overridden by internal method")
            
            # Get function signature and docstring for parameters
            sig = inspect.signature(tool)
            doc = inspect.getdoc(tool) or ""
            
            # Create tool definition from method
            tool_def = {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": doc.split("\n")[0],  # First line of docstring
                    "parameters": {
                        "type": "object",
                        "properties": {
                            param.name: {"type": "string"}  # Simplified - could be enhanced
                            for param in sig.parameters.values()
                        },
                        "required": [
                            param.name
                            for param in sig.parameters.values()
                            if param.default == inspect.Parameter.empty
                        ]
                    }
                }
            }
            
            self.internal_tools[tool_name] = tool
            self.tools[tool_name] = tool_def
            log_event(self.agent.logger, "tool.registered", f"Registered internal tool: {tool_name}")
            
        else:
            # Handle external tool definition (existing logic)
            if "function" not in tool:
                tool = {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool["parameters"]
                    }
                }
                
            # Validate tool definition
            required_fields = {"name", "description", "parameters"}
            if not all(field in tool["function"] for field in required_fields):
                raise ValueError(f"Tool definition missing required fields: {required_fields}")
                
            tool_name = tool["function"]["name"]
            if tool_name in self.internal_tools:
                log_event(self.agent.logger, "tool.warning", 
                         f"External tool {tool_name} will not override existing internal method")
                return
            
            if tool_name in self.tools:
                raise ValueError(f"Tool {tool_name} already registered")
                
            self.tools[tool_name] = tool
            log_event(self.agent.logger, "tool.registered", f"Registered external tool: {tool_name}")
        
    async def execute(self, tool_call: ToolCall) -> Dict[str, Any]:
        """Execute a tool call and return the result."""
        tool_name = tool_call.function["name"]
        tool_args = tool_call.function["arguments"]
        
        log_event(self.agent.logger, "tool.executing", 
                  f"Starting execution of tool: {tool_name}", level="DEBUG")
        
        if tool_name not in self.tools:
            error_msg = f"Unknown tool: {tool_name}"
            log_event(self.agent.logger, "tool.error", error_msg, level="ERROR")
            raise ToolError(error_msg)
        
        try:
            # Load the tool
            log_event(self.agent.logger, "tool.loading", 
                     f"Loading tool implementation: {tool_name}", level="DEBUG")
            tool_func = await self.load(tool_name)
            
            # Parse arguments
            try:
                args = json.loads(tool_args)
                log_event(self.agent.logger, "tool.executing", 
                         f"Parsed arguments for {tool_name}: {args}", level="DEBUG")
            except json.JSONDecodeError as e:
                error_msg = f"Invalid tool arguments: {tool_args}"
                log_event(self.agent.logger, "tool.error", error_msg, level="ERROR")
                raise ToolError(error_msg) from e
            
            # Execute the tool
            source = "internal" if tool_name in self.internal_tools else "external"
            log_event(self.agent.logger, "tool.executing", 
                     f"Executing {source} tool {tool_name} with args: {args}")
            
            result = await tool_func(**args)
            
            # Log the successful execution and result
            log_tool_call(self.agent.logger, tool_name, args, result)
            log_event(self.agent.logger, "tool.success", 
                     f"Tool {tool_name} executed successfully")
            
            return result
                
        except ToolError:
            raise
        except Exception as e:
            error_msg = f"Tool execution failed: {str(e)}"
            log_error(self.agent.logger, error_msg, exc_info=e)
            raise ToolError(error_msg)
        finally:
            log_event(self.agent.logger, "tool.status", f"Setting status to PROCESSING after tool execution")
        
    async def load(self, tool_name: str) -> Callable:
        """Load a tool by name, preferring internal tools over external ones.
        
        Args:
            tool_name: Name of the tool to load
            
        Returns:
            Callable: The loaded tool function
            
        Raises:
            ToolError: If tool cannot be loaded
        """
        try:
            # First check for internal tool
            if tool_name in self.internal_tools:
                log_event(self.agent.logger, "tool.loading", f"Loading internal tool: {tool_name}", level="DEBUG")
                return self.internal_tools[tool_name]
            
            # Fall back to external tool
            log_event(self.agent.logger, "tool.loading", f"Loading external tool: {tool_name}", level="DEBUG")
            if str(self.agent.config.tools_path) not in sys.path:
                sys.path.append(str(self.agent.config.tools_path))
            
            spec = importlib.util.spec_from_file_location(
                tool_name,
                str(self.agent.config.tools_path / f"{tool_name}.py")
            )
            if not spec or not spec.loader:
                raise ToolError(f"Could not find tool module: {tool_name}")
                
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            tool_func = getattr(module, tool_name)
            log_event(self.agent.logger, "tool.success", f"Successfully loaded external tool: {tool_name}")
            return tool_func
                
        except Exception as e:
            log_event(self.agent.logger, "tool.error", f"Failed to load tool {tool_name}: {str(e)}")
            raise ToolError(f"Failed to load tool {tool_name}: {str(e)}")