"""
Production-ready Snowflake Cortex client with MCP server integration
Complete implementation with all validators and missing logic filled in
"""

import json
import logging
import os
import sqlite3
import uuid
import asyncio
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager
import aiohttp
import aiofiles
from concurrent.futures import ThreadPoolExecutor
import re

from pydantic import BaseModel, SecretStr, validator, Field
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import Annotated

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('snowflake_client.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# EXCEPTIONS AND ENUMS
# =============================================================================

class SnowflakeClientError(Exception):
    """Base exception for Snowflake client errors"""
    pass

class AuthenticationError(SnowflakeClientError):
    """Authentication-related errors"""
    pass

class MCPError(SnowflakeClientError):
    """MCP server communication errors"""
    pass

class SessionError(SnowflakeClientError):
    """Session management errors"""
    pass

class NodeType(Enum):
    ASSISTANT = "assistant_node"
    TOOL_EXECUTION = "tool_execution_node"
    FINAL = "final_node"

class SessionStorageType(Enum):
    LOCAL = "local"
    S3 = "s3"  # Future implementation

# =============================================================================
# CONFIGURATION MODELS
# =============================================================================

class SnowflakeConfig(BaseModel):
    """Complete Snowflake Cortex configuration"""
    # Core connection settings
    account: str = Field(..., description="Snowflake account identifier")
    user: str = Field(..., description="Snowflake username")
    password: SecretStr = Field(..., description="Snowflake password")
    warehouse: Optional[str] = Field(None, description="Snowflake warehouse")
    role: Optional[str] = Field(None, description="Snowflake role")
    database: Optional[str] = Field(None, description="Default database")
    schema: Optional[str] = Field(None, description="Default schema")
    
    # Host settings (for custom deployments)
    host: Optional[str] = Field(None, description="Custom Snowflake host URL")
    port: Optional[int] = Field(None, description="Custom port (usually 443)")
    region: Optional[str] = Field(None, description="Snowflake region")
    
    # Authentication settings
    authenticator: str = Field(default="snowflake", description="Authentication method")
    okta_endpoint: Optional[str] = Field(None, description="OKTA endpoint URL")
    private_key: Optional[str] = Field(None, description="RSA private key for key-pair auth")
    private_key_passphrase: Optional[SecretStr] = Field(None, description="Private key passphrase")
    token: Optional[SecretStr] = Field(None, description="OAuth token")
    
    # API settings
    model: str = Field(default="mistral-large", description="Cortex model to use")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Model temperature")
    max_tokens: int = Field(default=4096, gt=0, le=32000, description="Maximum tokens to generate")
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="Top-p sampling parameter")
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0, description="Frequency penalty")
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0, description="Presence penalty")
    
    # Connection settings
    timeout: int = Field(default=60, gt=0, le=300, description="Request timeout in seconds")
    max_retries: int = Field(default=3, ge=0, le=10, description="Maximum retry attempts")
    retry_delay: float = Field(default=1.0, ge=0.1, le=10.0, description="Base delay between retries")
    connection_timeout: int = Field(default=30, gt=0, le=120, description="Connection timeout")
    read_timeout: int = Field(default=120, gt=0, le=600, description="Read timeout")
    
    # SSL and security
    ssl_verify: bool = Field(default=True, description="Verify SSL certificates")
    ssl_cert_file: Optional[str] = Field(None, description="SSL certificate file path")
    ssl_key_file: Optional[str] = Field(None, description="SSL key file path")
    ssl_ca_file: Optional[str] = Field(None, description="SSL CA file path")
    
    # Session settings
    session_parameters: Dict[str, Any] = Field(default_factory=dict, description="Session parameters")
    client_session_keep_alive: bool = Field(default=True, description="Keep session alive")
    
    # Logging and debugging
    log_level: str = Field(default="INFO", description="Logging level")
    enable_request_logging: bool = Field(default=False, description="Log all requests/responses")
    
    @validator('account')
    def validate_account(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError("Account must be a non-empty string")
        # Remove whitespace and convert to lowercase
        v = v.strip().lower()
        # Validate account format (alphanumeric, dots, hyphens allowed)
        if not re.match(r'^[a-zA-Z0-9.-]+$', v):
            raise ValueError("Account must contain only alphanumeric characters, dots, and hyphens")
        return v
    
    @validator('user')
    def validate_user(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError("User must be a non-empty string")
        v = v.strip()
        if len(v) < 1:
            raise ValueError("User cannot be empty")
        return v
    
    @validator('authenticator')
    def validate_authenticator(cls, v):
        valid_authenticators = ['snowflake', 'okta', 'externalbrowser', 'oauth', 'jwt']
        if v.lower() not in valid_authenticators:
            raise ValueError(f"Authenticator must be one of {valid_authenticators}")
        return v.lower()
    
    @validator('model')
    def validate_model(cls, v):
        # Common Snowflake Cortex models
        valid_models = [
            'mistral-large', 'mistral-7b', 'mixtral-8x7b',
            'llama2-70b-chat', 'llama3-8b', 'llama3-70b',
            'gemma-7b', 'reka-core', 'reka-flash'
        ]
        if v not in valid_models:
            logger.info(f"Retrieved {len(tools)} tools from MCP server")
            return tools
            
        except Exception as e:
            logger.error(f"Failed to list tools: {str(e)}")
            return self._tools_cache if self._tools_cache else []
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> ToolResult:
        """Call a tool on the MCP server"""
        call_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        try:
            payload = {
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments
                }
            }
            
            response = await self._make_request("mcp/tools", payload)
            
            if "error" in response:
                error_msg = response["error"].get("message", "Unknown error")
                return ToolResult(
                    call_id=call_id,
                    error=error_msg,
                    execution_time=(datetime.now() - start_time).total_seconds()
                )
            
            result = response.get("result", {})
            
            return ToolResult(
                call_id=call_id,
                result=result,
                execution_time=(datetime.now() - start_time).total_seconds()
            )
            
        except Exception as e:
            logger.error(f"Tool call failed for {tool_name}: {str(e)}")
            return ToolResult(
                call_id=call_id,
                error=str(e),
                execution_time=(datetime.now() - start_time).total_seconds()
            )

# =============================================================================
# SNOWFLAKE CORTEX CLIENT - COMPLETE IMPLEMENTATION
# =============================================================================

class SnowflakeCortexClient(BaseChatModel):
    """Complete Snowflake Cortex client with enhanced authentication"""
    
    def __init__(self, config: SnowflakeConfig):
        super().__init__()
        self.config = config
        self._token = None
        self._token_expiry = None
        self._session = None
        self._available_tools = []
        
        logger.info(f"Initialized Snowflake Cortex client for account: {config.account}")
    
    async def __aenter__(self):
        """Async context manager entry"""
        connector = aiohttp.TCPConnector(
            limit=100,
            limit_per_host=30,
            keepalive_timeout=30,
            enable_cleanup_closed=True,
            ssl=self.config.ssl_verify
        )
        
        timeout = aiohttp.ClientTimeout(
            total=self.config.timeout,
            connect=self.config.connection_timeout,
            sock_read=self.config.read_timeout
        )
        
        self._session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout
        )
        
        logger.info("Initialized Snowflake HTTP session")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self._session:
            await self._session.close()
            logger.debug("Closed Snowflake HTTP session")
    
    async def authenticate(self) -> str:
        """Authenticate with Snowflake and get access token"""
        if self._is_token_valid():
            return self._token
        
        try:
            if self.config.authenticator == "okta":
                return await self._authenticate_okta()
            else:
                return await self._authenticate_snowflake()
        except Exception as e:
            logger.error(f"Authentication failed: {str(e)}")
            raise AuthenticationError(f"Failed to authenticate: {str(e)}")
    
    async def _authenticate_snowflake(self) -> str:
        """Standard Snowflake authentication"""
        payload = {
            "grant_type": "password",
            "username": self.config.user,
            "password": self.config.password.get_secret_value(),
            "scope": "session:role-any"
        }
        
        if self.config.role:
            payload["role"] = self.config.role
        if self.config.warehouse:
            payload["warehouse"] = self.config.warehouse
        
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        
        for attempt in range(self.config.max_retries + 1):
            try:
                async with self._session.post(
                    self.config.auth_endpoint,
                    data=payload,
                    headers=headers
                ) as response:
                    response_text = await response.text()
                    
                    if response.status == 200:
                        token_data = json.loads(response_text)
                        self._token = token_data["access_token"]
                        self._token_expiry = datetime.now() + timedelta(
                            seconds=token_data.get("expires_in", 3600) - 60
                        )
                        logger.info("Successfully authenticated with Snowflake")
                        return self._token
                    else:
                        logger.warning(f"Auth attempt {attempt + 1} failed: {response.status} - {response_text}")
                        
                        if attempt == self.config.max_retries:
                            raise AuthenticationError(f"Authentication failed: {response.status} - {response_text}")
                        
                        await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
                        
            except aiohttp.ClientError as e:
                logger.warning(f"Auth connection error (attempt {attempt + 1}): {str(e)}")
                if attempt == self.config.max_retries:
                    raise AuthenticationError(f"Authentication connection failed: {str(e)}")
                await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
        
        raise AuthenticationError("Max authentication retries exceeded")
    
    async def _authenticate_okta(self) -> str:
        """OKTA authentication"""
        payload = {
            "grant_type": "password",
            "username": self.config.user,
            "password": self.config.password.get_secret_value(),
            "scope": "session:role-any"
        }
        
        if self.config.okta_endpoint:
            payload["authenticator"] = self.config.okta_endpoint
        
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json"
        }
        
        async with self._session.post(
            self.config.auth_endpoint,
            data=payload,
            headers=headers
        ) as response:
            response_text = await response.text()
            
            if response.status == 200:
                token_data = json.loads(response_text)
                self._token = token_data["access_token"]
                self._token_expiry = datetime.now() + timedelta(
                    seconds=token_data.get("expires_in", 3600) - 60
                )
                logger.info("Successfully authenticated with OKTA")
                return self._token
            else:
                raise AuthenticationError(f"OKTA authentication failed: {response.status} - {response_text}")
    
    def _is_token_valid(self) -> bool:
        """Check if current token is still valid"""
        return (
            self._token is not None 
            and self._token_expiry is not None 
            and datetime.now() < self._token_expiry
        )
    
    def set_available_tools(self, tools: List[Dict[str, Any]]):
        """Set available tools for function calling"""
        self._available_tools = tools
        logger.info(f"Set {len(tools)} available tools")
    
    def _convert_message_to_dict(self, message: BaseMessage) -> Dict[str, Any]:
        """Convert LangChain message to Snowflake format"""
        role_mapping = {
            AIMessage: "assistant",
            SystemMessage: "system", 
            HumanMessage: "user"
        }
        
        return {
            "role": role_mapping.get(type(message), "user"),
            "content": message.content
        }
    
    def _extract_tool_calls(self, content: str) -> List[ToolCall]:
        """Extract tool calls from assistant response"""
        tool_calls = []
        
        # Pattern 1: JSON function calls in code blocks
        json_pattern = r'```json\s*{\s*"function":\s*"([^"]+)",\s*"arguments":\s*(\{[^}]*\})\s*}\s*```'
        matches = re.findall(json_pattern, content, re.MULTILINE | re.DOTALL)
        
        for match in matches:
            try:
                func_name, args_str = match
                arguments = json.loads(args_str)
                tool_calls.append(ToolCall(name=func_name, arguments=arguments))
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON tool call: {str(e)}")
        
        # Pattern 2: Function call syntax
        func_pattern = r'call_function\s*\(\s*"([^"]+)",\s*(\{[^}]*\})\s*\)'
        matches = re.findall(func_pattern, content, re.MULTILINE | re.DOTALL)
        
        for match in matches:
            try:
                func_name, args_str = match
                arguments = json.loads(args_str)
                tool_calls.append(ToolCall(name=func_name, arguments=arguments))
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse function call: {str(e)}")
        
        # Pattern 3: Simple tool mentions with parameters
        tool_pattern = r'use_tool:\s*(\w+)\s*\((.*?)\)'
        matches = re.findall(tool_pattern, content, re.MULTILINE)
        
        for match in matches:
            try:
                func_name, params_str = match
                # Simple parameter parsing
                arguments = {}
                if params_str.strip():
                    # Basic parsing for key=value pairs
                    param_pairs = re.findall(r'(\w+)=([^,]+)', params_str)
                    for key, value in param_pairs:
                        # Try to parse as JSON, otherwise keep as string
                        try:
                            arguments[key] = json.loads(value.strip())
                        except:
                            arguments[key] = value.strip().strip('"\'')
                
                tool_calls.append(ToolCall(name=func_name, arguments=arguments))
            except Exception as e:
                logger.warning(f"Failed to parse tool mention: {str(e)}")
        
        return tool_calls
    
    async def _generate_async(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any
    ) -> ChatResult:
        """Async generation method"""
        try:
            token = await self.authenticate()
            
            message_dicts = [self._convert_message_to_dict(m) for m in messages]
            
            # Add tool information to system message if tools are available
            if self._available_tools and message_dicts:
                tools_info = self._format_tools_for_prompt()
                
                # Find or create system message
                system_msg_idx = None
                for i, msg in enumerate(message_dicts):
                    if msg["role"] == "system":
                        system_msg_idx = i
                        break
                
                if system_msg_idx is not None:
                    # Append to existing system message
                    message_dicts[system_msg_idx]["content"] += f"\n\nAvailable Tools:\n{tools_info}"
                else:
                    # Insert new system message at the beginning
                    system_message = {
                        "role": "system",
                        "content": f"You have access to the following tools:\n{tools_info}\n\nUse tools when needed to help answer user queries."
                    }
                    message_dicts.insert(0, system_message)
            
            payload = {
                "messages": message_dicts,
                "model": self.config.model,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "top_p": self.config.top_p,
                "stream": False
            }
            
            # Add optional parameters
            if self.config.frequency_penalty != 0.0:
                payload["frequency_penalty"] = self.config.frequency_penalty
            if self.config.presence_penalty != 0.0:
                payload["presence_penalty"] = self.config.presence_penalty
            
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }
            
            if self.config.warehouse:
                headers["X-Snowflake-Warehouse"] = self.config.warehouse
            if self.config.role:
                headers["X-Snowflake-Role"] = self.config.role
            
            # Log request if enabled
            if self.config.enable_request_logging:
                logger.debug(f"Cortex request: {json.dumps(payload, indent=2)}")
            
            for attempt in range(self.config.max_retries + 1):
                try:
                    async with self._session.post(
                        self.config.cortex_endpoint,
                        json=payload,
                        headers=headers
                    ) as response:
                        response_text = await response.text()
                        
                        if response.status == 200:
                            result = json.loads(response_text)
                            
                            # Log response if enabled
                            if self.config.enable_request_logging:
                                logger.debug(f"Cortex response: {json.dumps(result, indent=2)}")
                            
                            if "choices" not in result or not result["choices"]:
                                raise SnowflakeClientError("No choices returned from API")
                            
                            choice = result["choices"][0]
                            content = choice["message"]["content"]
                            
                            # Apply stop tokens
                            if stop:
                                for stop_token in stop:
                                    content = content.split(stop_token)[0]
                            
                            message = AIMessage(
                                content=content,
                                response_metadata={
                                    "usage": result.get("usage", {}),
                                    "model": self.config.model,
                                    "finish_reason": choice.get("finish_reason"),
                                    "created": result.get("created")
                                }
                            )
                            
                            return ChatResult(generations=[ChatGeneration(message=message)])
                        else:
                            logger.warning(f"Cortex API attempt {attempt + 1} failed: {response.status} - {response_text}")
                            
                            if attempt == self.config.max_retries:
                                raise SnowflakeClientError(f"Cortex API failed: {response.status} - {response_text}")
                            
                            await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
                            
                except aiohttp.ClientError as e:
                    logger.warning(f"Cortex connection error (attempt {attempt + 1}): {str(e)}")
                    if attempt == self.config.max_retries:
                        raise SnowflakeClientError(f"Cortex connection failed: {str(e)}")
                    await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
            
            raise SnowflakeClientError("Max retries exceeded for Cortex API")
            
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            raise SnowflakeClientError(f"Generation failed: {str(e)}")
    
    def _format_tools_for_prompt(self) -> str:
        """Format available tools for inclusion in the prompt"""
        if not self._available_tools:
            return "No tools available."
        
        formatted_tools = []
        for tool in self._available_tools:
            name = tool.get("name", "unknown")
            description = tool.get("description", "No description")
            
            # Format input schema if available
            input_schema = tool.get("inputSchema", {})
            properties = input_schema.get("properties", {})
            
            if properties:
                params = []
                required = input_schema.get("required", [])
                for param, details in properties.items():
                    param_type = details.get("type", "string")
                    param_desc = details.get("description", "")
                    required_marker = " (required)" if param in required else ""
                    params.append(f"  - {param} ({param_type}){required_marker}: {param_desc}")
                
                params_str = "\n".join(params) if params else "  No parameters"
                formatted_tools.append(f"• {name}: {description}\n  Parameters:\n{params_str}")
            else:
                formatted_tools.append(f"• {name}: {description}")
        
        return "\n\n".join(formatted_tools)
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any
    ) -> ChatResult:
        """Sync wrapper for async generation"""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self._generate_async(messages, stop, **kwargs))
    
    @property
    def _llm_type(self) -> str:
        return f"snowflake-cortex-{self.config.model}"

# =============================================================================
# GRAPH NODES - COMPLETE IMPLEMENTATION
# =============================================================================

class GraphNodes:
    """Complete graph nodes with async execution and robust error handling"""
    
    def __init__(self, cortex_client: SnowflakeCortexClient, mcp_client: MCPClient):
        self.cortex_client = cortex_client
        self.mcp_client = mcp_client
    
    async def assistant_node(self, state: GraphState) -> GraphState:
        """Assistant node that generates responses and identifies tool calls"""
        try:
            logger.info(f"Processing assistant node for session {state.session_id}")
            
            # Generate response
            result = await self.cortex_client._generate_async(state.messages)
            assistant_message = result.generations[0].message
            
            # Extract tool calls from the response
            tool_calls = self.cortex_client._extract_tool_calls(assistant_message.content)
            
            # Update state
            state.messages.append(assistant_message)
            state.tool_calls.extend(tool_calls)
            state.updated_at = datetime.now()
            
            logger.info(f"Assistant generated response with {len(tool_calls)} tool calls")
            return state
            
        except Exception as e:
            logger.error(f"Assistant node error: {str(e)}")
            error_message = AIMessage(
                content=f"I encountered an error processing your request: {str(e)}. Please try again or contact support if the issue persists."
            )
            state.messages.append(error_message)
            state.updated_at = datetime.now()
            return state
    
    async def tool_execution_node(self, state: GraphState) -> GraphState:
        """Execute tools identified by the assistant"""
        try:
            if not state.tool_calls:
                logger.info("No tool calls to execute")
                return state
            
            logger.info(f"Executing {len(state.tool_calls)} tool calls")
            
            # Execute tools concurrently
            tasks = []
            for tool_call in state.tool_calls:
                task = self.mcp_client.call_tool(tool_call.name, tool_call.arguments)
                tasks.append(task)
            
            # Wait for all tool executions to complete
            tool_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            processed_results = []
            for i, result in enumerate(tool_results):
                if isinstance(result, Exception):
                    # Handle exceptions from tool execution
                    error_result = ToolResult(
                        call_id=state.tool_calls[i].call_id,
                        error=str(result),
                        execution_time=0.0
                    )
                    processed_results.append(error_result)
                else:
                    processed_results.append(result)
            
            state.tool_results.extend(processed_results)
            
            # Add tool results to conversation
            for tool_call, result in zip(state.tool_calls, processed_results):
                if result.success:
                    content = f"Tool '{tool_call.name}' executed successfully.\nResult: {json.dumps(result.result, indent=2, default=str)}"
                else:
                    content = f"Tool '{tool_call.name}' failed with error: {result.error}"
                
                state.messages.append(SystemMessage(content=content))
            
            # Clear tool calls for next iteration
            state.tool_calls = []
            state.updated_at = datetime.now()
            
            successful_tools = len([r for r in processed_results if r.success])
            failed_tools = len(processed_results) - successful_tools
            logger.info(f"Tool execution completed: {successful_tools} successful, {failed_tools} failed")
            
            return state
            
        except Exception as e:
            logger.error(f"Tool execution node error: {str(e)}")
            error_message = SystemMessage(content=f"Tool execution encountered an error: {str(e)}")
            state.messages.append(error_message)
            state.tool_calls = []  # Clear pending tool calls
            state.updated_at = datetime.now()
            return state
    
    async def final_node(self, state: GraphState) -> GraphState:
        """Final node that generates the comprehensive response"""
        try:
            logger.info(f"Processing final node for session {state.session_id}")
            
            # If we have tool results, generate a summary response
            if state.tool_results:
                successful_results = [r for r in state.tool_results if r.success]
                failed_results = [r for r in state.tool_results if not r.success]
                
                if successful_results:
                    # Create a summary prompt
                    summary_prompt = (
                        "Based on the tool execution results above, please provide a comprehensive and helpful response "
                        "to the user's original query. Include relevant details from the tool results and provide "
                        "actionable insights where appropriate."
                    )
                    
                    if failed_results:
                        summary_prompt += f"\n\nNote: {len(failed_results)} tool(s) failed to execute, which may limit the completeness of this response."
                    
                    state.messages.append(HumanMessage(content=summary_prompt))
                    
                    # Generate final response
                    result = await self.cortex_client._generate_async(state.messages)
                    final_message = result.generations[0].message
                    state.messages.append(final_message)
                else:
                    # All tools failed
                    error_message = AIMessage(
                        content="I apologize, but I encountered issues executing the required tools for your request. "
                               "The tools may be temporarily unavailable or there may be configuration issues. "
                               "Please try again later or contact support if the problem persists."
                    )
                    state.messages.append(error_message)
            
            state.updated_at = datetime.now()
            logger.info(f"Final node completed for session {state.session_id}")
            return state
            
        except Exception as e:
            logger.error(f"Final node error: {str(e)}")
            error_message = AIMessage(
                content=f"I encountered an error generating the final response: {str(e)}. "
                       "Please try again or contact support if the issue persists."
            )
            state.messages.append(error_message)
            state.updated_at = datetime.now()
            return state

# =============================================================================
# MAIN CLIENT CLASS - COMPLETE IMPLEMENTATION
# =============================================================================

class SnowflakeConversationClient:
    """Complete production-ready Snowflake Cortex conversation client"""
    
    def __init__(
        self,
        snowflake_config: Optional[SnowflakeConfig] = None,
        mcp_config: Optional[MCPConfig] = None,
        session_config: Optional[SessionConfig] = None
    ):
        self.snowflake_config = snowflake_config or SnowflakeConfig.from_env()
        self.mcp_config = mcp_config or MCPConfig.from_env()
        self.session_config = session_config or SessionConfig.from_env()
        
        self.session_manager = SessionManager(self.session_config)
        self.cortex_client = None
        self.mcp_client = None
        self.graph = None
        self._initialized = False
        
        logger.info("Initialized Snowflake Conversation Client")
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
    
    async def initialize(self):
        """Initialize the client and its components"""
        if self._initialized:
            return
        
        try:
            logger.info("Initializing Snowflake Conversation Client...")
            
            # Initialize Cortex client
            self.cortex_client = SnowflakeCortexClient(self.snowflake_config)
            await self.cortex_client.__aenter__()
            
            # Initialize MCP client
            self.mcp_client = MCPClient(self.mcp_config)
            await self.mcp_client.__aenter__()
            
            # Load available tools from MCP server
            tools = await self.mcp_client.list_tools()
            self.cortex_client.set_available_tools(tools)
            
            # Build the conversation graph
            await self._build_graph()
            
            self._initialized = True
            logger.info("Client initialization completed successfully")
            
        except Exception as e:
            logger.error(f"Client initialization failed: {str(e)}")
            await self.close()
            raise SnowflakeClientError(f"Failed to initialize client: {str(e)}")
    
    async def close(self):
        """Clean up resources"""
        try:
            if self.cortex_client:
                await self.cortex_client.__aexit__(None, None, None)
            
            if self.mcp_client:
                await self.mcp_client.__aexit__(None, None, None)
            
            self._initialized = False
            logger.info("Client closed successfully")
        except Exception as e:
            logger.error(f"Error closing client: {str(e)}")
    
    async def _build_graph(self):
        """Build the conversation graph"""
        nodes = GraphNodes(self.cortex_client, self.mcp_client)
        
        # Create the state graph
        graph = StateGraph(GraphState)
        
        # Add nodes
        graph.add_node(NodeType.ASSISTANT.value, nodes.assistant_node)
        graph.add_node(NodeType.TOOL_EXECUTION.value, nodes.tool_execution_node)
        graph.add_node(NodeType.FINAL.value, nodes.final_node)
        
        # Define routing logic
        def route_after_assistant(state: GraphState) -> str:
            """Route after assistant node based on tool calls"""
            if state.tool_calls:
                logger.debug(f"Routing to tool execution with {len(state.tool_calls)} tool calls")
                return NodeType.TOOL_EXECUTION.value
            else:
                logger.debug("Routing to final node - no tools needed")
                return NodeType.FINAL.value
        
        # Define edges
        graph.add_edge(START, NodeType.ASSISTANT.value)
        graph.add_conditional_edges(
            NodeType.ASSISTANT.value,
            route_after_assistant,
            {
                NodeType.TOOL_EXECUTION.value: NodeType.TOOL_EXECUTION.value,
                NodeType.FINAL.value: NodeType.FINAL.value
            }
        )
        graph.add_edge(NodeType.TOOL_EXECUTION.value, NodeType.FINAL.value)
        graph.add_edge(NodeType.FINAL.value, END)
        
        self.graph = graph.compile()
        logger.info("Conversation graph built successfully")
    
    async def chat(
        self,
        message: str,
        session_id: Optional[str] = None,
        user_context: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Main chat interface"""
        if not self._initialized:
            raise SnowflakeClientError("Client not initialized. Use async context manager or call initialize().")
        
        try:
            # Generate session ID if not provided
            if not session_id:
                session_id = str(uuid.uuid4())
            
            # Try to load existing session
            existing_state = await self.session_manager.load_session(session_id)
            
            if existing_state:
                # Continue existing conversation
                initial_state = existing_state
                initial_state.messages.append(HumanMessage(content=message))
                if user_context:
                    initial_state.user_context.update(user_context)
                if metadata:
                    initial_state.metadata.update(metadata)
                
                logger.info(f"Continuing session {session_id} with {len(existing_state.messages)} existing messages")
            else:
                # Create new session
                system_message = SystemMessage(
                    content="You are a helpful AI assistant powered by Snowflake Cortex. "
                           "You have access to various tools to help answer user queries. "
                           "Use tools when appropriate and provide comprehensive, helpful responses."
                )
                
                initial_state = GraphState(
                    session_id=session_id,
                    messages=[system_message, HumanMessage(content=message)],
                    user_context=user_context or {},
                    metadata=metadata or {}
                )
                
                logger.info(f"Starting new session {session_id}")
            
            # Execute the graph
            start_time = datetime.now()
            final_state = await self.graph.ainvoke(initial_state)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Save the session
            await self.session_manager.save_session(final_state)
            
            # Extract the final response
            response_content = "I apologize, but I wasn't able to generate a proper response."
            for message in reversed(final_state.messages):
                if isinstance(message, AIMessage) and message.content.strip():
                    response_content = message.content
                    break
            
            # Prepare response
            response = {
                "session_id": session_id,
                "response": response_content,
                "execution_time": execution_time,
                "tool_calls_made": len(final_state.tool_results),
                "successful_tools": len([r for r in final_state.tool_results if r.success]),
                "failed_tools": len([r for r in final_state.tool_results if not r.success]),
                "message_count": len(final_state.messages)
            }
            
            logger.info(f"Chat completed for session {session_id} in {execution_time:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"Chat failed for session {session_id}: {str(e)}")
            raise SnowflakeClientError(f"Chat failed: {str(e)}")
    
    async def get_session_history(self, session_id: str) -> Optional[List[Dict[str, Any]]]:
        """Get conversation history for a session"""
        try:
            state = await self.session_manager.load_session(session_id)
            if not state:
                return None
            
            history = []
            for message in state.messages:
                if isinstance(message, (HumanMessage, AIMessage)):
                    history.append({
                        "role": "user" if isinstance(message, HumanMessage) else "assistant",
                        "content": message.content,
                        "timestamp": datetime.now().isoformat()  # In production, store actual timestamps
                    })
            
            return history
            
        except Exception as e:
            logger.error(f"Failed to get session history for {session_id}: {str(e)}")
            return None
    
    async def list_sessions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List recent sessions"""
        return await self.session_manager.list_sessions(limit)
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete a session"""
        return await self.session_manager.delete_session(session_id)
    
    async def refresh_tools(self) -> List[Dict[str, Any]]:
        """Refresh available tools from MCP server"""
        if not self._initialized:
            raise SnowflakeClientError("Client not initialized")
        
        try:
            tools = await self.mcp_client.list_tools(force_refresh=True)
            self.cortex_client.set_available_tools(tools)
            logger.info(f"Refreshed {len(tools)} tools from MCP server")
            return tools
        except Exception as e:
            logger.error(f"Failed to refresh tools: {str(e)}")
            raise MCPError(f"Failed to refresh tools: {str(e)}")
    
    async def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get currently available tools"""
        if not self._initialized:
            raise SnowflakeClientError("Client not initialized")
        
        return self.cortex_client._available_tools
    
    def validate_configuration(self) -> Dict[str, List[str]]:
        """Validate all configurations and return issues"""
        issues = {
            "snowflake": [],
            "mcp": [],
            "session": []
        }
        
        # Validate Snowflake config
        if not self.snowflake_config.account:
            issues["snowflake"].append("Account is required")
        if not self.snowflake_config.user:
            issues["snowflake"].append("User is required")
        if not self.snowflake_config.password.get_secret_value():
            issues["snowflake"].append("Password is required")
        
        # Validate MCP config
        if not self.mcp_config.server_url:
            issues["mcp"].append("Server URL is required")
        
        # Validate session config
        if not self.session_config.local_db_path:
            issues["session"].append("Database path is required")
        
        return {k: v for k, v in issues.items() if v}

# =============================================================================
# USAGE EXAMPLES AND TESTING
# =============================================================================

async def example_basic_usage():
    """Example of basic usage"""
    print("=== Basic Usage Example ===")
    
    async with SnowflakeConversationClient() as client:
        # Validate configuration first
        issues = client.validate_configuration()
        if issues:
            print(f"Configuration issues found: {issues}")
            return
        
        # Simple chat
        print("Testing simple chat...")
        response = await client.chat("Hello, what can you help me with?")
        print(f"Response: {response['response']}")
        print(f"Session ID: {response['session_id']}")
        print(f"Execution time: {response['execution_time']:.2f}s")
        
        # Continue conversation in same session
        print("\nTesting conversation continuation...")
        response2 = await client.chat(
            "Can you help me analyze some data?",
            session_id=response['session_id']
        )
        print(f"Follow-up Response: {response2['response']}")
        
        # Get session history
        print("\nTesting session history...")
        history = await client.get_session_history(response['session_id'])
        if history:
            print("Conversation History:")
            for msg in history:
                content_preview = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
                print(f"  {msg['role']}: {content_preview}")
        
        # List sessions
        sessions = await client.list_sessions(limit=5)
        print(f"\nRecent sessions: {len(sessions)}")
        
        # Get available tools
        tools = await client.get_available_tools()
        print(f"Available tools: {len(tools)}")
        if tools:
            print("Tool names:", [tool.get('name', 'unknown') for tool in tools[:3]])

async def example_with_user_context():
    """Example with user context and metadata"""
    print("\n=== User Context Example ===")
    
    async with SnowflakeConversationClient() as client:
        response = await client.chat(
            "What's my current project status?",
            user_context={
                "user_id": "user123", 
                "role": "data_analyst",
                "department": "analytics"
            },
            metadata={
                "source": "web_app", 
                "version": "1.0",
                "request_id": str(uuid.uuid4())
            }
        )
        
        print(f"Contextual Response: {response['response']}")
        print(f"Tools used: {response['tool_calls_made']}")

async def test_error_handling():
    """Test error handling scenarios"""
    print("\n=== Error Handling Test ===")
    
    # Test with invalid configuration
    invalid_config = SnowflakeConfig(
        account="",  # Invalid empty account
        user="test",
        password=SecretStr("test")
    )
    
    try:
        client = SnowflakeConversationClient(snowflake_config=invalid_config)
        issues = client.validate_configuration()
        print(f"Validation caught issues: {issues}")
    except Exception as e:
        print(f"Configuration validation failed as expected: {str(e)}")

def main():
    """Main function for testing"""
    import sys
    
    # Set up logging for testing
    logging.getLogger().setLevel(logging.INFO)
    
    # Set default environment variables for testing
    default_env_vars = {
        "SNOWFLAKE_ACCOUNT": "your-account",
        "SNOWFLAKE_USER": "your-username",
        "SNOWFLAKE_PASSWORD": "your-password",
        "MCP_SERVER_URL": "http://localhost:8000"
    }
    
    for key, value in default_env_vars.items():
        os.environ.setdefault(key, value)
    
    print("Starting Snowflake Cortex Client Examples...")
    
    try:
        # Run examples
        asyncio.run(example_basic_usage())
        asyncio.run(example_with_user_context())
        asyncio.run(test_error_handling())
        
        print("\n✅ All examples completed successfully!")
        
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
    except Exception as e:
        logger.error(f"Example failed: {str(e)}")
        print(f"\n❌ Example failed: {str(e)}")
        sys.exit(1)

# =============================================================================
# QUICK START TEMPLATE
# =============================================================================

class QuickStart:
    """Quick start template for easy deployment"""
    
    @staticmethod
    def create_env_template():
        """Create environment template file"""
        template = """# Snowflake Configuration
SNOWFLAKE_ACCOUNT=your-account-here
SNOWFLAKE_USER=your-username
SNOWFLAKE_PASSWORD=your-password
SNOWFLAKE_WAREHOUSE=your-warehouse
SNOWFLAKE_ROLE=your-role
SNOWFLAKE_DATABASE=your-database
SNOWFLAKE_SCHEMA=your-schema

# For OKTA authentication
SNOWFLAKE_AUTHENTICATOR=okta
SNOWFLAKE_OKTA_ENDPOINT=https://your-company.okta.com

# MCP Server Configuration
MCP_SERVER_URL=http://localhost:8000
MCP_API_KEY=your-mcp-api-key

# Session Management
SESSION_STORAGE_TYPE=local
SESSION_DB_PATH=./sessions.db
SESSION_TTL=86400

# Optional: Advanced Settings
SNOWFLAKE_MODEL=mistral-large
SNOWFLAKE_TEMPERATURE=0.7
SNOWFLAKE_MAX_TOKENS=4096
SNOWFLAKE_TIMEOUT=60
SNOWFLAKE_MAX_RETRIES=3
"""
        
        with open(".env.template", "w") as f:
            f.write(template)
        
        print("Created .env.template file")
        print("Copy to .env and fill in your actual values")
    
    @staticmethod
    async def quick_test():
        """Quick test of the client"""
        print("🚀 Running quick test...")
        
        try:
            async with SnowflakeConversationClient() as client:
                # Validate configuration
                issues = client.validate_configuration()
                if issues:
                    print(f"❌ Configuration issues: {issues}")
                    return False
                
                # Test basic functionality
                response = await client.chat("Hello, can you introduce yourself?")
                print(f"✅ Client working! Response: {response['response'][:100]}...")
                return True
                
        except Exception as e:
            print(f"❌ Quick test failed: {str(e)}")
            return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Snowflake Cortex Client")
    parser.add_argument("--create-template", action="store_true", help="Create .env template")
    parser.add_argument("--quick-test", action="store_true", help="Run quick test")
    parser.add_argument("--examples", action="store_true", help="Run examples")
    
    args = parser.parse_args()
    
    if args.create_template:
        QuickStart.create_env_template()
    elif args.quick_test:
        asyncio.run(QuickStart.quick_test())
    elif args.examples:
        main()
    else:
        print("Use --help to see available options")
        print("Quick start: python client.py --create-template")
warning(f"Model '{v}' not in known valid models: {valid_models}")
        return v
    
    @validator('log_level')
    def validate_log_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()
    
    @validator('port')
    def validate_port(cls, v):
        if v is not None and (v < 1 or v > 65535):
            raise ValueError("Port must be between 1 and 65535")
        return v
    
    @validator('ssl_cert_file', 'ssl_key_file', 'ssl_ca_file')
    def validate_ssl_files(cls, v):
        if v is not None and not Path(v).exists():
            logger.warning(f"SSL file not found: {v}")
        return v
    
    @property
    def connection_url(self) -> str:
        """Generate Snowflake connection URL"""
        if self.host:
            protocol = "https"
            port_str = f":{self.port}" if self.port and self.port != 443 else ""
            return f"{protocol}://{self.host}{port_str}"
        
        # Standard Snowflake URL format
        return f"https://{self.account}.snowflakecomputing.com"
    
    @property
    def auth_endpoint(self) -> str:
        """Authentication endpoint"""
        return f"{self.connection_url}/oauth/token"
    
    @property
    def cortex_endpoint(self) -> str:
        """Cortex API endpoint"""
        return f"{self.connection_url}/api/v2/cortex/analyst/message"
    
    @classmethod
    def from_env(cls) -> "SnowflakeConfig":
        """Create configuration from environment variables"""
        # Helper functions for type conversion
        def get_env_float(key: str, default: str) -> float:
            try:
                return float(os.getenv(key, default))
            except (ValueError, TypeError):
                return float(default)
        
        def get_env_int(key: str, default: str) -> int:
            try:
                return int(os.getenv(key, default))
            except (ValueError, TypeError):
                return int(default)
        
        def get_env_bool(key: str, default: str) -> bool:
            value = os.getenv(key, default).lower()
            return value in ('true', '1', 'yes', 'on')
        
        def get_env_dict(key: str, default: str = "{}") -> Dict[str, Any]:
            try:
                return json.loads(os.getenv(key, default))
            except (json.JSONDecodeError, TypeError):
                return {}
        
        # Build configuration
        config_data = {
            # Core connection
            "account": os.getenv("SNOWFLAKE_ACCOUNT", ""),
            "user": os.getenv("SNOWFLAKE_USER", ""),
            "password": SecretStr(os.getenv("SNOWFLAKE_PASSWORD", "")),
            "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
            "role": os.getenv("SNOWFLAKE_ROLE"),
            "database": os.getenv("SNOWFLAKE_DATABASE"),
            "schema": os.getenv("SNOWFLAKE_SCHEMA"),
            
            # Host settings
            "host": os.getenv("SNOWFLAKE_HOST"),
            "port": get_env_int("SNOWFLAKE_PORT", "443") if os.getenv("SNOWFLAKE_PORT") else None,
            "region": os.getenv("SNOWFLAKE_REGION"),
            
            # Authentication
            "authenticator": os.getenv("SNOWFLAKE_AUTHENTICATOR", "snowflake"),
            "okta_endpoint": os.getenv("SNOWFLAKE_OKTA_ENDPOINT"),
            "private_key": os.getenv("SNOWFLAKE_PRIVATE_KEY"),
            "private_key_passphrase": SecretStr(os.getenv("SNOWFLAKE_PRIVATE_KEY_PASSPHRASE", "")) if os.getenv("SNOWFLAKE_PRIVATE_KEY_PASSPHRASE") else None,
            "token": SecretStr(os.getenv("SNOWFLAKE_TOKEN", "")) if os.getenv("SNOWFLAKE_TOKEN") else None,
            
            # API settings
            "model": os.getenv("SNOWFLAKE_MODEL", "mistral-large"),
            "temperature": get_env_float("SNOWFLAKE_TEMPERATURE", "0.7"),
            "max_tokens": get_env_int("SNOWFLAKE_MAX_TOKENS", "4096"),
            "top_p": get_env_float("SNOWFLAKE_TOP_P", "1.0"),
            "frequency_penalty": get_env_float("SNOWFLAKE_FREQUENCY_PENALTY", "0.0"),
            "presence_penalty": get_env_float("SNOWFLAKE_PRESENCE_PENALTY", "0.0"),
            
            # Connection settings
            "timeout": get_env_int("SNOWFLAKE_TIMEOUT", "60"),
            "max_retries": get_env_int("SNOWFLAKE_MAX_RETRIES", "3"),
            "retry_delay": get_env_float("SNOWFLAKE_RETRY_DELAY", "1.0"),
            "connection_timeout": get_env_int("SNOWFLAKE_CONNECTION_TIMEOUT", "30"),
            "read_timeout": get_env_int("SNOWFLAKE_READ_TIMEOUT", "120"),
            
            # SSL settings
            "ssl_verify": get_env_bool("SNOWFLAKE_SSL_VERIFY", "true"),
            "ssl_cert_file": os.getenv("SNOWFLAKE_SSL_CERT_FILE"),
            "ssl_key_file": os.getenv("SNOWFLAKE_SSL_KEY_FILE"),
            "ssl_ca_file": os.getenv("SNOWFLAKE_SSL_CA_FILE"),
            
            # Session settings
            "session_parameters": get_env_dict("SNOWFLAKE_SESSION_PARAMETERS"),
            "client_session_keep_alive": get_env_bool("SNOWFLAKE_KEEP_ALIVE", "true"),
            
            # Logging
            "log_level": os.getenv("SNOWFLAKE_LOG_LEVEL", "INFO"),
            "enable_request_logging": get_env_bool("SNOWFLAKE_ENABLE_REQUEST_LOGGING", "false"),
        }
        
        # Remove None values
        config_data = {k: v for k, v in config_data.items() if v is not None}
        
        return cls(**config_data)

class MCPConfig(BaseModel):
    """Complete MCP server configuration"""
    server_url: str = Field(..., description="MCP server HTTP URL")
    timeout: int = Field(default=30, gt=0, le=300, description="Request timeout")
    max_retries: int = Field(default=3, ge=0, le=10, description="Maximum retry attempts")
    api_key: Optional[str] = Field(None, description="API key for authentication")
    headers: Dict[str, str] = Field(default_factory=dict, description="Additional headers")
    
    @validator('server_url')
    def validate_server_url(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError("Server URL must be a non-empty string")
        v = v.strip()
        if not v.startswith(('http://', 'https://')):
            raise ValueError("Server URL must start with http:// or https://")
        return v.rstrip('/')
    
    @validator('headers')
    def validate_headers(cls, v):
        if not isinstance(v, dict):
            raise ValueError("Headers must be a dictionary")
        return v
    
    @classmethod
    def from_env(cls) -> "MCPConfig":
        """Create MCP configuration from environment variables"""
        def get_env_int(key: str, default: str) -> int:
            try:
                return int(os.getenv(key, default))
            except (ValueError, TypeError):
                return int(default)
        
        def get_env_dict(key: str, default: str = "{}") -> Dict[str, str]:
            try:
                return json.loads(os.getenv(key, default))
            except (json.JSONDecodeError, TypeError):
                return {}
        
        return cls(
            server_url=os.getenv("MCP_SERVER_URL", "http://localhost:8000"),
            timeout=get_env_int("MCP_TIMEOUT", "30"),
            max_retries=get_env_int("MCP_MAX_RETRIES", "3"),
            api_key=os.getenv("MCP_API_KEY"),
            headers=get_env_dict("MCP_HEADERS")
        )

class SessionConfig(BaseModel):
    """Complete session management configuration"""
    storage_type: SessionStorageType = Field(default=SessionStorageType.LOCAL)
    local_db_path: str = Field(default="sessions.db", description="Local SQLite database path")
    session_ttl: int = Field(default=86400, gt=0, description="Session TTL in seconds (24 hours)")
    max_sessions: int = Field(default=1000, gt=0, description="Maximum sessions to keep")
    
    # Future S3 configuration
    s3_bucket: Optional[str] = Field(None, description="S3 bucket for session storage")
    s3_prefix: Optional[str] = Field(None, description="S3 key prefix")
    
    @validator('local_db_path')
    def validate_db_path(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError("Database path must be a non-empty string")
        # Ensure directory exists
        Path(v).parent.mkdir(parents=True, exist_ok=True)
        return v
    
    @validator('s3_bucket')
    def validate_s3_bucket(cls, v):
        if v is not None:
            if not isinstance(v, str) or not v.strip():
                raise ValueError("S3 bucket must be a non-empty string")
            # Basic S3 bucket name validation
            if not re.match(r'^[a-z0-9.-]+$', v.lower()):
                raise ValueError("S3 bucket name must contain only lowercase letters, numbers, dots, and hyphens")
        return v
    
    @classmethod
    def from_env(cls) -> "SessionConfig":
        """Create session configuration from environment variables"""
        def get_env_int(key: str, default: str) -> int:
            try:
                return int(os.getenv(key, default))
            except (ValueError, TypeError):
                return int(default)
        
        storage_type_str = os.getenv("SESSION_STORAGE_TYPE", "local").lower()
        storage_type = SessionStorageType.LOCAL if storage_type_str == "local" else SessionStorageType.S3
        
        return cls(
            storage_type=storage_type,
            local_db_path=os.getenv("SESSION_DB_PATH", "sessions.db"),
            session_ttl=get_env_int("SESSION_TTL", "86400"),
            max_sessions=get_env_int("MAX_SESSIONS", "1000"),
            s3_bucket=os.getenv("SESSION_S3_BUCKET"),
            s3_prefix=os.getenv("SESSION_S3_PREFIX", "sessions/")
        )

# =============================================================================
# CORE DATA MODELS
# =============================================================================

@dataclass
class ToolCall:
    """Represents a tool call from the LLM"""
    name: str
    arguments: Dict[str, Any]
    call_id: str = field(default_factory=lambda: str(uuid.uuid4()))

@dataclass
class ToolResult:
    """Result of a tool execution"""
    call_id: str
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    
    @property
    def success(self) -> bool:
        return self.error is None

class GraphState(BaseModel):
    """Enhanced state for LangGraph execution"""
    messages: Annotated[List[BaseMessage], add_messages] = []
    tool_calls: List[ToolCall] = []
    tool_results: List[ToolResult] = []
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_context: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

# =============================================================================
# SESSION MANAGEMENT - COMPLETE IMPLEMENTATION
# =============================================================================

class SessionManager:
    """Complete session manager with local storage"""
    
    def __init__(self, config: SessionConfig):
        self.config = config
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="session")
        
        if config.storage_type == SessionStorageType.LOCAL:
            self._init_local_storage()
        elif config.storage_type == SessionStorageType.S3:
            raise NotImplementedError("S3 storage not yet implemented")
    
    def _init_local_storage(self):
        """Initialize local SQLite storage"""
        db_path = Path(self.config.local_db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        def _create_tables():
            with sqlite3.connect(str(db_path)) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id TEXT PRIMARY KEY,
                        state_data BLOB NOT NULL,
                        user_context TEXT DEFAULT '{}',
                        metadata TEXT DEFAULT '{}',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        expires_at TIMESTAMP NOT NULL
                    )
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_updated_at ON sessions(updated_at)
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_expires_at ON sessions(expires_at)
                """)
                
                conn.commit()
        
        _create_tables()
        logger.info(f"Initialized local session storage: {db_path}")
    
    async def save_session(self, state: GraphState) -> bool:
        """Save session state"""
        try:
            if self.config.storage_type == SessionStorageType.LOCAL:
                return await self._save_local_session(state)
            else:
                raise NotImplementedError("S3 storage not yet implemented")
        except Exception as e:
            logger.error(f"Failed to save session {state.session_id}: {str(e)}")
            return False
    
    async def _save_local_session(self, state: GraphState) -> bool:
        """Save session to local SQLite database"""
        def _save():
            try:
                with sqlite3.connect(self.config.local_db_path) as conn:
                    expires_at = datetime.now() + timedelta(seconds=self.config.session_ttl)
                    state_data = pickle.dumps(state)
                    
                    conn.execute("""
                        INSERT OR REPLACE INTO sessions 
                        (session_id, state_data, user_context, metadata, updated_at, expires_at)
                        VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, ?)
                    """, (
                        state.session_id,
                        state_data,
                        json.dumps(state.user_context),
                        json.dumps(state.metadata),
                        expires_at.isoformat()
                    ))
                    
                    # Cleanup expired sessions
                    conn.execute("DELETE FROM sessions WHERE expires_at < CURRENT_TIMESTAMP")
                    
                    # Limit maximum sessions
                    conn.execute("""
                        DELETE FROM sessions WHERE session_id NOT IN (
                            SELECT session_id FROM sessions 
                            ORDER BY updated_at DESC 
                            LIMIT ?
                        )
                    """, (self.config.max_sessions,))
                    
                    conn.commit()
                    return True
            except Exception as e:
                logger.error(f"Database save error: {str(e)}")
                return False
        
        result = await asyncio.get_event_loop().run_in_executor(self._executor, _save)
        if result:
            logger.debug(f"Saved session {state.session_id}")
        return result
    
    async def load_session(self, session_id: str) -> Optional[GraphState]:
        """Load session state"""
        try:
            if self.config.storage_type == SessionStorageType.LOCAL:
                return await self._load_local_session(session_id)
            else:
                raise NotImplementedError("S3 storage not yet implemented")
        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {str(e)}")
            return None
    
    async def _load_local_session(self, session_id: str) -> Optional[GraphState]:
        """Load session from local SQLite database"""
        def _load():
            try:
                with sqlite3.connect(self.config.local_db_path) as conn:
                    cursor = conn.execute("""
                        SELECT state_data FROM sessions 
                        WHERE session_id = ? AND expires_at > CURRENT_TIMESTAMP
                    """, (session_id,))
                    
                    row = cursor.fetchone()
                    if row:
                        return pickle.loads(row[0])
                    return None
            except Exception as e:
                logger.error(f"Database load error: {str(e)}")
                return None
        
        state = await asyncio.get_event_loop().run_in_executor(self._executor, _load)
        if state:
            logger.debug(f"Loaded session {session_id}")
        return state
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete a session"""
        def _delete():
            try:
                with sqlite3.connect(self.config.local_db_path) as conn:
                    cursor = conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
                    conn.commit()
                    return cursor.rowcount > 0
            except Exception as e:
                logger.error(f"Database delete error: {str(e)}")
                return False
        
        result = await asyncio.get_event_loop().run_in_executor(self._executor, _delete)
        if result:
            logger.debug(f"Deleted session {session_id}")
        return result
    
    async def list_sessions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List recent sessions"""
        def _list():
            try:
                with sqlite3.connect(self.config.local_db_path) as conn:
                    cursor = conn.execute("""
                        SELECT session_id, metadata, created_at, updated_at, expires_at
                        FROM sessions 
                        WHERE expires_at > CURRENT_TIMESTAMP
                        ORDER BY updated_at DESC 
                        LIMIT ?
                    """, (limit,))
                    
                    sessions = []
                    for row in cursor.fetchall():
                        sessions.append({
                            "session_id": row[0],
                            "metadata": json.loads(row[1]) if row[1] else {},
                            "created_at": row[2],
                            "updated_at": row[3],
                            "expires_at": row[4]
                        })
                    return sessions
            except Exception as e:
                logger.error(f"Database list error: {str(e)}")
                return []
        
        return await asyncio.get_event_loop().run_in_executor(self._executor, _list)

# =============================================================================
# MCP CLIENT - COMPLETE IMPLEMENTATION
# =============================================================================

class MCPClient:
    """Complete HTTP-based MCP client for tool integration"""
    
    def __init__(self, config: MCPConfig):
        self.config = config
        self.session = None
        self._tools_cache = {}
        self._cache_timestamp = None
        self._cache_ttl = 300  # 5 minutes
    
    async def __aenter__(self):
        """Async context manager entry"""
        connector = aiohttp.TCPConnector(
            limit=100,
            limit_per_host=30,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )
        
        timeout = aiohttp.ClientTimeout(total=self.config.timeout)
        headers = {"Content-Type": "application/json"}
        
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        
        headers.update(self.config.headers)
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers=headers
        )
        
        logger.info(f"Initialized MCP client for {self.config.server_url}")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
            logger.debug("Closed MCP client session")
    
    async def _make_request(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make HTTP request with retry logic"""
        url = f"{self.config.server_url}/{endpoint.lstrip('/')}"
        
        for attempt in range(self.config.max_retries + 1):
            try:
                async with self.session.post(url, json=payload) as response:
                    response_text = await response.text()
                    
                    if response.status == 200:
                        try:
                            return json.loads(response_text)
                        except json.JSONDecodeError as e:
                            raise MCPError(f"Invalid JSON response: {str(e)}")
                    else:
                        logger.warning(f"MCP request failed (attempt {attempt + 1}): {response.status} - {response_text}")
                        
                        if attempt == self.config.max_retries:
                            raise MCPError(f"MCP request failed: {response.status} - {response_text}")
                        
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                        
            except aiohttp.ClientError as e:
                logger.warning(f"MCP connection error (attempt {attempt + 1}): {str(e)}")
                if attempt == self.config.max_retries:
                    raise MCPError(f"MCP connection failed: {str(e)}")
                await asyncio.sleep(2 ** attempt)
        
        raise MCPError("Max retries exceeded")
    
    async def list_tools(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """List available tools from MCP server"""
        now = datetime.now()
        
        # Check cache
        if (not force_refresh and 
            self._tools_cache and 
            self._cache_timestamp and 
            (now - self._cache_timestamp).total_seconds() < self._cache_ttl):
            return self._tools_cache
        
        try:
            payload = {"method": "tools/list", "params": {}}
            response = await self._make_request("mcp/tools", payload)
            
            tools = response.get("result", {}).get("tools", [])
            self._tools_cache = tools
            self._cache_timestamp = now
            
            logger.
