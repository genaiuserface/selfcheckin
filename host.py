
"""
LangGraph MCP Host with Automatic Token Management
Handles OAuth tokens and passes them to MCP clients seamlessly
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, TypedDict, Annotated
from dataclasses import dataclass

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# LangChain imports
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# MCP Client (from previous artifact)
from hybrid_mcp_client_complete import MCPClient, MCPTool

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================================================================
# State Management
# ==============================================================================

class ConversationState(TypedDict):
    """State for LangGraph conversation flow"""
    messages: Annotated[List[BaseMessage], add_messages]
    user_context: Dict[str, Any]
    mcp_tools: List[BaseTool]
    oauth_token: Optional[str]
    user_id: Optional[str]
    conversation_id: str
    tool_results: Dict[str, Any]
    error_context: Optional[Dict[str, Any]]

@dataclass
class UserSession:
    """User session with token and context"""
    user_id: str
    oauth_token: str
    email: str
    scopes: List[str]
    expires_at: datetime
    session_id: str
    ldap_groups: List[str] = None
    repo_permissions: Dict[str, List[str]] = None

# ==============================================================================
# Token Management Service
# ==============================================================================

class TokenManager:
    """Manages OAuth tokens for users"""
    
    def __init__(self):
        self.active_sessions: Dict[str, UserSession] = {}
        self.session_by_user: Dict[str, str] = {}  # user_id -> session_id
    
    async def create_session(self, 
                           user_id: str, 
                           oauth_token: str,
                           email: str = None,
                           scopes: List[str] = None,
                           expires_in: int = 3600) -> UserSession:
        """Create a new user session with OAuth token"""
        
        session_id = f"session_{user_id}_{int(datetime.utcnow().timestamp())}"
        expires_at = datetime.utcnow() + timedelta(seconds=expires_in)
        
        session = UserSession(
            user_id=user_id,
            oauth_token=oauth_token,
            email=email or f"{user_id}@company.com",
            scopes=scopes or [],
            expires_at=expires_at,
            session_id=session_id
        )
        
        # Store session
        self.active_sessions[session_id] = session
        self.session_by_user[user_id] = session_id
        
        logger.info(f"Created session {session_id} for user {user_id}")
        return session
    
    def get_session(self, session_id: str) -> Optional[UserSession]:
        """Get session by ID"""
        session = self.active_sessions.get(session_id)
        
        # Check if session is expired
        if session and session.expires_at < datetime.utcnow():
            self.remove_session(session_id)
            return None
        
        return session
    
    def get_session_by_user(self, user_id: str) -> Optional[UserSession]:
        """Get active session for user"""
        session_id = self.session_by_user.get(user_id)
        if session_id:
            return self.get_session(session_id)
        return None
    
    def remove_session(self, session_id: str):
        """Remove a session"""
        session = self.active_sessions.pop(session_id, None)
        if session:
            self.session_by_user.pop(session.user_id, None)
            logger.info(f"Removed session {session_id}")
    
    def refresh_token(self, session_id: str, new_token: str, expires_in: int = 3600):
        """Refresh OAuth token for session"""
        session = self.get_session(session_id)
        if session:
            session.oauth_token = new_token
            session.expires_at = datetime.utcnow() + timedelta(seconds=expires_in)
            logger.info(f"Refreshed token for session {session_id}")

# ==============================================================================
# MCP Tool Factory with Token Injection
# ==============================================================================

class TokenAwareMCPTool(BaseTool):
    """MCP Tool that automatically injects OAuth token"""
    
    name: str = Field(...)
    description: str = Field(...)
    mcp_client: MCPClient = Field(...)
    tool_name: str = Field(...)
    token_manager: TokenManager = Field(...)
    session_id: str = Field(...)
    
    class Config:
        arbitrary_types_allowed = True
    
    def _run(self, **kwargs) -> str:
        """Synchronous wrapper for async MCP call with token injection"""
        try:
            return asyncio.run(self._async_run(**kwargs))
        except Exception as e:
            return f"Error: {str(e)}"
    
    async def _arun(self, **kwargs) -> str:
        """Async MCP call with automatic token injection"""
        return await self._async_run(**kwargs)
    
    async def _async_run(self, **kwargs) -> str:
        """Execute MCP tool with token validation"""
        try:
            # Get current session and validate token
            session = self.token_manager.get_session(self.session_id)
            if not session:
                return "Error: Session expired. Please re-authenticate."
            
            # Update MCP client with current token
            self.mcp_client.oauth_token = session.oauth_token
            
            # Execute the tool
            result = await self.mcp_client.call_tool(self.tool_name, kwargs)
            
            # Log successful execution
            logger.info(f"Tool {self.tool_name} executed successfully for user {session.user_id}")
            
            return result
            
        except Exception as e:
            error_msg = f"Tool execution failed: {str(e)}"
            logger.error(f"Tool {self.tool_name} failed for session {self.session_id}: {error_msg}")
            return error_msg

class MCPToolFactory:
    """Factory for creating token-aware MCP tools"""
    
    def __init__(self, token_manager: TokenManager):
        self.token_manager = token_manager
        self.mcp_clients: Dict[str, MCPClient] = {}
    
    async def create_mcp_client(self, server_url: str, session_id: str) -> MCPClient:
        """Create or reuse MCP client for session"""
        session = self.token_manager.get_session(session_id)
        if not session:
            raise ValueError("Invalid session")
        
        client_key = f"{server_url}_{session_id}"
        
        if client_key not in self.mcp_clients:
            # Create new MCP client
            client = MCPClient(server_url, session.oauth_token)
            await client.__aenter__()
            await client.initialize()
            
            self.mcp_clients[client_key] = client
            logger.info(f"Created MCP client for session {session_id}")
        
        return self.mcp_clients[client_key]
    
    async def create_tools_for_session(self, 
                                     server_url: str, 
                                     session_id: str) -> List[TokenAwareMCPTool]:
        """Create all MCP tools for a user session"""
        try:
            # Get MCP client
            mcp_client = await self.create_mcp_client(server_url, session_id)
            
            # Get available tools from server
            tools_response = await mcp_client.list_tools()
            tools = []
            
            if "result" in tools_response and "tools" in tools_response["result"]:
                for tool_info in tools_response["result"]["tools"]:
                    tool = TokenAwareMCPTool(
                        name=tool_info["name"],
                        description=tool_info["description"],
                        mcp_client=mcp_client,
                        tool_name=tool_info["name"],
                        token_manager=self.token_manager,
                        session_id=session_id
                    )
                    tools.append(tool)
                    
                logger.info(f"Created {len(tools)} tools for session {session_id}")
            
            return tools
            
        except Exception as e:
            logger.error(f"Failed to create tools for session {session_id}: {str(e)}")
            return []
    
    async def cleanup_session(self, session_id: str):
        """Cleanup MCP clients for session"""
        clients_to_remove = [key for key in self.mcp_clients.keys() if key.endswith(f"_{session_id}")]
        
        for client_key in clients_to_remove:
            client = self.mcp_clients.pop(client_key)
            try:
                await client.__aexit__(None, None, None)
            except Exception as e:
                logger.warning(f"Error closing MCP client {client_key}: {str(e)}")

# ==============================================================================
# LangGraph Workflow Nodes
# ==============================================================================

class MCPHost:
    """LangGraph-based MCP Host with token management"""
    
    def __init__(self, 
                 mcp_server_url: str,
                 llm_model: str = "gpt-4o",
                 system_prompt: str = None):
        
        self.mcp_server_url = mcp_server_url
        self.llm = ChatOpenAI(model=llm_model, temperature=0)
        
        # Token and session management
        self.token_manager = TokenManager()
        self.tool_factory = MCPToolFactory(self.token_manager)
        
        # LangGraph setup
        self.checkpointer = MemorySaver()
        self.workflow = self._create_workflow()
        
        # System prompt
        self.system_prompt = system_prompt or self._default_system_prompt()
    
    def _default_system_prompt(self) -> str:
        """Default system prompt for the MCP host"""
        return """You are an AI assistant that helps users manage Git repositories and pull requests.

You have access to various Git operations through MCP (Model Context Protocol) tools:
- validate_pr: Validate pull requests for code quality and compliance
- merge_pr: Merge validated pull requests
- get_pr_status: Get current status of pull requests
- create_branch: Create new branches
- get_commit_history: View commit history
- deploy_to_staging: Deploy to staging environment

When users ask about Git operations:
1. First understand what they want to accomplish
2. Check if they have proper permissions for the repository
3. Use the appropriate tools to complete their request
4. Provide clear feedback about the results

Always be helpful and explain what you're doing. If a user lacks permissions for a repository, explain why and suggest alternatives.

Current conversation context will include user authentication and repository access information."""
    
    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow"""
        
        workflow = StateGraph(ConversationState)
        
        # Add nodes
        workflow.add_node("authenticate_user", self.authenticate_user_node)
        workflow.add_node("load_mcp_tools", self.load_mcp_tools_node)
        workflow.add_node("process_message", self.process_message_node)
        workflow.add_node("execute_tools", self.execute_tools_node)
        workflow.add_node("generate_response", self.generate_response_node)
        workflow.add_node("handle_error", self.handle_error_node)
        
        # Define edges
        workflow.add_edge(START, "authenticate_user")
        workflow.add_edge("authenticate_user", "load_mcp_tools")
        workflow.add_edge("load_mcp_tools", "process_message")
        
        # Conditional edges from process_message
        workflow.add_conditional_edges(
            "process_message",
            self.should_use_tools,
            {
                "use_tools": "execute_tools",
                "generate_response": "generate_response",
                "error": "handle_error"
            }
        )
        
        workflow.add_edge("execute_tools", "generate_response")
        workflow.add_edge("generate_response", END)
        workflow.add_edge("handle_error", END)
        
        return workflow.compile(checkpointer=self.checkpointer)
    
    async def authenticate_user_node(self, state: ConversationState) -> ConversationState:
        """Authenticate user and validate session"""
        logger.info("Authenticating user...")
        
        try:
            # Extract user context from state
            user_context = state.get("user_context", {})
            oauth_token = state.get("oauth_token")
            user_id = state.get("user_id")
            
            if not oauth_token or not user_id:
                # Check if we have an existing session
                if "session_id" in user_context:
                    session = self.token_manager.get_session(user_context["session_id"])
                    if session:
                        state["oauth_token"] = session.oauth_token
                        state["user_id"] = session.user_id
                        logger.info(f"Restored session for user {session.user_id}")
                        return state
                
                # No valid authentication
                state["error_context"] = {
                    "type": "authentication_required",
                    "message": "Please provide OAuth token and user ID"
                }
                return state
            
            # Create or update session
            session = await self.token_manager.create_session(
                user_id=user_id,
                oauth_token=oauth_token,
                email=user_context.get("email"),
                scopes=user_context.get("scopes", [])
            )
            
            # Update state with session info
            state["user_context"]["session_id"] = session.session_id
            state["oauth_token"] = session.oauth_token
            state["user_id"] = session.user_id
            
            logger.info(f"User {user_id} authenticated successfully")
            
        except Exception as e:
            logger.error(f"Authentication failed: {str(e)}")
            state["error_context"] = {
                "type": "authentication_error",
                "message": str(e)
            }
        
        return state
    
    async def load_mcp_tools_node(self, state: ConversationState) -> ConversationState:
        """Load MCP tools for the authenticated user"""
        logger.info("Loading MCP tools...")
        
        try:
            session_id = state["user_context"].get("session_id")
            if not session_id:
                raise ValueError("No valid session found")
            
            # Create MCP tools with token injection
            tools = await self.tool_factory.create_tools_for_session(
                self.mcp_server_url, 
                session_id
            )
            
            state["mcp_tools"] = tools
            logger.info(f"Loaded {len(tools)} MCP tools")
            
        except Exception as e:
            logger.error(f"Failed to load MCP tools: {str(e)}")
            state["error_context"] = {
                "type": "tool_loading_error",
                "message": str(e)
            }
        
        return state
    
    async def process_message_node(self, state: ConversationState) -> ConversationState:
        """Process the user message and determine next action"""
        logger.info("Processing user message...")
        
        try:
            # Get the latest user message
            messages = state["messages"]
            if not messages or not isinstance(messages[-1], HumanMessage):
                raise ValueError("No user message found")
            
            user_message = messages[-1].content
            
            # Add system message with user context
            session_id = state["user_context"].get("session_id")
            session = self.token_manager.get_session(session_id)
            
            context_message = f"""
User Context:
- User ID: {session.user_id}
- Email: {session.email}
- Available Scopes: {', '.join(session.scopes)}
- Session ID: {session.session_id}

Available MCP Tools: {', '.join([tool.name for tool in state.get('mcp_tools', [])])}

User Message: {user_message}
"""
            
            # Create messages for LLM
            llm_messages = [
                SystemMessage(content=self.system_prompt),
                SystemMessage(content=context_message)
            ] + messages
            
            # Get LLM response with tools
            response = await self.llm.bind_tools(state["mcp_tools"]).ainvoke(llm_messages)
            
            # Add LLM response to messages
            state["messages"].append(response)
            
        except Exception as e:
            logger.error(f"Message processing failed: {str(e)}")
            state["error_context"] = {
                "type": "message_processing_error",
                "message": str(e)
            }
        
        return state
    
    def should_use_tools(self, state: ConversationState) -> str:
        """Determine if tools should be used"""
        if state.get("error_context"):
            return "error"
        
        messages = state["messages"]
        if messages and hasattr(messages[-1], 'tool_calls') and messages[-1].tool_calls:
            return "use_tools"
        
        return "generate_response"
    
    async def execute_tools_node(self, state: ConversationState) -> ConversationState:
        """Execute the requested tools"""
        logger.info("Executing MCP tools...")
        
        try:
            # Get the tool calls from the last AI message
            last_message = state["messages"][-1]
            tool_results = {}
            
            for tool_call in last_message.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_id = tool_call["id"]
                
                # Find and execute the tool
                tool = next((t for t in state["mcp_tools"] if t.name == tool_name), None)
                if tool:
                    try:
                        result = await tool._arun(**tool_args)
                        tool_results[tool_id] = result
                        
                        # Add tool message to conversation
                        tool_message = ToolMessage(
                            content=result,
                            tool_call_id=tool_id
                        )
                        state["messages"].append(tool_message)
                        
                        logger.info(f"Tool {tool_name} executed successfully")
                        
                    except Exception as e:
                        error_result = f"Tool execution failed: {str(e)}"
                        tool_results[tool_id] = error_result
                        
                        tool_message = ToolMessage(
                            content=error_result,
                            tool_call_id=tool_id
                        )
                        state["messages"].append(tool_message)
                        
                        logger.error(f"Tool {tool_name} failed: {str(e)}")
                else:
                    error_result = f"Tool {tool_name} not found"
                    tool_results[tool_id] = error_result
                    
                    tool_message = ToolMessage(
                        content=error_result,
                        tool_call_id=tool_id
                    )
                    state["messages"].append(tool_message)
            
            state["tool_results"] = tool_results
            
        except Exception as e:
            logger.error(f"Tool execution failed: {str(e)}")
            state["error_context"] = {
                "type": "tool_execution_error",
                "message": str(e)
            }
        
        return state
    
    async def generate_response_node(self, state: ConversationState) -> ConversationState:
        """Generate final response to user"""
        logger.info("Generating final response...")
        
        try:
            # Get LLM to generate final response based on tool results
            messages = [SystemMessage(content=self.system_prompt)] + state["messages"]
            
            response = await self.llm.ainvoke(messages)
            state["messages"].append(response)
            
            logger.info("Generated final response")
            
        except Exception as e:
            logger.error(f"Response generation failed: {str(e)}")
            error_message = AIMessage(content=f"I encountered an error: {str(e)}")
            state["messages"].append(error_message)
        
        return state
    
    async def handle_error_node(self, state: ConversationState) -> ConversationState:
        """Handle errors and provide helpful messages"""
        error_context = state.get("error_context", {})
        error_type = error_context.get("type", "unknown_error")
        error_message = error_context.get("message", "An unknown error occurred")
        
        if error_type == "authentication_required":
            response = AIMessage(content="""
I need you to authenticate first. Please provide your OAuth token and user ID.

You can do this by including your authentication details in the conversation context:
- OAuth Token: Your valid OAuth token
- User ID: Your user identifier
- Email: Your email address (optional)
- Scopes: Your authorized scopes (optional)

Once authenticated, I'll be able to help you with Git operations using the MCP tools.
""")
        elif error_type == "tool_loading_error":
            response = AIMessage(content=f"""
I'm having trouble connecting to the MCP server to load the Git tools.

Error: {error_message}

This might be due to:
- MCP server being unavailable
- Authentication issues with your token
- Network connectivity problems

Please check your authentication and try again.
""")
        else:
            response = AIMessage(content=f"""
I encountered an error while processing your request.

Error Type: {error_type}
Error Message: {error_message}

Please try again or contact support if the issue persists.
""")
        
        state["messages"].append(response)
        return state
    
    # Public interface methods
    
    async def start_conversation(self, 
                               user_id: str,
                               oauth_token: str,
                               email: str = None,
                               scopes: List[str] = None) -> str:
        """Start a new conversation with user authentication"""
        
        conversation_id = f"conv_{user_id}_{int(datetime.utcnow().timestamp())}"
        
        initial_state = ConversationState(
            messages=[],
            user_context={
                "email": email,
                "scopes": scopes or []
            },
            mcp_tools=[],
            oauth_token=oauth_token,
            user_id=user_id,
            conversation_id=conversation_id,
            tool_results={},
            error_context=None
        )
        
        # Run authentication and tool loading
        config = {"configurable": {"thread_id": conversation_id}}
        result = await self.workflow.ainvoke(initial_state, config)
        
        if result.get("error_context"):
            error = result["error_context"]
            return f"Authentication failed: {error['message']}"
        
        session_id = result["user_context"]["session_id"]
        tools_count = len(result["mcp_tools"])
        
        return f"""
✅ Authentication successful!
📋 Session ID: {session_id}
🔧 Loaded {tools_count} MCP tools
📝 Conversation ID: {conversation_id}

I'm ready to help you with Git operations. What would you like to do?

Available tools: {', '.join([tool.name for tool in result['mcp_tools']])}
"""
    
    async def send_message(self, 
                         conversation_id: str,
                         message: str) -> str:
        """Send a message in an existing conversation"""
        
        config = {"configurable": {"thread_id": conversation_id}}
        
        # Get current state
        current_state = await self.workflow.aget_state(config)
        if not current_state.values:
            return "❌ Conversation not found. Please start a new conversation."
        
        # Add user message
        new_message = HumanMessage(content=message)
        
        # Update state with new message
        updated_state = {
            **current_state.values,
            "messages": current_state.values["messages"] + [new_message]
        }
        
        # Run workflow
        result = await self.workflow.ainvoke(updated_state, config)
        
        # Return the last AI message
        messages = result["messages"]
        ai_messages = [msg for msg in reversed(messages) if isinstance(msg, AIMessage)]
        
        if ai_messages:
            return ai_messages[0].content
        else:
            return "I wasn't able to generate a response. Please try again."
    
    async def get_conversation_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get conversation history"""
        config = {"configurable": {"thread_id": conversation_id}}
        
        current_state = await self.workflow.aget_state(config)
        if not current_state.values:
            return []
        
        messages = current_state.values.get("messages", [])
        history = []
        
        for msg in messages:
            if isinstance(msg, (HumanMessage, AIMessage)):
                history.append({
                    "type": "human" if isinstance(msg, HumanMessage) else "ai",
                    "content": msg.content,
                    "timestamp": datetime.utcnow().isoformat()
                })
        
        return history
    
    async def cleanup_session(self, user_id: str):
        """Cleanup user session and resources"""
        session = self.token_manager.get_session_by_user(user_id)
        if session:
            await self.tool_factory.cleanup_session(session.session_id)
            self.token_manager.remove_session(session.session_id)
            logger.info(f"Cleaned up session for user {user_id}")

# ==============================================================================
# Usage Example
# ==============================================================================

async def main():
    """Example usage of the LangGraph MCP Host"""
    
    # Initialize the MCP Host
    host = MCPHost(
        mcp_server_url="http://localhost:8000",
        llm_model="gpt-4o"
    )
    
    # Example user authentication
    user_id = "john.doe"
    oauth_token = "your-oauth-token-here"  # In real usage, get from OAuth flow
    email = "john.doe@company.com"
    scopes = ["git:pr:read", "git:pr:validate", "git:pr:merge", "git:branch:create"]
    
    try:
        # Start conversation
        print("🚀 Starting conversation...")
        auth_result = await host.start_conversation(
            user_id=user_id,
            oauth_token=oauth_token,
            email=email,
            scopes=scopes
        )
        print(f"Auth Result: {auth_result}")
        
        # Get conversation ID (in real usage, extract from auth_result)
        conversation_id = f"conv_{user_id}_{int(datetime.utcnow().timestamp())}"
        
        # Send some example messages
        test_messages = [
            "Can you validate PR #123 in the frontend-app repository?",
            "What's the status of PR #456 in backend-api?",
            "Create a new branch called 'feature/new-ui' in frontend-app",
            "Show me the commit history for the main branch in backend-api"
        ]
        
        for message in test_messages:
            print(f"\n👤 User: {message}")
            response = await host.send_message(conversation_id, message)
            print(f"🤖 Assistant: {response}")
        
        # Get conversation history
        print("\n📜 Conversation History:")
        history = await host.get_conversation_history(conversation_id)
        for entry in history:
            print(f"{entry['type'].upper()}: {entry['content'][:100]}...")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        logger.error(f"Main execution failed: {str(e)}")
    
    finally:
        # Cleanup
        await host.cleanup_session(user_id)
        print("🧹 Session cleaned up")

if __name__ == "__main__":
    asyncio.run(main())
Made with
1
