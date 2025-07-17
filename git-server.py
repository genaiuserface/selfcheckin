
"""
Complete Hybrid MCP Git Server Implementation
Combines OAuth authentication, caching, and flexible tool registration
"""

import asyncio
import json
import time
import jwt
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
from cachetools import TTLCache
import inspect
import subprocess
import os
import tempfile
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================================================================
# MCP Protocol Models
# ==============================================================================

class MCPRequest(BaseModel):
    jsonrpc: str = "2.0"
    id: int
    method: str
    params: Dict[str, Any] = {}

class MCPResponse(BaseModel):
    jsonrpc: str = "2.0"
    id: int
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None

class TokenInfo(BaseModel):
    user_id: str
    scopes: List[str]
    exp: int
    iat: int
    valid: bool = True

@dataclass
class ToolMetadata:
    name: str
    description: str
    schema: Dict[str, Any]
    handler: Callable
    middleware: List[Callable] = None
    requires_auth: bool = True
    required_scopes: List[str] = None
    rate_limit: int = 100  # requests per minute

# ==============================================================================
# Middleware Classes
# ==============================================================================

class MCPMiddleware:
    """Base middleware class"""
    async def before_call(self, tool_name: str, arguments: Dict, request: Request, token_info: TokenInfo = None) -> Dict:
        """Called before tool execution"""
        return {}
    
    async def after_call(self, tool_name: str, result: Dict, request: Request, token_info: TokenInfo = None) -> Dict:
        """Called after tool execution"""
        return result
    
    async def filter_tools(self, tools: Dict[str, ToolMetadata], request: Request, token_info: TokenInfo = None) -> Dict[str, ToolMetadata]:
        """Filter available tools based on request context"""
        return tools

class AuthMiddleware(MCPMiddleware):
    """OAuth authentication middleware"""
    def __init__(self, server_instance):
        self.server = server_instance
    
    async def before_call(self, tool_name: str, arguments: Dict, request: Request, token_info: TokenInfo = None) -> Dict:
        if not token_info:
            return {"error": {"code": -32603, "message": "Authentication required"}}
        
        # Check if tool requires specific scopes
        tool = self.server.tools.get(tool_name)
        if tool and tool.required_scopes:
            missing_scopes = [scope for scope in tool.required_scopes if scope not in token_info.scopes]
            if missing_scopes:
                return {"error": {"code": -32603, "message": f"Missing required scopes: {missing_scopes}"}}
        
        return {}
    
    async def filter_tools(self, tools: Dict[str, ToolMetadata], request: Request, token_info: TokenInfo = None) -> Dict[str, ToolMetadata]:
        """Filter tools based on user scopes"""
        if not token_info:
            return {}
        
        filtered_tools = {}
        for name, tool in tools.items():
            if tool.required_scopes:
                # Check if user has all required scopes
                if all(scope in token_info.scopes for scope in tool.required_scopes):
                    filtered_tools[name] = tool
            else:
                filtered_tools[name] = tool
        
        return filtered_tools

class LoggingMiddleware(MCPMiddleware):
    """Logging middleware"""
    async def before_call(self, tool_name: str, arguments: Dict, request: Request, token_info: TokenInfo = None) -> Dict:
        user_id = token_info.user_id if token_info else "anonymous"
        logger.info(f"Tool call: {tool_name} by user: {user_id} with args: {arguments}")
        return {}
    
    async def after_call(self, tool_name: str, result: Dict, request: Request, token_info: TokenInfo = None) -> Dict:
        user_id = token_info.user_id if token_info else "anonymous"
        logger.info(f"Tool {tool_name} completed successfully for user: {user_id}")
        return result

class RateLimitMiddleware(MCPMiddleware):
    """Rate limiting middleware"""
    def __init__(self):
        self.rate_limits = TTLCache(maxsize=10000, ttl=60)  # 1 minute TTL
    
    async def before_call(self, tool_name: str, arguments: Dict, request: Request, token_info: TokenInfo = None) -> Dict:
        if not token_info:
            return {}
        
        user_id = token_info.user_id
        key = f"{user_id}:{tool_name}"
        
        current_count = self.rate_limits.get(key, 0)
        if current_count >= 100:  # 100 requests per minute limit
            return {"error": {"code": -32603, "message": "Rate limit exceeded"}}
        
        self.rate_limits[key] = current_count + 1
        return {}

class GitValidationMiddleware(MCPMiddleware):
    """Git-specific validation middleware"""
    async def before_call(self, tool_name: str, arguments: Dict, request: Request, token_info: TokenInfo = None) -> Dict:
        if tool_name == "merge_pr":
            # Check if PR is validated before merging
            pr_number = arguments.get("pr_number")
            repository = arguments.get("repository")
            
            # In real implementation, check validation status from database
            logger.info(f"Checking validation status for PR {pr_number} in {repository}")
            
            # Simulate validation check
            # In production, this would query your validation system
            validation_status = await self._check_pr_validation(pr_number, repository)
            if not validation_status.get("validated", False):
                return {"error": {"code": -32603, "message": "PR must be validated before merging"}}
        
        return {}
    
    async def _check_pr_validation(self, pr_number: str, repository: str) -> Dict:
        """Check if PR has been validated"""
        # Mock validation check - in production, check your validation system
        return {"validated": True, "checks_passed": True}

# ==============================================================================
# Git Operations Helper
# ==============================================================================

class GitOperations:
    """Helper class for Git operations"""
    
    @staticmethod
    async def run_git_command(command: List[str], repo_path: str = None) -> Dict[str, Any]:
        """Run a git command and return result"""
        try:
            # Change to repo directory if specified
            cwd = repo_path if repo_path else os.getcwd()
            
            # Run git command
            result = subprocess.run(
                command,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Command timed out",
                "return_code": -1
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "return_code": -1
            }
    
    @staticmethod
    async def clone_repository(repo_url: str, temp_dir: str) -> Dict[str, Any]:
        """Clone a repository to temporary directory"""
        try:
            result = await GitOperations.run_git_command([
                "git", "clone", repo_url, temp_dir
            ])
            return result
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    @staticmethod
    async def get_pr_info(pr_number: str, repository: str) -> Dict[str, Any]:
        """Get PR information (mock implementation)"""
        # In real implementation, this would call GitHub/GitLab API
        return {
            "pr_number": pr_number,
            "repository": repository,
            "title": f"Example PR #{pr_number}",
            "author": "developer",
            "status": "open",
            "base_branch": "main",
            "head_branch": f"feature/pr-{pr_number}",
            "created_at": datetime.utcnow().isoformat(),
            "files_changed": 5,
            "lines_added": 150,
            "lines_removed": 20
        }

# ==============================================================================
# Main Hybrid MCP Server
# ==============================================================================

class HybridMCPGitServer:
    """Production-ready Hybrid MCP Git Server"""
    
    def __init__(self, 
                 name: str = "hybrid-git-server",
                 jwt_secret: str = "your-jwt-secret",
                 oauth_introspect_url: str = None):
        self.name = name
        self.jwt_secret = jwt_secret
        self.oauth_introspect_url = oauth_introspect_url
        
        # Initialize FastAPI app
        self.app = FastAPI(title=f"MCP Git Server: {name}")
        self.security = HTTPBearer()
        
        # Tool registry
        self.tools: Dict[str, ToolMetadata] = {}
        
        # Middleware
        self.global_middleware: List[MCPMiddleware] = []
        
        # Token cache for OAuth validation
        self.token_cache = TTLCache(maxsize=1000, ttl=300)  # 5 minutes
        
        # Git operations helper
        self.git_ops = GitOperations()
        
        # Setup
        self.setup_app()
        self.register_default_tools()
    
    def setup_app(self):
        """Setup FastAPI app with middleware and routes"""
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Add global middleware
        self.add_middleware(LoggingMiddleware())
        self.add_middleware(AuthMiddleware(self))
        self.add_middleware(RateLimitMiddleware())
        self.add_middleware(GitValidationMiddleware())
        
        # Setup MCP routes
        self.setup_routes()
    
    async def validate_token(self, token: str) -> Optional[TokenInfo]:
        """Validate OAuth token with caching"""
        # Check cache first
        cached_token = self.token_cache.get(token)
        if cached_token and cached_token.exp > time.time():
            return cached_token
        
        # Try local JWT validation
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
            if payload.get("exp", 0) < time.time():
                return None
            
            token_info = TokenInfo(
                user_id=payload.get("sub", "unknown"),
                scopes=payload.get("scope", "").split(),
                exp=payload.get("exp", 0),
                iat=payload.get("iat", 0),
                valid=True
            )
            
            self.token_cache[token] = token_info
            return token_info
            
        except jwt.InvalidTokenError:
            pass
        
        # Fallback to OAuth introspection
        if self.oauth_introspect_url:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        self.oauth_introspect_url,
                        data={"token": token}
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            if data.get("active"):
                                token_info = TokenInfo(
                                    user_id=data.get("sub", "unknown"),
                                    scopes=data.get("scope", "").split(),
                                    exp=data.get("exp", 0),
                                    iat=data.get("iat", 0),
                                    valid=True
                                )
                                self.token_cache[token] = token_info
                                return token_info
            except Exception as e:
                logger.error(f"OAuth introspection failed: {e}")
        
        return None
    
    async def get_current_user(self, credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())):
        """Get current user from OAuth token"""
        token = credentials.credentials
        token_info = await self.validate_token(token)
        
        if not token_info:
            raise HTTPException(
                status_code=401,
                detail="Invalid or expired token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return token_info
    
    def setup_routes(self):
        """Setup MCP protocol routes"""
        
        @self.app.post("/mcp/initialize")
        async def initialize(request: MCPRequest):
            """Initialize MCP connection"""
            return MCPResponse(
                id=request.id,
                result={
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {
                        "name": self.name,
                        "version": "1.0.0"
                    }
                }
            )
        
        @self.app.post("/mcp/tools/list")
        async def list_tools(request: MCPRequest, http_request: Request):
            """List available tools"""
            token_info = None
            
            # Try to get token info for filtering
            try:
                auth_header = http_request.headers.get("Authorization")
                if auth_header and auth_header.startswith("Bearer "):
                    token = auth_header.split(" ")[1]
                    token_info = await self.validate_token(token)
            except Exception:
                pass
            
            # Apply middleware filtering
            available_tools = self.tools.copy()
            for middleware in self.global_middleware:
                available_tools = await middleware.filter_tools(available_tools, http_request, token_info)
            
            tools = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": tool.schema
                }
                for tool in available_tools.values()
            ]
            
            return MCPResponse(
                id=request.id,
                result={"tools": tools}
            )
        
        @self.app.post("/mcp/tools/call")
        async def call_tool(request: MCPRequest, http_request: Request):
            """Call a specific tool"""
            tool_name = request.params.get("name")
            arguments = request.params.get("arguments", {})
            
            # Get token info
            token_info = None
            try:
                auth_header = http_request.headers.get("Authorization")
                if auth_header and auth_header.startswith("Bearer "):
                    token = auth_header.split(" ")[1]
                    token_info = await self.validate_token(token)
            except Exception:
                pass
            
            if tool_name not in self.tools:
                return MCPResponse(
                    id=request.id,
                    error={"code": -32601, "message": f"Tool '{tool_name}' not found"}
                )
            
            tool = self.tools[tool_name]
            
            # Apply global middleware
            for middleware in self.global_middleware:
                result = await middleware.before_call(tool_name, arguments, http_request, token_info)
                if result.get("error"):
                    return MCPResponse(id=request.id, error=result["error"])
            
            # Apply tool-specific middleware
            if tool.middleware:
                for middleware in tool.middleware:
                    result = await middleware.before_call(tool_name, arguments, http_request, token_info)
                    if result.get("error"):
                        return MCPResponse(id=request.id, error=result["error"])
            
            try:
                # Execute the tool
                result = await tool.handler(**arguments)
                
                # Apply post-processing middleware
                for middleware in self.global_middleware:
                    result = await middleware.after_call(tool_name, result, http_request, token_info)
                
                if tool.middleware:
                    for middleware in tool.middleware:
                        result = await middleware.after_call(tool_name, result, http_request, token_info)
                
                return MCPResponse(id=request.id, result=result)
                
            except Exception as e:
                logger.error(f"Tool execution failed: {str(e)}")
                return MCPResponse(
                    id=request.id,
                    error={"code": -32603, "message": f"Tool execution failed: {str(e)}"}
                )
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "tools_count": len(self.tools)
            }
    
    def add_middleware(self, middleware: MCPMiddleware):
        """Add global middleware"""
        self.global_middleware.append(middleware)
    
    def _generate_schema(self, func: Callable) -> Dict[str, Any]:
        """Generate JSON schema from function signature"""
        sig = inspect.signature(func)
        schema = {
            "type": "object",
            "properties": {},
            "required": []
        }
        
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            
            prop_type = "string"
            if param.annotation != inspect.Parameter.empty:
                if param.annotation == int:
                    prop_type = "integer"
                elif param.annotation == float:
                    prop_type = "number"
                elif param.annotation == bool:
                    prop_type = "boolean"
                elif param.annotation == list:
                    prop_type = "array"
                elif param.annotation == dict:
                    prop_type = "object"
            
            schema["properties"][param_name] = {
                "type": prop_type,
                "description": f"Parameter {param_name}"
            }
            
            if param.default == inspect.Parameter.empty:
                schema["required"].append(param_name)
        
        return schema
    
    def tool(self, 
             name: str, 
             description: str, 
             schema: Optional[Dict[str, Any]] = None,
             middleware: Optional[List[MCPMiddleware]] = None,
             requires_auth: bool = True,
             required_scopes: Optional[List[str]] = None,
             rate_limit: int = 100):
        """Decorator for registering tools (annotation pattern)"""
        def decorator(func: Callable):
            final_schema = schema or self._generate_schema(func)
            
            self.tools[name] = ToolMetadata(
                name=name,
                description=description,
                schema=final_schema,
                handler=func,
                middleware=middleware,
                requires_auth=requires_auth,
                required_scopes=required_scopes or [],
                rate_limit=rate_limit
            )
            
            return func
        return decorator
    
    def register_tool(self, 
                     name: str, 
                     description: str, 
                     handler: Callable,
                     schema: Optional[Dict[str, Any]] = None,
                     middleware: Optional[List[MCPMiddleware]] = None,
                     required_scopes: Optional[List[str]] = None):
        """Manual tool registration (router pattern)"""
        final_schema = schema or self._generate_schema(handler)
        
        self.tools[name] = ToolMetadata(
            name=name,
            description=description,
            schema=final_schema,
            handler=handler,
            middleware=middleware,
            required_scopes=required_scopes or []
        )
    
    def register_default_tools(self):
        """Register default Git tools using both patterns"""
        
        # Method 1: Using decorators (annotation pattern)
        @self.tool(
            "validate_pr", 
            "Validate a pull request by checking code quality, tests, and compliance",
            required_scopes=["git:pr:validate"]
        )
        async def validate_pr(pr_number: str, repository: str) -> Dict[str, Any]:
            """Validate a PR using automated checks"""
            logger.info(f"Validating PR {pr_number} in {repository}")
            
            # Get PR info
            pr_info = await self.git_ops.get_pr_info(pr_number, repository)
            
            # Simulate validation checks
            await asyncio.sleep(0.1)  # Simulate processing time
            
            # Mock validation results
            validation_result = {
                "pr_number": pr_number,
                "repository": repository,
                "status": "validated",
                "checks": {
                    "code_quality": "passed",
                    "tests": "passed",
                    "security": "passed",
                    "compliance": "passed",
                    "conflicts": "none"
                },
                "pr_info": pr_info,
                "validated_at": datetime.utcnow().isoformat()
            }
            
            return validation_result
        
        @self.tool(
            "merge_pr", 
            "Merge a validated pull request",
            required_scopes=["git:pr:merge"]
        )
        async def merge_pr(pr_number: str, repository: str, merge_method: str = "merge") -> Dict[str, Any]:
            """Merge a PR after validation"""
            logger.info(f"Merging PR {pr_number} in {repository} using {merge_method}")
            
            # Simulate merge operation
            await asyncio.sleep(0.2)
            
            # Mock merge result
            merge_result = {
                "pr_number": pr_number,
                "repository": repository,
                "status": "merged",
                "merge_method": merge_method,
                "commit_sha": f"abc123def456{pr_number}",
                "merged_at": datetime.utcnow().isoformat(),
                "target_branch": "main"
            }
            
            return merge_result
        
        @self.tool(
            "get_pr_status", 
            "Get the current status of a pull request",
            required_scopes=["git:pr:read"]
        )
        async def get_pr_status(pr_number: str, repository: str) -> Dict[str, Any]:
            """Get PR status and information"""
            pr_info = await self.git_ops.get_pr_info(pr_number, repository)
            
            # Add status information
            status_info = {
                **pr_info,
                "checks": {
                    "ci_status": "passed",
                    "review_status": "approved",
                    "mergeable": True
                },
                "last_updated": datetime.utcnow().isoformat()
            }
            
            return status_info
        
        # Method 2: Manual registration (router pattern)
        async def create_branch(repository: str, branch_name: str, base_branch: str = "main") -> Dict[str, Any]:
            """Create a new branch"""
            logger.info(f"Creating branch {branch_name} from {base_branch} in {repository}")
            
            # Simulate branch creation
            await asyncio.sleep(0.1)
            
            return {
                "repository": repository,
                "branch_name": branch_name,
                "base_branch": base_branch,
                "status": "created",
                "created_at": datetime.utcnow().isoformat()
            }
        
        self.register_tool(
            "create_branch",
            "Create a new branch from an existing branch",
            create_branch,
            schema={
                "type": "object",
                "properties": {
                    "repository": {"type": "string", "description": "Repository name"},
                    "branch_name": {"type": "string", "description": "Name of the new branch"},
                    "base_branch": {"type": "string", "description": "Base branch to create from", "default": "main"}
                },
                "required": ["repository", "branch_name"]
            },
            required_scopes=["git:branch:create"]
        )
        
        async def get_commit_history(repository: str, branch: str = "main", limit: int = 10) -> Dict[str, Any]:
            """Get commit history for a branch"""
            logger.info(f"Getting commit history for {branch} in {repository}")
            
            # Mock commit history
            commits = []
            for i in range(min(limit, 10)):
                commits.append({
                    "sha": f"commit{i}abc123def456",
                    "message": f"Commit message {i}",
                    "author": "developer",
                    "date": (datetime.utcnow() - timedelta(days=i)).isoformat()
                })
            
            return {
                "repository": repository,
                "branch": branch,
                "commits": commits,
                "total_commits": len(commits)
            }
        
        self.register_tool(
            "get_commit_history",
            "Get commit history for a repository branch",
            get_commit_history,
            required_scopes=["git:commits:read"]
        )
    
    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """Run the server"""
        import uvicorn
        logger.info(f"Starting {self.name} on {host}:{port}")
        logger.info(f"Registered tools: {list(self.tools.keys())}")
        uvicorn.run(self.app, host=host, port=port)

# ==============================================================================
# Example Usage and Configuration
# ==============================================================================

def create_production_server():
    """Create a production-ready server instance"""
    
    # Configuration
    JWT_SECRET = "your-super-secret-jwt-key-change-in-production"
    OAUTH_INTROSPECT_URL = "https://your-oauth-provider.com/oauth/introspect"
    
    # Create server
    server = HybridMCPGitServer(
        name="production-git-server",
        jwt_secret=JWT_SECRET,
        oauth_introspect_url=OAUTH_INTROSPECT_URL
    )
    
    # Add custom tools using annotation pattern
    @server.tool(
        "deploy_to_staging", 
        "Deploy a merged PR to staging environment",
        required_scopes=["git:deploy:staging"]
    )
    async def deploy_to_staging(pr_number: str, repository: str) -> Dict[str, Any]:
        """Deploy PR to staging after merge"""
        logger.info(f"Deploying PR {pr_number} from {repository} to staging")
        
        # Simulate deployment
        await asyncio.sleep(1.0)
        
        return {
            "pr_number": pr_number,
            "repository": repository,
            "environment": "staging",
            "status": "deployed",
            "deployment_url": f"https://staging.example.com/pr-{pr_number}",
            "deployed_at": datetime.utcnow().isoformat()
        }
    
    # Add custom middleware
    class DeploymentMiddleware(MCPMiddleware):
        async def before_call(self, tool_name: str, arguments: Dict, request: Request, token_info: TokenInfo = None) -> Dict:
            if tool_name == "deploy_to_staging":
                # Check if PR is merged before deployment
                pr_number = arguments.get("pr_number")
                logger.info(f"Checking if PR {pr_number} is merged before deployment")
                # In production, verify PR is actually merged
            return {}
    
    server.add_middleware(DeploymentMiddleware())
    
    return server

# Create and run server
if __name__ == "__main__":
    server = create_production_server()
    server.run()
