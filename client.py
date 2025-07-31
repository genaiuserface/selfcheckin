"""
Simple Snowflake Cortex client with local memory sessions
Uses environment variables, no database required
"""

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import asyncio
import aiohttp
from pydantic import BaseModel, SecretStr, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# EXCEPTIONS
# =============================================================================

class SnowflakeError(Exception):
    """Base Snowflake error"""
    pass

class AuthenticationError(SnowflakeError):
    """Authentication failed"""
    pass

# =============================================================================
# CONFIGURATION
# =============================================================================

class SnowflakeConfig(BaseModel):
    """Snowflake configuration from environment variables"""
    account: str = Field(..., description="Snowflake account")
    user: str = Field(..., description="Username")
    password: SecretStr = Field(..., description="Password")
    warehouse: Optional[str] = Field(None, description="Warehouse")
    role: Optional[str] = Field(None, description="Role")
    
    # Authentication
    authenticator: str = Field(default="snowflake", description="Auth method")
    okta_endpoint: Optional[str] = Field(None, description="OKTA endpoint")
    
    # Cortex API settings
    model: str = Field(default="mistral-large", description="Cortex model")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, gt=0)
    
    # Connection settings
    timeout: int = Field(default=60, description="Request timeout")
    max_retries: int = Field(default=3, description="Max retries")
    
    @property
    def connection_url(self) -> str:
        return f"https://{self.account}.snowflakecomputing.com"
    
    @property
    def auth_endpoint(self) -> str:
        return f"{self.connection_url}/oauth/token"
    
    @property
    def cortex_endpoint(self) -> str:
        return f"{self.connection_url}/api/v2/cortex/analyst/message"
    
    @classmethod
    def from_env(cls) -> "SnowflakeConfig":
        """Load configuration from environment variables"""
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
        
        return cls(
            account=os.getenv("SNOWFLAKE_ACCOUNT", ""),
            user=os.getenv("SNOWFLAKE_USER", ""),
            password=SecretStr(os.getenv("SNOWFLAKE_PASSWORD", "")),
            warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
            role=os.getenv("SNOWFLAKE_ROLE"),
            authenticator=os.getenv("SNOWFLAKE_AUTHENTICATOR", "snowflake"),
            okta_endpoint=os.getenv("SNOWFLAKE_OKTA_ENDPOINT"),
            model=os.getenv("SNOWFLAKE_MODEL", "mistral-large"),
            temperature=get_env_float("SNOWFLAKE_TEMPERATURE", "0.7"),
            max_tokens=get_env_int("SNOWFLAKE_MAX_TOKENS", "4096"),
            timeout=get_env_int("SNOWFLAKE_TIMEOUT", "60"),
            max_retries=get_env_int("SNOWFLAKE_MAX_RETRIES", "3")
        )
    
    def validate(self) -> List[str]:
        """Validate required configuration"""
        issues = []
        if not self.account:
            issues.append("SNOWFLAKE_ACCOUNT is required")
        if not self.user:
            issues.append("SNOWFLAKE_USER is required")
        if not self.password.get_secret_value():
            issues.append("SNOWFLAKE_PASSWORD is required")
        return issues

# =============================================================================
# LOCAL SESSION STORAGE
# =============================================================================

class LocalSessionStorage:
    """Simple in-memory session storage"""
    
    def __init__(self):
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._session_ttl = 3600  # 1 hour TTL
    
    def save_session(self, session_id: str, messages: List[Dict[str, str]], metadata: Dict[str, Any] = None):
        """Save session to memory"""
        self._sessions[session_id] = {
            "messages": messages,
            "metadata": metadata or {},
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        }
        self._cleanup_expired()
    
    def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load session from memory"""
        self._cleanup_expired()
        return self._sessions.get(session_id)
    
    def delete_session(self, session_id: str) -> bool:
        """Delete session"""
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False
    
    def list_sessions(self) -> List[str]:
        """List active session IDs"""
        self._cleanup_expired()
        return list(self._sessions.keys())
    
    def _cleanup_expired(self):
        """Remove expired sessions"""
        now = datetime.now()
        expired = [
            sid for sid, data in self._sessions.items()
            if (now - data["updated_at"]).total_seconds() > self._session_ttl
        ]
        for sid in expired:
            del self._sessions[sid]

# =============================================================================
# SNOWFLAKE CORTEX CLIENT
# =============================================================================

class SnowflakeClient:
    """Simple Snowflake Cortex client with local sessions"""
    
    def __init__(self, config: Optional[SnowflakeConfig] = None):
        self.config = config or SnowflakeConfig.from_env()
        self.storage = LocalSessionStorage()
        
        # Authentication state
        self._session = None
        self._token = None
        self._token_expiry = None
        
        logger.info(f"Initialized Snowflake client for account: {self.config.account}")
    
    async def __aenter__(self):
        """Async context manager entry"""
        connector = aiohttp.TCPConnector(
            limit=50,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )
        
        timeout = aiohttp.ClientTimeout(total=self.config.timeout)
        
        self._session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout
        )
        
        logger.info("HTTP session initialized")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self._session:
            await self._session.close()
            logger.info("HTTP session closed")
    
    def validate_config(self) -> List[str]:
        """Validate configuration"""
        return self.config.validate()
    
    def _is_token_valid(self) -> bool:
        """Check if token is still valid"""
        return (
            self._token is not None 
            and self._token_expiry is not None 
            and datetime.now() < self._token_expiry
        )
    
    async def authenticate(self) -> str:
        """Authenticate and get access token"""
        if self._is_token_valid():
            return self._token
        
        try:
            if self.config.authenticator == "okta":
                return await self._authenticate_okta()
            else:
                return await self._authenticate_snowflake()
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            raise AuthenticationError(f"Authentication failed: {e}")
    
    async def _authenticate_snowflake(self) -> str:
        """Standard Snowflake authentication"""
        payload = {
            "grant_type": "password",
            "username": self.config.user,
            "password": self.config.password.get_secret_value(),
            "scope": "session:role-any"
        }
        
        # Add optional parameters
        if self.config.role:
            payload["role"] = self.config.role
        if self.config.warehouse:
            payload["warehouse"] = self.config.warehouse
        
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        
        # Retry logic
        for attempt in range(self.config.max_retries + 1):
            try:
                async with self._session.post(
                    self.config.auth_endpoint,
                    data=payload,
                    headers=headers
                ) as response:
                    if response.status == 200:
                        token_data = await response.json()
                        self._token = token_data["access_token"]
                        
                        # Set token expiry (subtract 60 seconds for safety)
                        expires_in = token_data.get("expires_in", 3600)
                        self._token_expiry = datetime.now() + timedelta(seconds=expires_in - 60)
                        
                        logger.info("Successfully authenticated with Snowflake")
                        return self._token
                    else:
                        error_text = await response.text()
                        logger.warning(f"Auth attempt {attempt + 1} failed: {response.status}")
                        
                        if attempt == self.config.max_retries:
                            raise AuthenticationError(f"Auth failed: {response.status} - {error_text}")
                        
                        # Exponential backoff
                        await asyncio.sleep(2 ** attempt)
                        
            except aiohttp.ClientError as e:
                logger.warning(f"Connection error (attempt {attempt + 1}): {e}")
                if attempt == self.config.max_retries:
                    raise AuthenticationError(f"Connection failed: {e}")
                await asyncio.sleep(2 ** attempt)
        
        raise AuthenticationError("Max retries exceeded")
    
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
            if response.status == 200:
                token_data = await response.json()
                self._token = token_data["access_token"]
                
                expires_in = token_data.get("expires_in", 3600)
                self._token_expiry = datetime.now() + timedelta(seconds=expires_in - 60)
                
                logger.info("Successfully authenticated with OKTA")
                return self._token
            else:
                error_text = await response.text()
                raise AuthenticationError(f"OKTA auth failed: {response.status} - {error_text}")
    
    async def generate_response(self, messages: List[Dict[str, str]]) -> str:
        """Generate response using Snowflake Cortex API"""
        try:
            # Get valid token
            token = await self.authenticate()
            
            # Prepare request payload
            payload = {
                "messages": messages,
                "model": self.config.model,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "stream": False
            }
            
            # Prepare headers
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }
            
            # Add optional Snowflake headers
            if self.config.warehouse:
                headers["X-Snowflake-Warehouse"] = self.config.warehouse
            if self.config.role:
                headers["X-Snowflake-Role"] = self.config.role
            
            # Retry logic for API calls
            for attempt in range(self.config.max_retries + 1):
                try:
                    async with self._session.post(
                        self.config.cortex_endpoint,
                        json=payload,
                        headers=headers
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            
                            # Extract response from Cortex API
                            if "choices" not in result or not result["choices"]:
                                raise SnowflakeError("No response choices returned from Cortex API")
                            
                            choice = result["choices"][0]
                            content = choice["message"]["content"]
                            
                            logger.info("Successfully generated response from Cortex")
                            return content
                            
                        else:
                            error_text = await response.text()
                            logger.warning(f"Cortex API attempt {attempt + 1} failed: {response.status}")
                            
                            if attempt == self.config.max_retries:
                                raise SnowflakeError(f"Cortex API failed: {response.status} - {error_text}")
                            
                            await asyncio.sleep(2 ** attempt)
                            
                except aiohttp.ClientError as e:
                    logger.warning(f"Cortex connection error (attempt {attempt + 1}): {e}")
                    if attempt == self.config.max_retries:
                        raise SnowflakeError(f"Cortex connection failed: {e}")
                    await asyncio.sleep(2 ** attempt)
                    
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            raise SnowflakeError(f"Generation failed: {e}")
    
    async def chat(
        self, 
        message: str, 
        session_id: Optional[str] = None,
        system_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """Main chat interface with local session management"""
        try:
            # Generate session ID if not provided
            if not session_id:
                import uuid
                session_id = str(uuid.uuid4())
            
            # Load existing session or create new one
            session_data = self.storage.load_session(session_id)
            
            if session_data:
                # Continue existing conversation
                messages = session_data["messages"]
                logger.info(f"Continuing session {session_id} with {len(messages)} messages")
            else:
                # Start new conversation
                messages = []
                
                # Add system message if provided
                if system_message:
                    messages.append({"role": "system", "content": system_message})
                elif not messages:  # Default system message
                    messages.append({
                        "role": "system", 
                        "content": "You are a helpful AI assistant powered by Snowflake Cortex."
                    })
                
                logger.info(f"Starting new session {session_id}")
            
            # Add user message
            messages.append({"role": "user", "content": message})
            
            # Generate response
            start_time = datetime.now()
            response_content = await self.generate_response(messages)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Add assistant response
            messages.append({"role": "assistant", "content": response_content})
            
            # Save updated session
            self.storage.save_session(session_id, messages)
            
            # Return response info
            return {
                "session_id": session_id,
                "response": response_content,
                "execution_time": execution_time,
                "message_count": len(messages)
            }
            
        except Exception as e:
            logger.error(f"Chat failed: {e}")
            raise SnowflakeError(f"Chat failed: {e}")
    
    def get_session_history(self, session_id: str) -> Optional[List[Dict[str, str]]]:
        """Get chat history for a session"""
        session_data = self.storage.load_session(session_id)
        return session_data["messages"] if session_data else None
    
    def list_sessions(self) -> List[str]:
        """List active session IDs"""
        return self.storage.list_sessions()
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session"""
        return self.storage.delete_session(session_id)
    
    def clear_all_sessions(self):
        """Clear all sessions"""
        self.storage._sessions.clear()
        logger.info("All sessions cleared")

# =============================================================================
# USAGE EXAMPLES
# =============================================================================

async def example_usage():
    """Example usage with environment variables"""
    print("=== Snowflake Cortex Client Example ===")
    
    async with SnowflakeClient() as client:
        # Validate configuration
        issues = client.validate_config()
        if issues:
            print(f"❌ Configuration issues: {issues}")
            print("Please set the required environment variables:")
            print("- SNOWFLAKE_ACCOUNT")
            print("- SNOWFLAKE_USER") 
            print("- SNOWFLAKE_PASSWORD")
            return
        
        print("✅ Configuration valid")
        
        # Simple chat
        print("\n--- Simple Chat ---")
        response = await client.chat("Hello! Can you introduce yourself?")
        print(f"Response: {response['response']}")
        print(f"Session ID: {response['session_id']}")
        print(f"Execution time: {response['execution_time']:.2f}s")
        
        # Continue conversation
        print("\n--- Continue Conversation ---")
        response2 = await client.chat(
            "What can you help me with?",
            session_id=response['session_id']
        )
        print(f"Follow-up: {response2['response']}")
        
        # Show session history
        print("\n--- Session History ---")
        history = client.get_session_history(response['session_id'])
        if history:
            for i, msg in enumerate(history):
                role = msg['role'].upper()
                content = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
                print(f"{i+1}. {role}: {content}")
        
        # List all sessions
        sessions = client.list_sessions()
        print(f"\n--- Active Sessions: {len(sessions)} ---")

async def example_with_system_message():
    """Example with custom system message"""
    print("\n=== Custom System Message Example ===")
    
    async with SnowflakeClient() as client:
        response = await client.chat(
            "Analyze the sales data trends",
            system_message="You are a senior data analyst specializing in sales analytics. "
                          "Provide detailed insights and actionable recommendations."
        )
        print(f"Analyst Response: {response['response']}")

def main():
    """Main function"""
    # Set default environment variables for demo
    os.environ.setdefault("SNOWFLAKE_ACCOUNT", "your-account")
    os.environ.setdefault("SNOWFLAKE_USER", "your-username")
    os.environ.setdefault("SNOWFLAKE_PASSWORD", "your-password")
    
    print("Snowflake Cortex Client - Local Mode")
    print("=" * 40)
    
    try:
        asyncio.run(example_usage())
        asyncio.run(example_with_system_message())
        print("\n✅ Examples completed successfully!")
        
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()
