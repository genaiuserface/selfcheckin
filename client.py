"""
Clean Snowflake Cortex client with authentication and session management
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
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
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

class SessionError(SnowflakeError):
    """Session management error"""
    pass

# =============================================================================
# CONFIGURATION
# =============================================================================

class SnowflakeConfig(BaseModel):
    """Snowflake configuration"""
    account: str = Field(..., description="Snowflake account")
    user: str = Field(..., description="Username")
    password: SecretStr = Field(..., description="Password")
    warehouse: Optional[str] = Field(None, description="Warehouse")
    role: Optional[str] = Field(None, description="Role")
    database: Optional[str] = Field(None, description="Database")
    schema: Optional[str] = Field(None, description="Schema")
    
    # Authentication
    authenticator: str = Field(default="snowflake", description="Auth method")
    okta_endpoint: Optional[str] = Field(None, description="OKTA endpoint")
    
    # API settings
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
        """Load from environment variables"""
        return cls(
            account=os.getenv("SNOWFLAKE_ACCOUNT", ""),
            user=os.getenv("SNOWFLAKE_USER", ""),
            password=SecretStr(os.getenv("SNOWFLAKE_PASSWORD", "")),
            warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
            role=os.getenv("SNOWFLAKE_ROLE"),
            database=os.getenv("SNOWFLAKE_DATABASE"),
            schema=os.getenv("SNOWFLAKE_SCHEMA"),
            authenticator=os.getenv("SNOWFLAKE_AUTHENTICATOR", "snowflake"),
            okta_endpoint=os.getenv("SNOWFLAKE_OKTA_ENDPOINT"),
            model=os.getenv("SNOWFLAKE_MODEL", "mistral-large"),
            temperature=float(os.getenv("SNOWFLAKE_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("SNOWFLAKE_MAX_TOKENS", "4096")),
            timeout=int(os.getenv("SNOWFLAKE_TIMEOUT", "60")),
            max_retries=int(os.getenv("SNOWFLAKE_MAX_RETRIES", "3"))
        )

# =============================================================================
# SESSION DATA MODEL
# =============================================================================

@dataclass
class ChatSession:
    """Chat session data"""
    session_id: str
    messages: List[Dict[str, str]]
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any]

# =============================================================================
# SESSION MANAGER
# =============================================================================

class SessionManager:
    """Simple session manager with SQLite storage"""
    
    def __init__(self, db_path: str = "sessions.db", session_ttl: int = 86400):
        self.db_path = db_path
        self.session_ttl = session_ttl  # 24 hours default
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite database"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    messages TEXT NOT NULL,
                    metadata TEXT DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_expires ON sessions(expires_at)")
    
    def save_session(self, session: ChatSession):
        """Save session to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                expires_at = datetime.now() + timedelta(seconds=self.session_ttl)
                
                conn.execute("""
                    INSERT OR REPLACE INTO sessions 
                    (session_id, messages, metadata, updated_at, expires_at)
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP, ?)
                """, (
                    session.session_id,
                    json.dumps(session.messages),
                    json.dumps(session.metadata),
                    expires_at.isoformat()
                ))
                
                # Cleanup expired sessions
                conn.execute("DELETE FROM sessions WHERE expires_at < CURRENT_TIMESTAMP")
                
        except Exception as e:
            logger.error(f"Failed to save session {session.session_id}: {e}")
            raise SessionError(f"Save failed: {e}")
    
    def load_session(self, session_id: str) -> Optional[ChatSession]:
        """Load session from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT messages, metadata, created_at, updated_at
                    FROM sessions 
                    WHERE session_id = ? AND expires_at > CURRENT_TIMESTAMP
                """, (session_id,))
                
                row = cursor.fetchone()
                if not row:
                    return None
                
                return ChatSession(
                    session_id=session_id,
                    messages=json.loads(row[0]),
                    metadata=json.loads(row[1]),
                    created_at=datetime.fromisoformat(row[2]),
                    updated_at=datetime.fromisoformat(row[3])
                )
                
        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            return None
    
    def delete_session(self, session_id: str) -> bool:
        """Delete session"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")
            return False
    
    def list_sessions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List recent sessions"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT session_id, created_at, updated_at
                    FROM sessions 
                    WHERE expires_at > CURRENT_TIMESTAMP
                    ORDER BY updated_at DESC 
                    LIMIT ?
                """, (limit,))
                
                return [
                    {
                        "session_id": row[0],
                        "created_at": row[1],
                        "updated_at": row[2]
                    }
                    for row in cursor.fetchall()
                ]
        except Exception as e:
            logger.error(f"Failed to list sessions: {e}")
            return []

# =============================================================================
# SNOWFLAKE CLIENT
# =============================================================================

class SnowflakeClient:
    """Clean Snowflake Cortex client with authentication and sessions"""
    
    def __init__(self, config: Optional[SnowflakeConfig] = None):
        self.config = config or SnowflakeConfig.from_env()
        self.session_manager = SessionManager()
        self._session = None
        self._token = None
        self._token_expiry = None
        
        logger.info(f"Initialized Snowflake client for {self.config.account}")
    
    async def __aenter__(self):
        """Async context manager entry"""
        connector = aiohttp.TCPConnector(limit=100, keepalive_timeout=30)
        timeout = aiohttp.ClientTimeout(total=self.config.timeout)
        
        self._session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout
        )
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self._session:
            await self._session.close()
    
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
            raise AuthenticationError(f"Auth failed: {e}")
    
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
                    if response.status == 200:
                        token_data = await response.json()
                        self._token = token_data["access_token"]
                        self._token_expiry = datetime.now() + timedelta(
                            seconds=token_data.get("expires_in", 3600) - 60
                        )
                        logger.info("Successfully authenticated with Snowflake")
                        return self._token
                    else:
                        error_text = await response.text()
                        if attempt == self.config.max_retries:
                            raise AuthenticationError(f"Auth failed: {response.status} - {error_text}")
                        await asyncio.sleep(2 ** attempt)
                        
            except aiohttp.ClientError as e:
                if attempt == self.config.max_retries:
                    raise AuthenticationError(f"Connection failed: {e}")
                await asyncio.sleep(2 ** attempt)
    
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
                self._token_expiry = datetime.now() + timedelta(
                    seconds=token_data.get("expires_in", 3600) - 60
                )
                logger.info("Successfully authenticated with OKTA")
                return self._token
            else:
                error_text = await response.text()
                raise AuthenticationError(f"OKTA auth failed: {response.status} - {error_text}")
    
    async def generate_response(self, messages: List[Dict[str, str]]) -> str:
        """Generate response using Snowflake Cortex"""
        try:
            token = await self.authenticate()
            
            payload = {
                "messages": messages,
                "model": self.config.model,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "stream": False
            }
            
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }
            
            if self.config.warehouse:
                headers["X-Snowflake-Warehouse"] = self.config.warehouse
            if self.config.role:
                headers["X-Snowflake-Role"] = self.config.role
            
            for attempt in range(self.config.max_retries + 1):
                try:
                    async with self._session.post(
                        self.config.cortex_endpoint,
                        json=payload,
                        headers=headers
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            
                            if "choices" not in result or not result["choices"]:
                                raise SnowflakeError("No response choices returned")
                            
                            return result["choices"][0]["message"]["content"]
                        else:
                            error_text = await response.text()
                            if attempt == self.config.max_retries:
                                raise SnowflakeError(f"API failed: {response.status} - {error_text}")
                            await asyncio.sleep(2 ** attempt)
                            
                except aiohttp.ClientError as e:
                    if attempt == self.config.max_retries:
                        raise SnowflakeError(f"Connection failed: {e}")
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
        """Main chat interface with session management"""
        try:
            # Generate session ID if not provided
            if not session_id:
                session_id = str(uuid.uuid4())
            
            # Load existing session or create new one
            session = self.session_manager.load_session(session_id)
            
            if session:
                # Continue existing conversation
                messages = session.messages
                logger.info(f"Continuing session {session_id} with {len(messages)} messages")
            else:
                # Start new conversation
                messages = []
                if system_message:
                    messages.append({"role": "system", "content": system_message})
                
                session = ChatSession(
                    session_id=session_id,
                    messages=messages,
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                    metadata={}
                )
                logger.info(f"Starting new session {session_id}")
            
            # Add user message
            messages.append({"role": "user", "content": message})
            
            # Generate response
            start_time = datetime.now()
            response_content = await self.generate_response(messages)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Add assistant response
            messages.append({"role": "assistant", "content": response_content})
            
            # Update session
            session.messages = messages
            session.updated_at = datetime.now()
            
            # Save session
            self.session_manager.save_session(session)
            
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
        session = self.session_manager.load_session(session_id)
        return session.messages if session else None
    
    def list_sessions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List recent sessions"""
        return self.session_manager.list_sessions(limit)
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session"""
        return self.session_manager.delete_session(session_id)
    
    def validate_config(self) -> List[str]:
        """Validate configuration"""
        issues = []
        
        if not self.config.account:
            issues.append("Account is required")
        if not self.config.user:
            issues.append("User is required")
        if not self.config.password.get_secret_value():
            issues.append("Password is required")
        
        return issues

# =============================================================================
# USAGE EXAMPLE
# =============================================================================

async def example_usage():
    """Example usage"""
    
    # Option 1: Use environment variables
    async with SnowflakeClient() as client:
        # Validate configuration
        issues = client.validate_config()
        if issues:
            print(f"Config issues: {issues}")
            return
        
        # Simple chat
        response = await client.chat("Hello, how are you?")
        print(f"Response: {response['response']}")
        print(f"Session: {response['session_id']}")
        
        # Continue conversation
        response2 = await client.chat(
            "What's the weather like?",
            session_id=response['session_id']
        )
        print(f"Follow-up: {response2['response']}")
        
        # Get history
        history = client.get_session_history(response['session_id'])
        print(f"History has {len(history)} messages")

# Option 2: Direct configuration
async def example_with_config():
    """Example with direct configuration"""
    config = SnowflakeConfig(
        account="your-account",
        user="your-username", 
        password=SecretStr("your-password"),
        warehouse="your-warehouse",
        model="mistral-large"
    )
    
    async with SnowflakeClient(config) as client:
        response = await client.chat(
            "Analyze this data for me",
            system_message="You are a helpful data analyst."
        )
        print(f"Response: {response['response']}")

if __name__ == "__main__":
    # Set environment variables for testing
    os.environ.setdefault("SNOWFLAKE_ACCOUNT", "your-account")
    os.environ.setdefault("SNOWFLAKE_USER", "your-username")
    os.environ.setdefault("SNOWFLAKE_PASSWORD", "your-password")
    
    # Run example
    asyncio.run(example_usage())
