# ===================================================================
# COMPLETE LLMClient CLASS - FINAL VERSION
# Simple, clean, and production-ready with tool extraction
# ===================================================================

import logging
import time
import asyncio
import json
import aiohttp
import re

class LLMClient:
    """
    Simple, production-ready LLM client with tool extraction
    Handles: LLM API calls, retries, token management, tool parsing
    """
    
    def __init__(self, session_manager, host: str, model: str = "claude-4-sonnet", temperature: float = 0.2, max_tokens: int = 1024):
        """Initialize LLM client with essential configuration"""
        self.session_manager = session_manager
        self.host = host
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize components
        self.security_filter = PromptSecurityFilter()
        self.prompt_manager = SystemPromptManager()
        self.prompt_manager.set_role("merge_helper")  # Your default role
        
        logging.info(f"LLMClient initialized: {model} @ {host}")

    def _get_token(self):
        """Get valid token with simple retry logic"""
        for attempt in range(3):
            try:
                conn = self.session_manager.get_connection()
                
                if not conn or not self.session_manager.is_session_active():
                    logging.info(f"Reconnecting session (attempt {attempt + 1})")
                    self.session_manager.connect()
                    conn = self.session_manager.get_connection()
                
                if conn and self.session_manager.is_session_active():
                    token = getattr(conn.rest, "token", None)
                    if token and len(str(token)) > 10:
                        return token
                
                if attempt < 2:  # Don't sleep on last attempt
                    time.sleep(1)
                    
            except Exception as e:
                logging.error(f"Token attempt {attempt + 1} failed: {e}")
                if attempt < 2:
                    time.sleep(1)
        
        raise RuntimeError("Failed to get valid token after 3 attempts")

    def extract_tool_calls(self, text: str) -> list:
        """
        Extract tool calls from LLM response
        Simple and reliable parsing for your LangGraph workflow
        """
        if not text or "TOOL_CALL:" not in text:
            return []
        
        tool_calls = []
        
        # Pattern 1: TOOL_CALL:tool_name:{"arg": "value"}
        pattern1 = r"TOOL_CALL:([a-zA-Z0-9_]+):\s*(\{[^}]+\})"
        matches1 = re.findall(pattern1, text, re.DOTALL)
        
        for tool_name, args_str in matches1:
            try:
                arguments = json.loads(args_str)
                tool_calls.append({"name": tool_name, "arguments": arguments})
            except json.JSONDecodeError:
                logging.warning(f"Failed to parse tool arguments: {args_str}")
                tool_calls.append({"name": tool_name, "arguments": {}})
        
        # Pattern 2: TOOL_CALL:tool_name (no arguments)
        pattern2 = r"TOOL_CALL:([a-zA-Z0-9_]+)(?!\s*:)"
        matches2 = re.findall(pattern2, text)
        
        for tool_name in matches2:
            # Only add if not already added by pattern1
            if not any(tc["name"] == tool_name for tc in tool_calls):
                tool_calls.append({"name": tool_name, "arguments": {}})
        
        if tool_calls:
            logging.info(f"Extracted {len(tool_calls)} tool calls: {[tc['name'] for tc in tool_calls]}")
        
        return tool_calls

    def has_tool_calls(self, text: str) -> bool:
        """Quick check if response contains tool calls"""
        return "TOOL_CALL:" in text if text else False

    async def call_with_context_async(self, messages: list, tools_description: str = "", 
                                    stream_update_func=None, role: str = None) -> dict:
        """
        Main method: Build prompt and call LLM with tool extraction
        Returns: {"response": str, "tool_calls": list, "has_tools": bool}
        """
        try:
            # Input validation
            if not messages or not isinstance(messages, list):
                return {"response": "Invalid messages provided", "tool_calls": [], "has_tools": False}
            
            # Security check and sanitization
            safe_tools = self.security_filter.sanitize(tools_description)
            
            # Build system prompt
            system_prompt = self.prompt_manager.build_prompt(
                role=role,
                tools_description=safe_tools
            )
            
            # Call LLM
            llm_response = await self.call_llm(messages, system_prompt, stream_update_func)
            
            # Extract tool calls
            tool_calls = self.extract_tool_calls(llm_response)
            
            return {
                "response": llm_response,
                "tool_calls": tool_calls,
                "has_tools": len(tool_calls) > 0
            }
            
        except Exception as e:
            logging.error(f"Error in call_with_context_async: {e}")
            return {
                "response": "Sorry, I encountered an error. Please try again.",
                "tool_calls": [],
                "has_tools": False
            }

    async def call_llm(self, messages: list, system_prompt: str, stream_update_func=None) -> str:
        """
        Core LLM API call with retry logic
        Handles both streaming and non-streaming responses
        """
        for attempt in range(3):
            try:
                # Get token
                token = self._get_token()
                
                # Prepare request
                request_body = {
                    "model": self.model,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "messages": [{"role": "system", "content": system_prompt}] + messages
                }
                
                headers = {
                    "Authorization": f"Snowflake Token={token}",
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                }
                
                url = f"https://{self.host}/api/v2/cortex/inference/complete"
                timeout = aiohttp.ClientTimeout(total=30)
                
                # Make request
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(url, json=request_body, headers=headers) as resp:
                        
                        # Handle response status
                        if resp.status == 401:
                            logging.warning("Authentication failed")
                            if attempt < 2:
                                continue  # Retry
                            return "Authentication failed. Please check credentials."
                        
                        elif resp.status == 429:
                            logging.warning("Rate limited")
                            if attempt < 2:
                                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                                continue
                            return "Service busy. Please try again later."
                        
                        elif resp.status >= 500:
                            logging.warning(f"Server error: {resp.status}")
                            if attempt < 2:
                                await asyncio.sleep(2 ** attempt)
                                continue
                            return "Server error. Please try again later."
                        
                        elif resp.status != 200:
                            return f"Unexpected response: {resp.status}"
                        
                        # Process successful response
                        content_type = resp.headers.get("Content-Type", "")
                        
                        if content_type.startswith("text/event-stream"):
                            # Streaming response
                            return await self._handle_streaming_response(resp, stream_update_func)
                        else:
                            # Non-streaming response
                            return await self._handle_json_response(resp)
                
            except asyncio.TimeoutError:
                logging.error(f"Request timeout (attempt {attempt + 1})")
                if attempt < 2:
                    await asyncio.sleep(1)
                    continue
            
            except Exception as e:
                logging.error(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt < 2:
                    await asyncio.sleep(1)
                    continue
        
        return "Failed to get response after 3 attempts. Please try again."

    async def _handle_streaming_response(self, resp, stream_update_func):
        """Handle streaming response"""
        full_response = ""
        try:
            async for line in resp.content:
                if not line:
                    continue
                
                line_content = line.decode("utf-8")
                if line_content.startswith("data:"):
                    try:
                        data = json.loads(line_content.replace("data: ", ""))
                        if "choices" in data and "delta" in data["choices"][0]:
                            delta = data["choices"][0]["delta"]
                            if "delta" in delta and "content" in delta:
                                chunk = delta["content"]
                                full_response += chunk
                                if stream_update_func:
                                    await stream_update_func(f"[REASONING] {chunk}")
                    except:
                        continue  # Skip malformed chunks
            
            return full_response.strip() or "Empty response received"
            
        except Exception as e:
            logging.error(f"Streaming error: {e}")
            return "Error processing streaming response"

    async def _handle_json_response(self, resp):
        """Handle JSON response"""
        try:
            raw_text = await resp.text()
            data = json.loads(raw_text)
            
            if "choices" in data and data["choices"] and "message" in data["choices"][0]:
                content = data["choices"][0]["message"]["content"]
                return content.strip() if content else "Empty response received"
            else:
                return "Invalid response format"
                
        except json.JSONDecodeError:
            return "Invalid JSON response"
        except Exception as e:
            logging.error(f"JSON parsing error: {e}")
            return "Error parsing response"

    # BACKWARD COMPATIBILITY: Keep the old interface working
    async def call_with_context_async_simple(self, messages: list, tools_description: str = "", 
                                           stream_update_func=None, role: str = None) -> str:
        """Old interface for backward compatibility"""
        result = await self.call_with_context_async(messages, tools_description, stream_update_func, role)
        return result["response"]

    # Convenience methods for different roles
    async def call_as_merge_helper(self, messages: list, tools_description: str = "", stream_update_func=None):
        """Call as merge request helper (your main use case)"""
        return await self.call_with_context_async(messages, tools_description, stream_update_func, "merge_helper")
    
    async def call_as_code_reviewer(self, messages: list, tools_description: str = "", stream_update_func=None):
        """Call as code reviewer"""
        return await self.call_with_context_async(messages, tools_description, stream_update_func, "code_reviewer")
    
    async def call_as_general_assistant(self, messages: list, tools_description: str = "", stream_update_func=None):
        """Call as general assistant"""
        return await self.call_with_context_async(messages, tools_description, stream_update_func, "general")
    
    def set_role(self, role: str):
        """Change default role"""
        self.prompt_manager.set_role(role)
    
    def add_custom_role(self, name: str, base_prompt: str, instructions: str):
        """Add custom role easily"""
        return self.prompt_manager.add_custom_role(name, base_prompt, instructions)


# ===================================================================
# USAGE EXAMPLES - HOW TO USE THIS LLMClient
# ===================================================================

"""
🚀 USAGE EXAMPLES:

1. BASIC USAGE (your existing code works):
   result = await client.call_with_context_async(messages, tools_description)
   response = result["response"]
   tool_calls = result["tool_calls"]

2. BACKWARD COMPATIBILITY:
   response = await client.call_with_context_async_simple(messages, tools_description)

3. CONVENIENCE METHODS:
   result = await client.call_as_merge_helper(messages, tools_description)
   result = await client.call_as_code_reviewer(messages, tools_description)

4. ROLE MANAGEMENT:
   client.set_role("code_reviewer")
   client.add_custom_role("db_helper", "You help with databases", "- Optimize SQL")

5. TOOL EXTRACTION:
   result = await client.call_with_context_async(messages, tools_description)
   if result["has_tools"]:
       for tool in result["tool_calls"]:
           print(f"Tool: {tool['name']}, Args: {tool['arguments']}")

6. INTEGRATION WITH LANGGRAPH:
   # In your assistant_node:
   result = await llm_client.call_with_context_async(messages, tools_description)
   ai_response = result["response"]
   extracted_tools = result["tool_calls"]
   
   if extracted_tools:
       # Pass to your tools_node
       for tool in extracted_tools:
           # Your existing tool execution logic
           pass
"""
