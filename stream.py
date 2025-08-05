import asyncio
from typing import Dict, Any, List, TypedDict, AsyncGenerator
from langgraph.graph import StateGraph, END
import re

# Define the state
class GraphState(TypedDict):
    messages: List[str]
    current_step: str
    tool_results: Dict[str, Any]
    should_continue: bool
    user_input: str
    streaming_updates: List[str]  # NEW: For internal streaming

# Custom streaming function
async def stream_update(message: str, state: GraphState = None):
    """Stream internal node operations in real-time"""
    print(f"⚡ {message}")
    if state:
        state["streaming_updates"].append(message)
    await asyncio.sleep(0.3)  # Simulate processing time

# Define 4 tools with internal streaming
async def weather_tool_with_streaming(location: str = "New York") -> str:
    """Get weather information with internal streaming"""
    await stream_update(f"🌐 Connecting to weather service...")
    await stream_update(f"📍 Looking up location: {location}")
    await stream_update(f"🌡️ Fetching temperature data...")
    await stream_update(f"☁️ Checking cloud conditions...")
    result = f"🌤️ Weather in {location}: Sunny, 75°F, light breeze"
    await stream_update(f"✅ Weather data retrieved successfully")
    return result

async def calculator_tool_with_streaming(expression: str) -> str:
    """Calculate with internal streaming"""
    await stream_update(f"🔢 Parsing expression: {expression}")
    await stream_update(f"🧮 Validating mathematical syntax...")
    await stream_update(f"⚙️ Performing calculation...")
    
    try:
        allowed_chars = set('0123456789+-*/(). ')
        if all(c in allowed_chars for c in expression):
            result = eval(expression)
            await stream_update(f"✅ Calculation completed: {result}")
            return f"🔢 Calculation result: {expression} = {result}"
        else:
            await stream_update(f"❌ Invalid characters detected")
            return "❌ Error: Only basic math operations allowed"
    except Exception as e:
        await stream_update(f"❌ Calculation failed: {str(e)}")
        return f"❌ Calculation error: {str(e)}"

async def search_tool_with_streaming(query: str) -> str:
    """Search with internal streaming"""
    await stream_update(f"🔍 Initializing search engine...")
    await stream_update(f"🔎 Searching for: '{query}'")
    await stream_update(f"📊 Analyzing search results...")
    await stream_update(f"🏆 Ranking by relevance...")
    await stream_update(f"📝 Formatting results...")
    return f"🔍 Search results for '{query}':\n• Top AI developments in 2024\n• 5 related articles found\n• Latest trends and insights"

async def translator_tool_with_streaming(text: str, target_language: str = "Spanish") -> str:
    """Translate with internal streaming"""
    await stream_update(f"🌐 Loading translation model...")
    await stream_update(f"📝 Analyzing text: '{text}'")
    await stream_update(f"🔄 Translating to {target_language}...")
    await stream_update(f"✅ Translation completed")
    
    translations = {
        "hello": "hola", "goodbye": "adiós", "thank you": "gracias",
        "good morning": "buenos días", "how are you": "¿cómo estás?"
    }
    translated = translations.get(text.lower(), f"[Translated to {target_language}]: {text}")
    return f"🌐 Translation: '{text}' → '{translated}'"

# Tool registry with streaming
tools_with_streaming = {
    "weather": weather_tool_with_streaming,
    "calculator": calculator_tool_with_streaming, 
    "search": search_tool_with_streaming,
    "translator": translator_tool_with_streaming
}

# Define nodes with INTERNAL real-time streaming
async def input_processor_node_streaming(state: GraphState) -> GraphState:
    """Node 1: Process input with real-time internal streaming"""
    await stream_update("🔄 Starting input analysis...", state)
    
    user_input = state["user_input"].lower()
    await stream_update(f"📝 Received input: '{state['user_input']}'", state)
    
    await stream_update("🔍 Detecting query type...", state)
    
    if any(word in user_input for word in ["weather", "temperature", "forecast", "climate"]):
        state["current_step"] = "weather"
        await stream_update("🌤️ Detected: Weather query", state)
    elif any(word in user_input for word in ["calculate", "math", "+", "-", "*", "/", "="]):
        state["current_step"] = "calculator"
        await stream_update("🔢 Detected: Math calculation", state)
    elif any(word in user_input for word in ["translate", "spanish", "french", "language"]):
        state["current_step"] = "translator"
        await stream_update("🌐 Detected: Translation request", state)
    else:
        state["current_step"] = "search"
        await stream_update("🔍 Detected: Search query (default)", state)
    
    await stream_update(f"✅ Routing to: {state['current_step']} tool", state)
    state["should_continue"] = True
    return state

async def tool_execution_node_streaming(state: GraphState) -> GraphState:
    """Node 2: Execute tool with real-time internal streaming"""
    await stream_update(f"🛠️ Initializing {state['current_step']} tool...", state)
    
    user_input = state["user_input"]
    tool_name = state["current_step"]
    
    if tool_name == "weather":
        await stream_update("🌍 Extracting location from input...", state)
        location_match = re.search(r'weather.*?in\s+(\w+)', user_input.lower())
        location = location_match.group(1) if location_match else "New York"
        await stream_update(f"📍 Location identified: {location}", state)
        result = await tools_with_streaming["weather"](location)
        
    elif tool_name == "calculator":
        await stream_update("🔢 Extracting mathematical expression...", state)
        math_match = re.search(r'[\d+\-*/\s().]+', user_input)
        expression = math_match.group(0).strip() if math_match else "2+2"
        await stream_update(f"📐 Expression found: {expression}", state)
        result = await tools_with_streaming["calculator"](expression)
        
    elif tool_name == "translator":
        await stream_update("📝 Extracting text to translate...", state)
        quote_match = re.search(r"'([^']+)'", user_input)
        if quote_match:
            text = quote_match.group(1)
        else:
            translate_match = re.search(r'translate\s+(.+?)(?:\s+to|$)', user_input.lower())
            text = translate_match.group(1).strip() if translate_match else "hello"
        await stream_update(f"📖 Text to translate: '{text}'", state)
        result = await tools_with_streaming["translator"](text)
        
    else:  # search
        await stream_update("🔎 Preparing search query...", state)
        result = await tools_with_streaming["search"](user_input)
    
    state["tool_results"][tool_name] = result
    await stream_update("✅ Tool execution completed successfully", state)
    return state

async def response_generator_node_streaming(state: GraphState) -> GraphState:
    """Node 3: Generate response with real-time internal streaming"""
    await stream_update("📝 Starting response generation...", state)
    
    await stream_update("📊 Retrieving tool results...", state)
    tool_result = state["tool_results"].get(state["current_step"], "No result")
    
    await stream_update("✍️ Formatting response...", state)
    response = f"Here's what I found for your request:\n\n{tool_result}"
    
    await stream_update("💾 Storing response in messages...", state)
    state["messages"].append(response)
    
    await stream_update("✅ Response generation completed", state)
    return state

async def finalizer_node_streaming(state: GraphState) -> GraphState:
    """Node 4: Finalize with real-time internal streaming"""
    await stream_update("🏁 Starting workflow finalization...", state)
    
    await stream_update("🔄 Updating workflow status...", state)
    state["should_continue"] = False
    state["current_step"] = "completed"
    
    await stream_update("📈 Workflow statistics:", state)
    await stream_update(f"   • Total streaming updates: {len(state['streaming_updates'])}", state)
    await stream_update(f"   • Tool used: {list(state['tool_results'].keys())[0] if state['tool_results'] else 'None'}", state)
    
    await stream_update("✅ Workflow completed successfully", state)
    return state

# Routing functions (same as before)
def route_after_input(state: GraphState) -> str:
    if state["current_step"] in ["weather", "calculator", "translator", "search"]:
        return "tool_execution"
    return "finalizer"

def route_after_tool(state: GraphState) -> str:
    if state["current_step"] in state["tool_results"]:
        return "response_generator"
    return "finalizer"

def route_after_response(state: GraphState) -> str:
    return "finalizer"

# Create workflow with streaming nodes
def create_streaming_workflow():
    workflow = StateGraph(GraphState)
    
    # Use the streaming versions of nodes
    workflow.add_node("input_processor", input_processor_node_streaming)
    workflow.add_node("tool_execution", tool_execution_node_streaming)
    workflow.add_node("response_generator", response_generator_node_streaming)
    workflow.add_node("finalizer", finalizer_node_streaming)
    
    workflow.set_entry_point("input_processor")
    
    workflow.add_conditional_edges(
        "input_processor", route_after_input,
        {"tool_execution": "tool_execution", "finalizer": "finalizer"}
    )
    
    workflow.add_conditional_edges(
        "tool_execution", route_after_tool,
        {"response_generator": "response_generator", "finalizer": "finalizer"}
    )
    
    workflow.add_conditional_edges(
        "response_generator", route_after_response,
        {"finalizer": "finalizer"}
    )
    
    workflow.add_edge("finalizer", END)
    return workflow.compile()

# TRUE REAL-TIME STREAMING DEMO
async def true_realtime_streaming_demo(user_input: str):
    """Demo showing TRUE internal real-time streaming"""
    print("🎯 TRUE REAL-TIME STREAMING DEMO")
    print("=" * 60)
    print(f"🚀 Processing: '{user_input}'")
    print("-" * 60)
    
    app = create_streaming_workflow()
    initial_state = GraphState(
        messages=[], current_step="start", tool_results={}, 
        should_continue=True, user_input=user_input,
        streaming_updates=[]  # Track all updates
    )
    
    # This will show INTERNAL operations in real-time
    async for output in app.astream(initial_state):
        for node_name, state_update in output.items():
            print(f"\n📍 NODE COMPLETED: {node_name}")
            
            # Show final response when available
            if "messages" in state_update and state_update["messages"]:
                print(f"\n💬 FINAL RESPONSE:")
                print(state_update["messages"][-1])
    
    print("\n🏁 STREAMING COMPLETED!")
    print("=" * 60)

# Comparison demo
async def streaming_comparison():
    """Compare regular vs real-time streaming"""
    query = "What's the weather in Tokyo?"
    
    print("COMPARISON: Regular vs True Real-Time Streaming")
    print("=" * 70)
    
    await true_realtime_streaming_demo(query)

if __name__ == "__main__":
    asyncio.run(streaming_comparison())
