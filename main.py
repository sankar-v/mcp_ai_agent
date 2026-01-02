from fastapi import FastAPI
from langchain_mcp_adapters.client import MultiServerMCPClient
from fastmcp import FastMCP
from fastmcp.server.event_store import EventStore
from key_value.aio.stores.redis import RedisStore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from typing import TypedDict
import httpx
import wikipedia
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastMCP instance
mcp = FastMCP("agent")

# Define state for the graph
class EnrichmentState(TypedDict):
    messages: list

# Initialize FastAPI app (will be connected to mcp_app later)
app = FastAPI()

# Initialize OpenAI LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Redis store and event store setup
redis_store = RedisStore(url="redis://localhost:6379")

event_store = EventStore(
    storage=redis_store,
    max_events_per_stream=100,
    ttl=3600,
)

# MCP client for connecting to the agent
client = MultiServerMCPClient({
    "agent": {
        "transport": "http",
        "url": "http://localhost:8000/agent/mcp",
    },
})

# Function to create the LangGraph workflow
async def create_graph(session, llm):
    """Create the agent graph with tools and prompts from MCP session"""
    from langchain_mcp_adapters.client import load_mcp_tools, load_mcp_prompt
    
    tools = await load_mcp_tools(session)

    system_prompt = await load_mcp_prompt(
        session=session,
        name="common_prompt"
    )

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt[0].content),
        MessagesPlaceholder("messages")
    ])

    llm_with_tool = llm.bind_tools(tools)
    chat_llm = prompt_template | llm_with_tool

    # Define a simple chat node function
    async def chat_node(state: EnrichmentState):
        response = await chat_llm.ainvoke(state["messages"])
        return {"messages": [response]}

    # Define tools condition
    def tools_condition(state: EnrichmentState):
        last_message = state["messages"][-1]
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools"
        return "__end__"

    graph = StateGraph(EnrichmentState)

    graph.add_node("chat_node", chat_node)
    graph.add_node("tool_node", ToolNode(tools=tools))
    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges(
        "chat_node",
        tools_condition,
        {"tools": "tool_node", "__end__": END}
    )
    graph.add_edge("tool_node", "chat_node")
    compiled_graph = graph.compile(checkpointer=MemorySaver())
    return compiled_graph

@app.post("/workflow")
async def run_workflow(message: str):
    """Run the agent workflow with a user message"""
    config = {"configurable": {"thread_id": "001"}}

    async with client.session("agent") as session:
        agent = await create_graph(session=session, llm=llm)
        response = await agent.ainvoke(
            {"messages": [message]},
            config=config
        )
        return response["messages"][-1].content

# Wikipedia Tool
@mcp.tool(
    name="global_news",
    description="Get global news from Wikipedia"
)
async def global_news(query: str):
    return wikipedia.summary(query)

# Country Details Tool
@mcp.tool(
    name="get_countries_details",
    description="Get details of a country"
)
async def get_countries_details(country_name: str):
    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.get(
            f"https://restcountries.com/v3.1/name/{country_name}?fullText=true"
        )
        response.raise_for_status()
        return response.json()

# Common Prompt
@mcp.prompt
async def common_prompt() -> str:
    return """
    You are a helpful assistant.
    Answer the question based on the tools provided.
    """

# Create MCP app with event store
def create_mcp_app():
    """Create and configure the MCP application"""
    return mcp.http_app(
        event_store=event_store,
        path="/mcp"
    )

# Create the MCP app
mcp_app = create_mcp_app()

# Mount MCP app to FastAPI
app.mount("/agent", mcp_app)

# Main entry point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)