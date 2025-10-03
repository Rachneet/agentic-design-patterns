from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode

import os
import asyncio
from dotenv import load_dotenv

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")  # Ensure your Groq API key is set # type: ignore


async def main():
    """
    Sets up and runs a language model integrated with multiple MCP tools.
    """

    # Initialize the model
    model = init_chat_model(
        model="qwen/qwen3-32b",  # or another model of your choice
        model_provider="groq",
    )

    # Set up MCP client
    client = MultiServerMCPClient(
        {
            # "math": {
            #     "command": "python",
            #     # Make sure to update to the full absolute path to your math_server.py file
            #     "args": ["./examples/math_server.py"],
            #     "transport": "stdio",
            # },
            # "weather": {
            #     # make sure you start your weather server on port 8000
            #     "url": "http://localhost:8000/mcp/",
            #     "transport": "streamable_http",
            # },
            "greet": {
                # make sure you start your greet server on port 8000
                "command": "/opt/miniconda3/bin/uv",
                "args": ["run", "server.py"],
                "transport": "stdio",
            }
        }
    )

    tools = await client.get_tools()  # type: ignore

    # Bind tools to model
    model_with_tools = model.bind_tools(tools)

    # Create ToolNode
    tool_node = ToolNode(tools)

    def should_continue(state: MessagesState):
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:  # type: ignore
            return "tools"
        return END

    # Define call_model function
    async def call_model(state: MessagesState):
        messages = state["messages"]
        response = await model_with_tools.ainvoke(messages)
        return {"messages": [response]}

    # Build the graph
    builder = StateGraph(MessagesState)
    builder.add_node("call_model", call_model)
    builder.add_node("tools", tool_node)

    builder.add_edge(START, "call_model")
    builder.add_conditional_edges(
        "call_model",
        should_continue,
    )
    builder.add_edge("tools", "call_model")

    # Compile the graph
    graph = builder.compile()

    # Test the graph
    # math_response = await graph.ainvoke(
    #     {"messages": [{"role": "user", "content": "what's (3 + 5) x 12?"}]}  # type: ignore
    # )
    # weather_response = await graph.ainvoke(
    #     {"messages": [{"role": "user", "content": "what is the weather in nyc?"}]}  # type: ignore
    # )

    greet_response = await graph.ainvoke(  # type: ignore
        {"messages": [{"role": "user", "content": "Greet Rachneet"}]}  # type: ignore
    )

    print("Greet Response:", greet_response["messages"][-1].content)  # type: ignore


if __name__ == "__main__":
    asyncio.run(main())
