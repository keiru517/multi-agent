from langchain.tools import tool
from langchain.agents import create_agent
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage

load_dotenv()


from langfuse.langchain import CallbackHandler

langfuse_handler = CallbackHandler()

model = init_chat_model(
    "gpt-5",
    temperature=0,
    api_key=os.environ.get("OPENAI_API_KEY"),
)


# Define tools
@tool
def multiply(a: int, b: int) -> int:
    """Multiply `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Adds `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a + b


@tool
def divide(a: int, b: int) -> float:
    """Divide `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a / b


tools = [multiply, add, divide]

# Create a subagent
subagent = create_agent(model=model, tools=tools)


# Wrap it as a tool
@tool("generalist", description="uses LLM general knowledge")
def call_generalist_agent(query: str):
    result = model.invoke(
        [
            SystemMessage(
                content="You are a generalist agent. You are given a query and you need to use your general knowledge to answer the query."
            ),
            HumanMessage(content=query),
        ]
    )
    return result["messages"][-1].content


# @tool("knowledge_assistant", description="searches internal documents")
# def call_knowledge_assistant_agent(query: str):
#     result = subagent.invoke({"messages": [{"role": "user", "content": query}]})
#     return result["messages"][-1].content


@tool("calculator", description="arithmetic operations")
def call_calculator_agent(query: str):
    result = subagent.invoke({"messages": [{"role": "user", "content": query}]})
    return result["messages"][-1].content


@tool("weather", description="weather information")
def call_weather_agent(query: str):
    return "The weather is sunny."


# Main agent with subagent as a tool
main_agent = create_agent(
    model=model,
    tools=[call_generalist_agent, call_calculator_agent, call_weather_agent],
    system_prompt=(
        "You are a supervisor managing three agents.\n"
        "The first agent is a generalist agent and the second agent is a calculator agent. \n"
        "The generalist agent is responsible for using its general knowledge "
        "to answer the user's question. The calculator agent is responsible for performing "
        "arithmetic operations. The weather agent is responsible for providing weather information."
    ),
    name="main_agent",
)

# png_bytes = main_agent.get_graph().draw_mermaid_png()
# with open("graph_visualization.png", "wb") as f:
#     f.write(png_bytes)
# print("Graph visualization saved to graph_visualization.png")

query = "What is the weather in Tokyo?"
result = main_agent.invoke(
    {"messages": [{"role": "user", "content": query}]},
    config={"callbacks": [langfuse_handler]},
)
print(result)
