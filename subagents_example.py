from langchain.tools import tool
from langchain.agents import create_agent
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()


from langfuse.langchain import CallbackHandler

langfuse_handler = CallbackHandler()

model = init_chat_model(
    "gpt-5", temperature=0, api_key=os.environ.get("OPENAI_API_KEY")
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
@tool("calculator", description="Perform arithmetic operations")
def call_calculator_agent(query: str):
    result = subagent.invoke({"messages": [{"role": "user", "content": query}]})
    return result["messages"][-1].content


# Main agent with subagent as a tool
main_agent = create_agent(model=model, tools=[call_calculator_agent])
result = main_agent.invoke(
    {"messages": [{"role": "user", "content": "Add 3 and 4."}]},
    config={"callbacks": [langfuse_handler]},
)
print(result)
