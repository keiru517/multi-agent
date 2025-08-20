import os

from langchain_openai import ChatOpenAI

from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent

# Create specialized agents

model = ChatOpenAI(
    model="gpt-4o",
    openai_api_key=os.environ.get("OPENAI_API_KEY"),
)


def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


def btc_price_search(query: str) -> str:
    """
    Use this tool for getting the current price of BTC
    Search the web for information."""
    return "The current price of BTC is 100 USD"


math_agent = create_react_agent(
    model=model,
    tools=[add],
    name="math_expert",
)

btc_price_agent = create_react_agent(
    model=model,
    tools=[btc_price_search],
    name="btc_price_search_expert",
)

# Create supervisor workflow
workflow = create_supervisor(
    [btc_price_agent, math_agent],
    model=model,
    add_handoff_back_messages=True,
    output_mode="full_history",
)

# Compile and run
app = workflow.compile()


def pretty_print_messages(chunk, last_message=False):
    for node, data in chunk.items():
        if "messages" in data:
            for msg in data["messages"]:
                role = getattr(msg, "type", None) or getattr(msg, "role", "unknown")

                # Skip human messages entirely
                if role == "human":
                    continue

                content = getattr(msg, "content", "")

                if role == "tool":
                    print(
                        "================================= Tool Message =================================="
                    )
                elif role == "ai":
                    print(
                        "================================== Ai Message =================================="
                    )
                else:
                    print(
                        f"=============================== {role.title()} Message ==============================="
                    )

                name = getattr(msg, "name", None)
                if name:
                    print(f"Name: {name}\n")
                print(content)

            if last_message:
                print("\n")


from typing_extensions import TypedDict


class AgentState(TypedDict):
    content: str
    meta: str
    pathos: str
    summary: str
    objective: str
    ethos: str
    logos: str
    argumentmap: str
    reco: str
    takeaways: str
    called_agents: list[str]


initial_state: AgentState = {
    "content": "",
    "meta": "",
    "pathos": "",
    "summary": "",
    "objective": "",
    "ethos": "",
    "logos": "",
    "argumentmap": "",
    "reco": "",
    "takeaways": "",
    "called_agents": [],
}
# Merge state + messages into the stream input expected by your supervisor
content = "what's the price of BTC?"

# stream_input = {
#     **initial_state,
#     "messages": [{"role": "user", "content": content}],
# }

# for chunk in app.stream(stream_input):
#     pretty_print_messages(chunk, last_message=True)

result = app.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "what's the price of BTC?",
            }
        ]
    }
)

print(result["messages"][-1].content)
