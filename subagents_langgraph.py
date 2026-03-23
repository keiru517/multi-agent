from langchain.tools import tool
from langchain.agents import create_agent
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import AnyMessage
from typing_extensions import TypedDict, Annotated
import operator
from langchain.messages import SystemMessage, HumanMessage, ToolMessage
from typing import Literal
from langgraph.graph import StateGraph, START, END

load_dotenv()


from langfuse.langchain import CallbackHandler

langfuse_handler = CallbackHandler()

model = init_chat_model(
    "gpt-5",
    temperature=0,
    api_key=os.environ.get("OPENAI_API_KEY"),
)


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int
    query: str
    suggested_agent: Literal["generalist", "knowledge_assistant"]


def generalist(state: dict):
    """Generalist agent uses general knowledge to answer the user's question"""

    return {
        "messages": [
            model.invoke(
                [
                    SystemMessage(
                        content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
                    )
                ]
                + state["messages"]
            )
        ],
    }


def knowledge_assistant(state: dict):
    """Knowledge assistant agent uses internal documents to answer the user's question"""

    return {
        "messages": [
            model.invoke(
                [
                    SystemMessage(
                        content="You are a knowledge assistant agent. You are given a question and you need to use your knowledge to answer the question."
                    )
                ]
                + [HumanMessage(content=state["query"])]
            )
        ],
    }


def intent_router(state: dict):
    """Analyze query to determin if it needs knowledge base"""
    query = state["query"]
    company_keywords = [
        "beantragen",
        "apply for",
        "zuschuss",
        "subsidy",
        "erstattung",
        "reimbursement",
        "kosten",
        "cost",
    ]

    # Check if query mentions processes that likely have company policies
    has_company_context = any(keyword in query for keyword in company_keywords)

    # If vague but about processes/policies, route to knowledge assistant
    if has_company_context:
        print("Routing to knowledge assistant")
        return {"suggested_agent": "knowledge_assistant"}

    print("Routing to generalist")
    return {"suggested_agent": "generalist"}


# Build workflow
agent_builder = StateGraph(AgentState)

# Add nodes
agent_builder.add_node("intent_router", intent_router)
agent_builder.add_node("generalist", generalist)
agent_builder.add_node("knowledge_assistant", knowledge_assistant)

# Add edges to connect nodes
agent_builder.add_edge(START, "intent_router")
agent_builder.add_conditional_edges(
    "intent_router",
    lambda state: state["suggested_agent"],
    {
        "generalist": "generalist",
        "knowledge_assistant": "knowledge_assistant",
        "END": END,
    },
)

# Compile the agent
agent = agent_builder.compile()

# png_bytes = agent.get_graph().draw_mermaid_png()
# with open("graph_visualization.png", "wb") as f:
#     f.write(png_bytes)
# print("Graph visualization saved to graph_visualization.png")

# Invoke
query = "where can I apply for computer "

messages = [HumanMessage(content=query)]
messages = agent.invoke(
    {"messages": messages, "query": query}, config={"callbacks": [langfuse_handler]}
)
for m in messages["messages"]:
    m.pretty_print()
