from typing import Annotated, TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage
)
from langchain_openai import ChatOpenAI

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

model = ChatOpenAI()

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

generate_prompt = SystemMessage(
    """You are an essay assistant tasked with writing excellent
    3-paragraph essays."""
    "Generate the best essay possible for the user's request."
    """If the user provieds critique, respond with a revised version
    of your previous attempts."""
)

def generate(state: State) -> State:
    answer=model.invoke([generate_prompt] + state["messages"])
    return {"messages":[answer]}

reflection_prompt = SystemMessage(
    """You are a teacher grading an essay submission. Generate critique and
    recommendations for the user's submission."""
    """Provide detailed recommendations, including requests for the length,
    depth, sysle, etc."""
)

def reflect(state: State) -> State:
    cls_map={AIMessage: HumanMessage, HumanMessage: AIMessage}
    translated=[reflection_prompt, state["messages"][0]] + [
        cls_map[msg._class_](content=msg.content)
        for msg in state["messages"][1:]
    ]
    answer=model.invoke(translated)
    return {"messages": [HumanMessage(content=answer.content)]}

def should_continue(state: State):
    if len(state["messages"])>6:
        return END
    else:
        return "reflect"

builder = StateGraph(State)
builder.add_node("generate", generate)
builder.add_node("reflect", reflect)

builder.add_edge(START, "generate")
builder.add_conditional_edges("generate", should_continue)
builder.add_edge("reflect", "generate")

graph = builder.compile()

initial_state = {
    "messages": [
        HumanMessage(
            content="Write an essay about the relevance of 'The Little Prince' today."
        )
    ]
}

# Run the graph
for output in graph.stream(initial_state):
    message_type = "generate" if "generate" in output else "reflect"
    print("\nNew message:", output[message_type]
          ["messages"][-1].content[:100], "...")
