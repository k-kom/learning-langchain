from typing import Annotated, TypedDict

# define state?
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

class State(TypedDict):
    # add_messages is a function
    messages: Annotated[list, add_messages]
    # messages: list

builder = StateGraph(State)

# adding chatbot node
# node is just a function
from langchain_openai import ChatOpenAI

model=ChatOpenAI()

def chatbot(state: State):
    # model should be an arg 🤔
    answer = model.invoke(state["messages"])
    return {"messages": [answer]}

builder.add_node("chatbot",chatbot)

# adding edges
# UGLY
builder.add_edge(START, 'chatbot')
builder.add_edge('chatbot', END)

graph=builder.compile()

# returns PNG byte array
# graph.get_graph().draw_mermaid_png()

# you can execute with strem
from langchain_core.messages import HumanMessage
user_input={"messages": [HumanMessage('hi!')]}
for chunk in graph.stream(user_input):
    print(chunk)

# persistent StateGraph
# from langgraph.checkpoint.memory import InMemorySaver
# graph=builder.compile(checkpointer=InMemorySaver())
