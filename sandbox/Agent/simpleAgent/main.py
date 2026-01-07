from typing import Annotated, Literal;
from langgraph.graph import StateGraph, START, END;
from langgraph.graph.message import add_messages;
from langchain.chat_models import init_chat_model;
from pydantic import BaseModel, Field;
from typing_extensions import TypedDict;

# Load LLM
llm = init_chat_model(
     "ollama:gemma3:4b"
);

# Set a state for our graph
class State(TypedDict):
     messages: Annotated[list, add_messages];
     

# Define a Graph Builder
graphBuilder = StateGraph(State);

# Define the nodes of the graph
def chatbot(state: State):
     return{"messages": [llm.invoke(state["messages"])]};

graphBuilder.add_node("chatbot", chatbot);

# Define the edges of the graph
graphBuilder.add_edge(START, "chatbot");
graphBuilder.add_edge("chatbot", END);

# Compile and Run the graph
graph = graphBuilder.compile();

# Repeated Run

userInput =  input("Enter a message: ");

while userInput.lower() != "exit":
     state = graph.invoke(
          {"messages": [
               {"role": "user", "content": userInput}
          ]}
     )

     print("Chatbot response: ", state["messages"][-1].content);
     print("\n---\n");
     userInput =  input("Enter a message: ");