from typing import Annotated, Literal;
from asciiGenerator import asciiArtFromText;
from langgraph.graph import StateGraph, START, END;
from langgraph.prebuilt import ToolNode
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
     
# Decision function
def decisionNode(state: State) -> str:
     lastMessage = state["messages"][-1]["content"].lower();
     if "ascii art" in lastMessage or "ascii" in lastMessage:
          return "asciiArtFromText";
     return "chatbot";
     
# Define a Graph Builder
graphBuilder = StateGraph(State);

# Define the nodes of the graph
def chatbot(state: State):
     return{"messages": [llm.invoke(state["messages"])]};

graphBuilder.add_node("chatbot", chatbot);

# Define tool nodes
asciiToolNode = ToolNode.from_function(
     asciiArtFromText,
     name="asciiArtFromText",
     description="Generate ASCII art from a text string using a specified font style.",
     return_direct=True,
     parse_docstring=True   
);
graphBuilder.add_tool_node(asciiToolNode);

# Define conditional edges based on decision function
graphBuilder.add_conditional_edge(
     "chatbot",
     decisionNode,
     {
          "asciiArtFromText": "asciiArtFromText",
          "chatbot": "chatbot"
     }
);


# Define the edges of the graph
graphBuilder.add_edge(START, "chatbot");
graphBuilder.add_edge("chatbot", END);

# Compile and Run the graph
graph = graphBuilder.compile();

# Repeated Run

userInput =  input("Enter a message: ");

while userInput.lower() != "exit" or userInput != "":
     state = graph.invoke(
          {"messages": [
               {"role": "user", "content": userInput}
          ]}
     )

     print("Chatbot response: ", state["messages"][-1].content);
     print("\n---\n");
     userInput =  input("Enter a message: ");