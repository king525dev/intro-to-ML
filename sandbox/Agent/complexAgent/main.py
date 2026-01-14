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

class MessageClassifier(BaseModel):
     message_type: Literal["emotional", "logical"] = Field(
          ...,
          description="Classify if the message has an emotional (therapist) or logical response"
     );

# Set a state for our graph
class State(TypedDict):
     messages: Annotated[list, add_messages];
     message_type: str | None;

# Define the nodes of the graph
def classifyMessage(state: State):
     lastMessage = state["messages"][-1];
     classifierLLM = llm.with_structured_output(MessageClassifier);
     
     result = classifierLLM.invoke([
               {
                    "role": "system",
                    "content": """Classify the user's message as either 
                    - 'emotional': if it asks for emotional support, therapy, deals with feelings, or personal problems
                    - 'logical': if it asks for facts, information, logical analysis, or practical solutions
                    """
               }, 
               {
                    "role": "user",
                    "content": lastMessage.content
               }
          ]);

     return {"message_type": result.message_type};

def router(state: State):
     message_type = state.get("message_type", "logical");
     
     if message_type == "emotional":
          return {"next": "therapist"};
     else:
          return {"next": "logical"};
     
def therapistAgent(state: State):
     lastMessage = state["messages"][-1];
     
     print("----Therapist Agent Invoked----");
     
     messages = [
          {
               "role": "system",
               "content": """You are a compassionate therapist. Focus on the emotional aspects of the user's message.
                         Show empathy, validate their feelings, and help them process their emotions.
                         Ask thoughtful questions to help them explore their feelings more deeply.
                         Avoid giving logical solutions unless explicitly asked."""
          },
          {
               "role": "user",
               "content": lastMessage.content
          }
     ];
     reply = llm.invoke(messages);
     return {"messages": [{
          "role": "assistant",
          "content": reply.content
     }]}

def logicalAgent(state: State):
     lastMessage = state["messages"][-1];
     
     print("----Logical Agent Invoked----");
     
     messages = [
          {
               "role": "system",
               "content": """You are a purely logical assistant. Focus only on facts and information.
               Provide clear, concise answers based on logic and evidence.
               Do not address emotions or provide emotional support.
               Be direct and straightforward in your responses."""
          },
          {
               "role": "user",
               "content": lastMessage.content
          }
     ];
     reply = llm.invoke(messages);
     return {"messages": [{
          "role": "assistant",
          "content": reply.content
     }]}

# Define a Graph Builder
graphBuilder = StateGraph(State);

# Define the edges of the graph
graphBuilder.add_node("classifier", classifyMessage);
graphBuilder.add_node("router", router);
graphBuilder.add_node("therapist", therapistAgent);
graphBuilder.add_node("logical", logicalAgent);

graphBuilder.add_edge(START, "classifier");
graphBuilder.add_edge("classifier", "router");

graphBuilder.add_conditional_edges(
     "router",
     lambda state: state.get("next"),
     {
          "therapist": "therapist",
          "logical": "logical"
     }
);

graphBuilder.add_edge("therapist", END);
graphBuilder.add_edge("logical", END);

# Compile and Run the graph
graph = graphBuilder.compile();

def runAgent():
     initialState: State = {
          "messages": [],
          "message_type": None
     };
     
     while True:
          userInput = input("Message: ");
          if userInput == "":
               print("Exiting...");
               break;
          
          initialState["messages"] = initialState.get("messages", []) + [{
               "role": "user",
               "content": userInput
          }];
          
          state = graph.invoke(initialState);
          
          if state.get("messages") and len(state["messages"]) > 0:
               lastMessage = state["messages"][-1];
               print("Assistant:", lastMessage.content);

if __name__ == "__main__":
     runAgent();