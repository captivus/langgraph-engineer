from langgraph.graph import StateGraph, END
from langgraph.graph import MessagesState
from typing import TypedDict, Annotated, List
import operator

# Define the state with a messages key
class CodeState(MessagesState):
    requirements: str
    code: str
    feedback: str

# Define the logic for each node
def gather_requirements(state: CodeState) -> CodeState:
    return {"messages": [("system", "Gathered requirements")], "requirements": "specifications"}

def write_code(state: CodeState) -> CodeState:
    return {"messages": [("system", "Wrote initial code")], "code": "print('Hello, World!')"}

def validate_code(state: CodeState) -> CodeState:
    return {"messages": [("system", "Validated code")]}  # Assume validation is successful

def execute_code(state: CodeState) -> CodeState:
    return {"messages": [("system", "Executed code")]}  # Assume execution is successful

def test_code(state: CodeState) -> CodeState:
    return {"messages": [("system", "Tested code")]}  # Assume tests are successful

def collect_feedback(state: CodeState) -> CodeState:
    return {"messages": [("system", "Collected feedback")], "feedback": "No issues"}

def revise_code(state: CodeState) -> CodeState:
    return {"messages": [("system", "Revised code")]}  # Assume revision is successful

def finalize_code(state: CodeState) -> CodeState:
    return {"messages": [("system", "Finalized code")]}

# Create the graph
graph = StateGraph(CodeState)

# Add nodes
graph.add_node("gather_requirements", gather_requirements)
graph.add_node("write_code", write_code)
graph.add_node("validate_code", validate_code)
graph.add_node("execute_code", execute_code)
graph.add_node("test_code", test_code)
graph.add_node("collect_feedback", collect_feedback)
graph.add_node("revise_code", revise_code)
graph.add_node("finalize_code", finalize_code)

# Add edges
graph.add_edge("gather_requirements", "write_code")
graph.add_edge("write_code", "validate_code")
graph.add_edge("validate_code", "execute_code")
graph.add_edge("execute_code", "test_code")
graph.add_edge("test_code", "collect_feedback")
graph.add_edge("collect_feedback", "write_code")  # Feedback loop
graph.add_edge("collect_feedback", "revise_code")
graph.add_edge("revise_code", "validate_code")
graph.add_edge("test_code", "finalize_code")
graph.add_edge("finalize_code", END)

# Compile the graph
app = graph.compile()

# Example invocation
result = app.invoke({"messages": []})
print(result)