from typing import TypedDict
from langgraph.graph import StateGraph, START, END


class GraphState(TypedDict):
    message: str


def test_node(state: GraphState):
    print("Node received:", state["message"])

    return {
        "message": state["message"] + " → LangGraph worked!"
    }


graph = StateGraph(GraphState)

graph.add_node("test_node", test_node)

graph.add_edge(START, "test_node")
graph.add_edge("test_node", END)

app = graph.compile()

result = app.invoke({
    "message": "Hello"
})

print("Final result:")
print(result)