from graphs.wrappers import (
    convo_orchestrator_wrapper,
    convo_reuse_node,
    convo_new_run_node,
    convo_cannot_answer_node,
    convo_route_decision,
    )

from states.conversation_graph_state import ConversationGraphState
from langgraph.graph import StateGraph

class ConversationGraph:
    def __init__(self):
        # Build the graph once at instantiation
        self._graph = self._build_graph()

    @property
    def graph(self):
        return self._graph

    def _build_graph(self):
        convo_builder = StateGraph(ConversationGraphState)

        # Nodes
        convo_builder.add_node("convo_orchestrator", convo_orchestrator_wrapper)
        convo_builder.add_node("reuse_answer", convo_reuse_node)
        convo_builder.add_node("new_run", convo_new_run_node)
        convo_builder.add_node("cannot_answer", convo_cannot_answer_node)

        # Entry point
        convo_builder.set_entry_point("convo_orchestrator")

        # Routing based on orchestrator decision
        convo_builder.add_conditional_edges(
            "convo_orchestrator",
            convo_route_decision,
            {
                "reuse": "reuse_answer",
                "new_run": "new_run",
                "cannot_answer": "cannot_answer",
            },
        )

        # Compile and return
        return convo_builder.compile()
