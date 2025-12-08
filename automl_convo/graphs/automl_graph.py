from graphs.wrappers import (
    profile_node_wrapped, 
    orchestrator_node_wrapped, 
    feature_critic_node_wrapped, 
    apply_transformations_node_wrapped,
    train_node_wrapped,
    clean_node_wrapped,
    analysis_node_wrapped,
    model_plan_node_wrapped,
    feature_engineer_node_wrapped, 
    should_continue
    )

from states.graph_state import GraphState
from langgraph.graph import StateGraph

class AutoMLGraph:
    def __init__(self):
        # Build the graph when class is instantiated
        self._graph = self._build_graph()

    @property
    def graph(self):
        return self._graph

    def _build_graph(self):
        builder = StateGraph(GraphState)

        builder.add_node("profile", profile_node_wrapped)
        builder.add_node("orchestrate", orchestrator_node_wrapped)
        builder.add_node("feature_engineer", feature_engineer_node_wrapped)
        builder.add_node("apply_transforms", apply_transformations_node_wrapped)
        builder.add_node("clean", clean_node_wrapped)
        builder.add_node("model_plan", model_plan_node_wrapped)
        builder.add_node("train", train_node_wrapped)
        builder.add_node("feature_critic", feature_critic_node_wrapped)
        builder.add_node("analysis", analysis_node_wrapped)

        # Entry point
        builder.set_entry_point("profile")

        # Linear path
        builder.add_edge("profile", "orchestrate")
        builder.add_edge("orchestrate", "feature_engineer")
        builder.add_edge("feature_engineer", "apply_transforms")
        builder.add_edge("apply_transforms", "clean")
        builder.add_edge("clean", "model_plan")
        builder.add_edge("model_plan", "train")
        builder.add_edge("train", "feature_critic")

        # Loop or finish
        builder.add_conditional_edges(
            "feature_critic",
            should_continue,
            {
                "continue": "feature_engineer",
                "stop": "analysis",
            },
        )

        # Compile
        return builder.compile()
