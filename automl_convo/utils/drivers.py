"""
This file defines different drivers of different aspects of the agentic framework.
"""

from utils.logger import Logger
from states.conversation_graph_state import ConversationGraphState
from states.auto_ml_state import AutoMLState
from states.conversation_state import ConversationState
from states.graph_state import GraphState
from typing import Optional
from langgraph.graph import StateGraph
from graphs.convo_automl_graph import ConversationGraph
import os

def run_multi_iteration_analysis(
    question: str,
    csv_path: str,
    max_iterations: int = 3,
    temp_dir: str = "tmp_datasets",
):
    """
    Run a full multi-iteration AutoML analysis for a single question/dataset.

    It constructs the AutoMLGraph lazily inside this function to avoid
    circular imports between utils.drivers, graphs.automl_graph, and wrappers.
    """
    # Lazy import here to break circular dependency
    from graphs.automl_graph import AutoMLGraph

    logger = Logger()

    os.makedirs(temp_dir, exist_ok=True)

    state = AutoMLState()
    state.iteration = 0
    state.max_iterations = max_iterations
    state.history = []
    state.temp_dir = temp_dir

    # Seed history with the original dataset path
    state.datasets_history = [csv_path]
    state.csv_path = csv_path

    gs: GraphState = {
        "state": state,
        "question": question,
    }

    logger.box(
        "AUTOML RUN START",
        f"Question: {question}\n"
        f"CSV path: {csv_path}\n"
        f"Max iterations: {max_iterations}\n"
        f"Temp dir: {temp_dir}",
        style="cyan",
    )

    # Build the inner AutoML graph
    automl_graph = AutoMLGraph().graph
    final_gs = automl_graph.invoke(gs)
    final_state = final_gs["state"]

    return final_state


class ConversationalAutoMLRunner:
    """
    High-level wrapper around the ConversationGraph.

    Responsibilities:
    - Hold the ConversationState across turns.
    - Expose a ask(question: str) interface.
    - Hide the ConversationGraphState plumbing and graph.invoke details.
    """

    def __init__(
        self,
        csv_path: str,
        max_iterations: int = 3,
        temp_dir: str = "augmented_datasets",
        conversation_graph: Optional[StateGraph] = None,
    ):
        self.csv_path = csv_path
        self.max_iterations = max_iterations
        self.temp_dir = temp_dir

        # Long-lived conversation state (shared across turns)
        self.conv_state = ConversationState()

        # Conversation-level graph wrapper
        if conversation_graph is None:
            convo = ConversationGraph()
            self._graph = convo.graph
        else:
            self._graph = conversation_graph

    def ask(self, question: str) -> str:
        """
        Process a single conversational turn.

        - Uses the existing conv_state (previous Q&A + AutoML results).
        - Decides reuse vs new run.
        - Returns the answer string and updates internal conv_state.
        """
        inputs: ConversationGraphState = {
            "conv_state": self.conv_state,
            "question": question,
            "csv_path": self.csv_path,
            "max_iterations": self.max_iterations,
            "temp_dir": self.temp_dir,
            "decision": None,
            "answer": None,
        }

        out = self._graph.invoke(inputs)

        # Update internal conversation state & return answer
        self.conv_state = out["conv_state"]
        answer = out["answer"]
        return answer
