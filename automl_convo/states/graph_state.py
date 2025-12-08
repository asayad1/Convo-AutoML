"""
Defines the internal graph state.
"""

from typing import TypedDict
from states.auto_ml_state import AutoMLState

class GraphState(TypedDict):
    state: AutoMLState
    question: str