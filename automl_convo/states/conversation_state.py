"""
This file defines the current conversation state across multiple questions. 
"""

from typing import Optional, List, Dict
from states.auto_ml_state import AutoMLState
from dataclasses import dataclass, field

@dataclass
class ConversationState:
    """
    Holds long-lived state across multiple user questions.
    - last_automl_state: the AutoMLState from the most recent full run
    - qa_history: list of {question, answer} for conversational context
    """
    last_automl_state: Optional[AutoMLState] = None
    qa_history: List[Dict[str, str]] = field(default_factory=list)
