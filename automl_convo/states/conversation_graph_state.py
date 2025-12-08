"""
This file defines the main conversational graph state that is used by the main
driver.
"""

from typing import TypedDict, Optional, Any, Dict
from states.conversation_state import ConversationState

class ConversationGraphState(TypedDict):
    conv_state: ConversationState
    question: str
    csv_path: str
    max_iterations: int
    temp_dir: str
    decision: Optional[Dict[str, Any]]
    answer: Optional[str]