"""
This file implements the conversation orchestrator agent.
"""

from typing import Dict, Any
from utils.logger import Logger
from utils.llm import summarize_automl_state_for_llm
from llm import LLM
from states.conversation_state import ConversationState
import json
import re

def conversation_orchestrator(question: str, llm: LLM, conv_state: ConversationState) -> Dict[str, Any]:
    """
    LLM-based meta-orchestrator.
    Decides whether to:
      - reuse previous AutoML results to answer the question, or
      - run a new AutoML pipeline (possibly with a different target / task / PCA choice).
    """

    logger = Logger()
    previous_runs_summary = (
        summarize_automl_state_for_llm(conv_state.last_automl_state)
        if conv_state.last_automl_state is not None
        else "No previous AutoML results."
    )

    qa_history_str = "\n".join(
        f"Q: {qa['question']}\nA: {qa['answer']}\n"
        for qa in conv_state.qa_history[-5:]
    ) or "No prior Q&A in this conversation."

    system_prompt = """
You are a meta-orchestrator for a conversational AutoML assistant.

You are given:
- The user's new question.
- A summary of previous AutoML runs (if any).
- A brief Q&A history from this conversation.

Your job:
- Decide whether the new question can be answered using only the existing results
  (previous AutoML runs and prior answers) WITHOUT re-running any models or
  feature engineering.
- Or whether we MUST run a new AutoML pipeline (e.g., because the target, task type,
  or interpretability needs are different).

Respond ONLY with JSON of the form:
{
  "reuse": true or false,
  "reason": "short explanation",
  "need_new_run": true or false
}

Rules:
- Set reuse=true when:
  - The question is essentially asking for clarification, re-explanation, or a slice
    of insight that is already supported by the previous results (same target, same task).
- Set need_new_run=true when:
  - The question requires a new target, a fundamentally different task (classification vs regression),
    or clearly different feature engineering that is NOT supported by prior results.
- It is possible that reuse=false and need_new_run=false, e.g., if the question
  cannot be answered at all with this system; in that case, clearly say so in "reason".
"""

    human_prompt = f"""
New user question:
{question}

Previous AutoML summary:
{previous_runs_summary}

Recent Q&A history:
{qa_history_str}
"""
    logger.info(f"[CONVERSATION ORCHESTRATOR] Determining whether to reuse previous results...", style="magenta")
    resp = llm.invoke(system_prompt, human_prompt)

    try:
        obj = json.loads(resp)
    except Exception:
        m = re.search(r"\{.*\}", resp, re.DOTALL)
        if not m:
            raise ValueError("Could not parse conversation orchestrator response as JSON.")
        obj = json.loads(m.group(0))

    logger.box(
        "CONVERSATION ORCHESTRATOR DECISION",
        f"reuse: {obj.get('reuse')}\n"
        f"need_new_run: {obj.get('need_new_run')}\n"
        f"reason: {obj.get('reason')}",
        style="magenta",
    )

    return obj
