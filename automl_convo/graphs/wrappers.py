from utils.logger import Logger
from tools.cleaner import clean_node
from tools.model_planner import model_plan_node
from tools.profiler import profile_node
from tools.trainer import train_node
from tools.transformer import apply_transformations_node
from agents.analyist import analysis_node
from agents.feature_critic import feature_critic_node
from agents.feature_engineer import feature_engineer_node
from agents.orchestrator import orchestrator_node
from agents.conversation_orchestrator import conversation_orchestrator
from agents.model_results_explainer import model_results_explainer
from states.graph_state import GraphState
from states.conversation_graph_state import ConversationGraphState
import pandas as pd 
from llm import LLM

def profile_node_wrapped(gs: GraphState) -> GraphState:
    logger = Logger()

    s = gs["state"]
    datasets_history = s.datasets_history

    if s.df_current is None:
        if not datasets_history:
            raise ValueError(
                "profile_node_wrapped: state.df_current is None and datasets_history is empty; "
                "no dataset path available to load."
            )
        latest_path = datasets_history[-1]
        logger.info(
            f"[PROFILE WRAPPER] df_current is None – loading dataset from: {latest_path}",
            style='cyan',
        )
        s.df_current = pd.read_csv(latest_path)
        s.df_raw = s.df_current.copy()

    s = profile_node(s)
    gs["state"] = s
    return gs

def orchestrator_node_wrapped(gs: GraphState) -> GraphState:
    s = gs["state"]
    q = gs["question"]

    llm = LLM()
    s = orchestrator_node(s, llm, q)
    gs["state"] = s
    return gs

def feature_engineer_node_wrapped(gs: GraphState) -> GraphState:
    logger = Logger()
    
    s = gs["state"]

    # Increment iteration counter at the start of each feature engineering round
    s.iteration += 1
    logger.info(f"[ITERATION] Starting iteration {s.iteration}", style='grey')

    llm = LLM()
    s = feature_engineer_node(s, llm)
    gs["state"] = s
    return gs

def apply_transformations_node_wrapped(gs: GraphState) -> GraphState:
    s = gs["state"]
    s = apply_transformations_node(s)
    gs["state"] = s
    return gs

def clean_node_wrapped(gs: GraphState) -> GraphState:
    s = gs["state"]
    s = clean_node(s)
    gs["state"] = s
    return gs

def model_plan_node_wrapped(gs: GraphState) -> GraphState:
    s = gs["state"]
    s = model_plan_node(s)
    gs["state"] = s
    return gs

def train_node_wrapped(gs: GraphState) -> GraphState:
    s = gs["state"]
    s = train_node(s)
    gs["state"] = s
    return gs

def feature_critic_node_wrapped(gs: GraphState) -> GraphState:
    s = gs["state"]

    llm = LLM()
    s = feature_critic_node(s, llm)
    gs["state"] = s
    return gs

def analysis_node_wrapped(gs: GraphState) -> GraphState:
    s = gs["state"]
    q = gs["question"]

    llm = LLM()
    s = analysis_node(s, llm, q)
    gs["state"] = s
    return gs

# Conditional routing: decide whether to loop or finish
def should_continue(gs: GraphState) -> str:
    s = gs["state"]
    plan = s.feature_critic_plan or {"apply": False}
    apply_flag = plan.get("apply", False)

    if s.iteration >= s.max_iterations:
        return "stop"
    if not apply_flag:
        return "stop"
    return "continue"


def convo_orchestrator_wrapper(gs: ConversationGraphState) -> ConversationGraphState:
    conv_state = gs["conv_state"]
    question = gs["question"]

    llm = LLM()
    decision = conversation_orchestrator(question, llm, conv_state)
    gs["decision"] = decision
    return gs

def convo_route_decision(gs: ConversationGraphState) -> str:
    decision = gs.get("decision") or {}
    reuse = decision.get("reuse", False)
    need_new_run = decision.get("need_new_run", not reuse)

    if reuse:
        return "reuse"
    if need_new_run:
        return "new_run"
    return "cannot_answer"


def convo_reuse_node(gs: ConversationGraphState) -> ConversationGraphState:
    logger = Logger()

    conv_state = gs["conv_state"]
    question = gs["question"]

    logger.info(
        "[CONVERSATION GRAPH] Reusing previous AutoML results; no new training.",
        style="cyan",
    )

    if conv_state.last_automl_state is None:
        answer = (
            "I don't have any previous modeling results yet; "
            "I would need to run a new analysis first."
        )
    else:
        llm = LLM()
        answer = model_results_explainer(
            question,
            llm,
            conv_state.last_automl_state,
        )

    # Log the reused answer in a markdown box
    logger.box_md(
        "RESPONSE TO USER (REUSED RESULTS)",
        answer,
        style="green",
    )

    conv_state.qa_history.append({"question": question, "answer": answer})
    gs["conv_state"] = conv_state
    gs["answer"] = answer
    return gs

def convo_new_run_node(gs: ConversationGraphState) -> ConversationGraphState:
    logger = Logger()

    conv_state = gs["conv_state"]
    question = gs["question"]
    csv_path = gs["csv_path"]
    max_iterations = gs["max_iterations"]
    temp_dir = gs["temp_dir"]

    logger.info("[CONVERSATION GRAPH] Running a NEW AutoML analysis for this question.", style='cyan')

    # Lazy import here to avoid circular import at module load time
    from utils.drivers import run_multi_iteration_analysis

    new_state = run_multi_iteration_analysis(
        question=question,
        csv_path=csv_path,
        max_iterations=max_iterations,
        temp_dir=temp_dir,
    )
    conv_state.last_automl_state = new_state
    answer = new_state.final_answer or "Analysis completed, but no final answer was stored."

    conv_state.qa_history.append({"question": question, "answer": answer})
    gs["conv_state"] = conv_state
    gs["answer"] = answer
    return gs

def convo_cannot_answer_node(gs: ConversationGraphState) -> ConversationGraphState:
    decision = gs.get("decision") or {}
    reason = decision.get("reason", "The question cannot be answered with this AutoML system.")
    answer = f"I cannot answer this question with the current AutoML setup: {reason}"

    conv_state = gs["conv_state"]
    question = gs["question"]
    conv_state.qa_history.append({"question": question, "answer": answer})

    gs["conv_state"] = conv_state
    gs["answer"] = answer
    return gs
