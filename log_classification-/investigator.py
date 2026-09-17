"""
investigator.py
----------------
An autonomous agent (LangGraph) that investigates a classified batch of
logs on its own - one click, no back-and-forth:

    find_critical_logs --(none found)--> compile_report --> END
                        --(found)-------> investigate_each --> compile_report --> END

For every critical/unclassified log it retrieves similar past examples
(RAG) and asks the LLM to explain what likely happened + suggest one
concrete next action, then compiles everything into a single report.
This is what makes it "agentic": it decides on its own which logs need
attention and what to do about each one, instead of you asking it
questions one at a time.
"""

from typing import TypedDict, List, Optional, Any

import pandas as pd
from langgraph.graph import StateGraph, END

from classifiers import retrieve_similar_examples

CRITICAL_LABELS = {"Unclassified", "Workflow Error", "Security Alert", "Critical Error"}
MAX_LOGS_TO_INVESTIGATE = 10  # cap for cost/latency


class InvestigationState(TypedDict):
    df: Any
    api_key: str
    rag_store: Optional[Any]
    critical_rows: Optional[List[dict]]
    findings: Optional[List[dict]]
    report: Optional[str]


def find_critical_node(state: InvestigationState) -> InvestigationState:
    df = state["df"]
    critical = df[df["target_label"].isin(CRITICAL_LABELS)]
    rows = critical.to_dict("records")[:MAX_LOGS_TO_INVESTIGATE]
    return {**state, "critical_rows": rows}


def investigate_node(state: InvestigationState) -> InvestigationState:
    from langchain_groq import ChatGroq

    llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=state["api_key"], temperature=0.3)

    findings = []
    for row in state["critical_rows"]:
        examples = retrieve_similar_examples(row.get("log_message", ""), k=2, rag_store=state.get("rag_store"))
        examples_block = "\n".join(f'- "{t}" -> {l}' for t, l in examples) or "(none)"

        prompt = f"""A log was flagged as potentially critical.
Source: {row.get('source')}
Message: {row.get('log_message')}
Current label: {row.get('target_label')}

Similar past examples:
{examples_block}

In 2-3 sentences: explain what likely went wrong and suggest one concrete next action."""

        response = llm.invoke(prompt)
        findings.append({**row, "analysis": response.content.strip()})

    return {**state, "findings": findings}


def compile_report_node(state: InvestigationState) -> InvestigationState:
    findings = state.get("findings") or []
    if not findings:
        report = "No critical or unclassified logs found in this batch. ✅"
    else:
        lines = [f"### {len(findings)} log(s) need attention\n"]
        for f in findings:
            lines.append(f"**{f.get('source')} — {f.get('target_label')}**")
            lines.append(f"> {f.get('log_message')}")
            lines.append(f"{f.get('analysis')}\n")
        report = "\n".join(lines)
    return {**state, "report": report}


def _route_after_find(state: InvestigationState) -> str:
    return "investigate" if state.get("critical_rows") else "compile_report"


def build_investigator_graph():
    graph = StateGraph(InvestigationState)
    graph.add_node("find_critical", find_critical_node)
    graph.add_node("investigate", investigate_node)
    graph.add_node("compile_report", compile_report_node)

    graph.set_entry_point("find_critical")
    graph.add_conditional_edges(
        "find_critical", _route_after_find, {"investigate": "investigate", "compile_report": "compile_report"}
    )
    graph.add_edge("investigate", "compile_report")
    graph.add_edge("compile_report", END)

    return graph.compile()


def run_investigation(df: pd.DataFrame, api_key: str, rag_store=None) -> str:
    graph = build_investigator_graph()
    result = graph.invoke(
        {
            "df": df,
            "api_key": api_key,
            "rag_store": rag_store,
            "critical_rows": None,
            "findings": None,
            "report": None,
        }
    )
    return result["report"]
