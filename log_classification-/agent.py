"""
agent.py
--------
A tool-calling agent for asking questions about your classified logs.
This is the "agentic" piece: instead of a fixed script, the LLM decides
on its own which tool(s) to call (count logs, search logs, inspect
unclassified rows, ...), reads the results, and loops until it can
answer - possibly calling multiple tools in sequence.
"""

import json
import pandas as pd

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "count_by_label",
            "description": "Count how many logs fall under each target_label.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "count_by_source",
            "description": "Count how many logs came from each source system.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_logs",
            "description": "Search log messages containing a keyword, return up to 5 matches.",
            "parameters": {
                "type": "object",
                "properties": {"keyword": {"type": "string"}},
                "required": ["keyword"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_unclassified",
            "description": "Get up to 5 log rows labeled 'Unclassified'.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def _run_tool(df: pd.DataFrame, name: str, args: dict) -> str:
    if name == "count_by_label":
        return df["target_label"].value_counts().to_string()
    if name == "count_by_source":
        return df["source"].value_counts().to_string()
    if name == "search_logs":
        keyword = args.get("keyword", "")
        matches = df[df["log_message"].str.contains(keyword, case=False, na=False)]
        return matches.head(5).to_string(index=False) if len(matches) else "No matches found."
    if name == "get_unclassified":
        matches = df[df["target_label"] == "Unclassified"]
        return matches.head(5).to_string(index=False) if len(matches) else "No unclassified logs."
    return f"Unknown tool: {name}"


def ask_agent(df: pd.DataFrame, question: str, api_key: str, max_steps: int = 5) -> str:
    from groq import Groq

    client = Groq(api_key=api_key)

    messages = [
        {
            "role": "system",
            "content": "You are a log-analysis assistant. Use the available tools to inspect "
            "the classified logs before answering. Be concise.",
        },
        {"role": "user", "content": question},
    ]

    for _ in range(max_steps):
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.3,
        )
        message = response.choices[0].message

        if not message.tool_calls:
            return message.content

        messages.append(message)
        for tool_call in message.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments or "{}")
            result = _run_tool(df, name, args)
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})

    return "Couldn't finish analyzing that in time — try a more specific question."
