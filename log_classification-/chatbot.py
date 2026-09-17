"""
chatbot.py
----------
A general-purpose conversational assistant for talking through logs,
errors, and troubleshooting. Keeps conversation memory across turns.
Unlike agent.py (which calls tools on your classified DataFrame), this
is free-form chat - explaining errors, suggesting fixes, discussing
patterns in plain language.
"""

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

SYSTEM_PROMPT = (
    "You are a helpful assistant for a software team dealing with system logs. "
    "You can explain error messages, suggest troubleshooting steps, and discuss "
    "log patterns in plain language. Be concise and practical."
)


def chat(history: list, user_message: str, api_key: str) -> str:
    """history: list of (role, content) tuples, role in {'user', 'assistant'}."""
    llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key, temperature=0.5)

    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    for role, content in history:
        messages.append(HumanMessage(content=content) if role == "user" else AIMessage(content=content))
    messages.append(HumanMessage(content=user_message))

    response = llm.invoke(messages)
    return response.content
