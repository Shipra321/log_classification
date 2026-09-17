"""
app.py
------
Streamlit front-end for the hybrid log classifier.
Run with: streamlit run app.py
"""

import pandas as pd
import streamlit as st

from classify import classify, classify_log
from classifiers import get_embedding_model
from rag_store import RAGStore
from agent import ask_agent
from chatbot import chat as chatbot_chat
from investigator import run_investigation

st.set_page_config(page_title="Log Classifier", page_icon="◆", layout="centered")

# ---------------------------------------------------------------------------
# Theme — dark charcoal panels, gold accent, serif display type paired
# with a clean sans body. Injected once, applies across every tab.
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&family=Inter:wght@400;500;600&display=swap');

:root {
  --ink: #EDEFF2;
  --ink-dim: #9AA5B1;
  --ink-faint: #6B7480;
  --panel: #1B222C;
  --panel-raised: #212A35;
  --bg: #12161D;
  --border: #2B3542;
  --gold: #C6A15B;
  --gold-soft: #D9B673;
  --gold-dim: #7C6538;
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: var(--bg); color: var(--ink); }

/* Headings */
h1, h2, h3 { font-family: 'Source Serif 4', Georgia, serif !important; color: var(--ink) !important; font-weight: 600 !important; }
h1 { letter-spacing: 0.01em; }

[data-testid="stCaptionContainer"] { color: var(--ink-dim) !important; letter-spacing: 0.01em; }

/* Sidebar */
section[data-testid="stSidebar"] { background: var(--panel); border-right: 1px solid var(--border); }
section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
  font-family: 'Source Serif 4', serif !important;
  color: var(--gold) !important;
  font-size: 0.95rem !important;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid var(--border); }
.stTabs [data-baseweb="tab"] {
  font-family: 'Inter', sans-serif; font-weight: 500; color: var(--ink-dim);
  background: transparent; border-radius: 0; padding: 10px 18px;
}
.stTabs [aria-selected="true"] {
  color: var(--gold) !important;
  border-bottom: 2px solid var(--gold) !important;
  background: transparent !important;
}

/* Buttons */
.stButton>button, .stDownloadButton>button {
  background: var(--gold); color: #1A140A; border: none; border-radius: 3px;
  font-weight: 600; letter-spacing: 0.02em; padding: 0.5rem 1.2rem;
  transition: background 0.15s;
}
.stButton>button:hover, .stDownloadButton>button:hover { background: var(--gold-soft); color: #1A140A; }
.stButton>button:disabled { background: var(--panel-raised); color: var(--ink-faint); }

/* Metrics */
div[data-testid="stMetric"] {
  background: var(--panel); border: 1px solid var(--border); border-radius: 4px; padding: 14px 16px;
}
div[data-testid="stMetricLabel"] {
  color: var(--ink-dim) !important; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.07em;
}
div[data-testid="stMetricValue"] { color: var(--gold) !important; font-family: 'Source Serif 4', serif !important; }

/* Inputs */
.stTextInput input, .stTextArea textarea,
.stSelectbox div[data-baseweb="select"] > div,
.stNumberInput input {
  background: var(--panel-raised) !important;
  border: 1px solid var(--border) !important;
  color: var(--ink) !important;
  border-radius: 3px !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
  border-color: var(--gold) !important;
  box-shadow: 0 0 0 1px var(--gold-dim) !important;
}

/* Dataframe / data editor */
div[data-testid="stDataFrame"], div[data-testid="stDataEditor"] {
  border: 1px solid var(--border); border-radius: 4px; overflow: hidden;
}

/* Expander */
details { background: var(--panel); border: 1px solid var(--border) !important; border-radius: 4px; }
summary { font-family: 'Source Serif 4', serif; color: var(--ink) !important; }

/* Alerts */
div[data-testid="stAlert"] { border-radius: 4px; border: 1px solid var(--border); }

/* Chat messages */
div[data-testid="stChatMessage"] { background: var(--panel); border: 1px solid var(--border); border-radius: 6px; }

/* File uploader */
[data-testid="stFileUploaderDropzone"] {
  background: var(--panel-raised); border: 1px dashed var(--border) !important; border-radius: 4px;
}

/* Divider */
hr { border-color: var(--border) !important; }

/* Bar chart container */
[data-testid="stVegaLiteChart"], .stBarChart {
  background: var(--panel); border: 1px solid var(--border); border-radius: 4px; padding: 8px;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div style="padding-bottom:10px; border-bottom:1px solid #2B3542; margin-bottom:24px;">
        <div style="font-size:0.72rem; letter-spacing:0.14em; color:#C6A15B;
                    text-transform:uppercase; margin-bottom:8px;">Log Intelligence</div>
        <h1 style="margin:0; font-size:2rem;">Hybrid Log Classifier</h1>
        <p style="color:#9AA5B1; margin-top:10px; max-width:640px; font-size:0.95rem;">
            A disciplined pipeline — rule matching, embedding classification, and
            retrieval-augmented reasoning — for labeling, explaining, and
            investigating system logs.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

KNOWN_LABELS = [
    "User Action", "System Notification", "Workflow Error",
    "Deprecation Warning", "Security Alert", "Critical Error", "Unclassified",
]

SAMPLE_CSV = """source,log_message
ModernCRM,User User123 logged in.
BillingSystem,Backup completed successfully.
AnalyticsEngine,Disk cleanup completed successfully.
LegacyCRM,Customer record sync failed for account 34521 due to timeout.
BillingSystem,"Database connection pool exhausted, queueing requests."
"""

# ---------------------------------------------------------------------------
# Sidebar: API key + optional custom RAG knowledge base
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Groq API Key")
    api_key = st.text_input(
        "Enter your Groq API key", type="password", placeholder="gsk_...",
        help="Needed for LegacyCRM logs, RAG fallback, and the agent tab.",
        label_visibility="collapsed",
    )
    st.session_state["groq_api_key"] = api_key or None
    if api_key:
        st.success("Key set.")
    else:
        st.info("No key entered yet.")

    st.divider()
    st.header("Knowledge Base")
    st.caption("Upload your own labeled logs to power retrieval instead of the built-in examples.")
    kb_file = st.file_uploader(
        "CSV with 'log_message' and 'target_label' (source column optional)",
        type=["csv"], key="kb_uploader", label_visibility="collapsed",
    )
    if kb_file is not None:
        kb_df = pd.read_csv(kb_file)
        if st.button("Build knowledge base"):
            try:
                with st.spinner("Embedding your logs..."):
                    store = RAGStore(get_embedding_model())
                    store.load_dataframe(kb_df)
                    st.session_state["rag_store"] = store
                st.success(f"Loaded {len(kb_df)} examples into your knowledge base.")
            except ValueError as e:
                st.error(str(e))

    if st.session_state.get("rag_store", None) and st.session_state["rag_store"].is_ready():
        st.caption(f"Using {len(st.session_state['rag_store'].texts)} uploaded examples for retrieval.")

    st.divider()
    st.download_button(
        "Download sample CSV",
        data=SAMPLE_CSV,
        file_name="sample_logs.csv",
        mime="text/csv",
        help="A ready-to-use CSV with 'source' and 'log_message' columns to try the classifier.",
    )

tab_csv, tab_single, tab_agent, tab_investigator, tab_chatbot = st.tabs(
    ["Classify a CSV", "Classify a Message", "Ask the Agent", "Investigator", "Chatbot"]
)

# ---------------------------------------------------------------------------
# Tab 1: CSV upload
# ---------------------------------------------------------------------------
with tab_csv:
    st.write("Upload a CSV with **`source`** and **`log_message`** columns.")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"], key="classify_uploader")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        missing = {"source", "log_message"} - set(df.columns)
        if missing:
            st.error(f"CSV is missing required column(s): {', '.join(missing)}")
        else:
            st.dataframe(df.head(), use_container_width=True)

            if st.button("Classify logs", type="primary"):
                with st.spinner("Classifying..."):
                    try:
                        results = classify(
                            list(zip(df["source"], df["log_message"])),
                            api_key=st.session_state.get("groq_api_key"),
                            rag_store=st.session_state.get("rag_store"),
                        )
                        df["target_label"] = [label for label, _, _ in results]
                        df["confidence"] = [round(conf, 3) for _, conf, _ in results]
                        df["explanation"] = [expl for _, _, expl in results]
                    except ValueError as e:
                        st.error(str(e))
                        st.stop()

                df["predicted_label"] = df["target_label"]
                st.session_state["classified_df"] = df
                st.success(f"Classified {len(df)} log messages.")

            if st.session_state.get("classified_df") is not None:
                result_df = st.session_state["classified_df"]

                st.subheader("Label distribution")
                st.bar_chart(result_df["target_label"].value_counts())

                low_conf_count = int((result_df["confidence"] < 0.5).sum())
                if low_conf_count:
                    st.caption(f"{low_conf_count} row(s) below 0.5 confidence — worth a manual look below.")

                st.subheader("Review & correct predictions")
                st.caption("Edit `target_label` directly for any row that's wrong. Saved corrections are "
                           "added to your knowledge base for future retrieval.")

                label_options = sorted(set(result_df["target_label"].unique()) | set(KNOWN_LABELS))
                editable_cols = ["source", "log_message", "target_label", "confidence"]

                edited_df = st.data_editor(
                    result_df[editable_cols],
                    column_config={
                        "target_label": st.column_config.SelectboxColumn(
                            "target_label", options=label_options, required=True,
                        ),
                        "confidence": st.column_config.ProgressColumn(
                            "confidence", min_value=0.0, max_value=1.0, format="%.0f%%",
                        ),
                        "source": st.column_config.TextColumn("source", disabled=True),
                        "log_message": st.column_config.TextColumn("log_message", disabled=True),
                    },
                    disabled=["confidence"],
                    use_container_width=True,
                    key="csv_editor",
                    hide_index=True,
                )

                corrected_mask = edited_df["target_label"] != result_df["predicted_label"].values
                corrected_count = int(corrected_mask.sum())

                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button(f"Save {corrected_count} correction(s)", disabled=corrected_count == 0):
                        corrected_texts = edited_df.loc[corrected_mask, "log_message"].tolist()
                        corrected_labels = edited_df.loc[corrected_mask, "target_label"].tolist()

                        store = st.session_state.get("rag_store")
                        if store is None:
                            store = RAGStore(get_embedding_model())
                            st.session_state["rag_store"] = store
                        with st.spinner("Embedding corrections..."):
                            store.append_examples(corrected_texts, corrected_labels)

                        result_df.loc[corrected_mask, "target_label"] = corrected_labels
                        st.session_state["classified_df"] = result_df
                        st.success(f"Saved {corrected_count} correction(s) to the knowledge base.")
                with col_b:
                    st.caption("Corrections improve future RAG/LLM classifications, "
                               "not past regex/BERT predictions.")

                st.subheader("Why was a row classified this way?")
                row_options = [f"{i}: {row.log_message[:60]}" for i, row in result_df.iterrows()]
                chosen = st.selectbox("Pick a row to inspect", row_options, key="explain_row_select")
                chosen_idx = int(chosen.split(":")[0])
                st.info(result_df.loc[chosen_idx, "explanation"])

                csv_bytes = result_df.drop(columns=["predicted_label"]).to_csv(index=False).encode("utf-8")
                st.download_button("Download results", data=csv_bytes,
                                    file_name="output.csv", mime="text/csv")

# ---------------------------------------------------------------------------
# Tab 2: single message
# ---------------------------------------------------------------------------
with tab_single:
    source = st.text_input("Source", placeholder="e.g. ModernCRM, BillingSystem, LegacyCRM")
    log_message = st.text_area("Log message", placeholder="e.g. User 12345 logged in.", height=100)

    if st.button("Classify"):
        if not source or not log_message:
            st.warning("Please fill in both fields.")
        else:
            with st.spinner("Classifying..."):
                try:
                    label, confidence, explanation = classify_log(
                        source, log_message,
                        api_key=st.session_state.get("groq_api_key"),
                        rag_store=st.session_state.get("rag_store"),
                    )
                except ValueError as e:
                    st.error(str(e))
                    st.stop()
            col1, col2 = st.columns(2)
            col1.metric("Predicted label", label)
            col2.metric("Confidence", f"{confidence:.0%}")
            if confidence < 0.5:
                st.caption("Low confidence — consider reviewing this one manually.")
            with st.expander("Why this label?"):
                st.write(explanation)

# ---------------------------------------------------------------------------
# Tab 3: agentic Q&A over the classified results
# ---------------------------------------------------------------------------
with tab_agent:
    st.write("Ask questions about the logs you've classified. The agent decides which tools to use on its own.")

    df = st.session_state.get("classified_df")
    if df is None:
        st.info("Classify a CSV in the first tab to unlock this.")
    elif not st.session_state.get("groq_api_key"):
        st.warning("Add a Groq API key in the sidebar to use the agent.")
    else:
        if "agent_history" not in st.session_state:
            st.session_state["agent_history"] = []

        st.caption("Try one of these, or type your own below:")
        starter_questions = [
            "How many logs are Unclassified?",
            "Show me logs about timeouts",
            "Which source has the most errors?",
        ]
        chip_cols = st.columns(len(starter_questions))
        clicked_question = None
        for col, q in zip(chip_cols, starter_questions):
            if col.button(q, key=f"chip_{q}"):
                clicked_question = q

        top_col1, top_col2 = st.columns([4, 1])
        with top_col2:
            if st.button("Clear conversation", key="clear_agent_chat"):
                st.session_state["agent_history"] = []
                st.rerun()

        for role, text in st.session_state["agent_history"]:
            st.chat_message(role).write(text)

        typed_question = st.chat_input("Ask about your classified logs...")
        question = clicked_question or typed_question

        if question:
            st.session_state["agent_history"].append(("user", question))
            st.chat_message("user").write(question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer = ask_agent(df, question, api_key=st.session_state["groq_api_key"])
                st.write(answer)
            st.session_state["agent_history"].append(("assistant", answer))

# ---------------------------------------------------------------------------
# Tab 4: autonomous investigator (LangGraph agent - runs on its own)
# ---------------------------------------------------------------------------
with tab_investigator:
    st.write(
        "One click, no questions needed: this agent finds every critical/unclassified log in your "
        "batch, investigates each one using similar past examples, and writes up a report — "
        "deciding on its own what needs attention."
    )

    df = st.session_state.get("classified_df")
    if df is None:
        st.info("Classify a CSV in the first tab to unlock this.")
    elif not st.session_state.get("groq_api_key"):
        st.warning("Add a Groq API key in the sidebar to use the investigator.")
    else:
        if st.button("Run investigation", type="primary"):
            with st.spinner("Finding critical logs and investigating each one..."):
                report = run_investigation(
                    df,
                    api_key=st.session_state["groq_api_key"],
                    rag_store=st.session_state.get("rag_store"),
                )
            st.session_state["investigation_report"] = report

        if st.session_state.get("investigation_report"):
            st.markdown(st.session_state["investigation_report"])
            st.download_button(
                "Download report",
                data=st.session_state["investigation_report"],
                file_name="investigation_report.md",
                mime="text/markdown",
            )

# ---------------------------------------------------------------------------
# Tab 5: general chatbot (free-form, no tools, keeps memory)
# ---------------------------------------------------------------------------
with tab_chatbot:
    st.write("Chat freely about logs, errors, or troubleshooting — not tied to a specific classified batch.")

    if not st.session_state.get("groq_api_key"):
        st.warning("Add a Groq API key in the sidebar to chat.")
    else:
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        top_col1, top_col2 = st.columns([4, 1])
        with top_col2:
            if st.button("Clear conversation", key="clear_chatbot_chat"):
                st.session_state["chat_history"] = []
                st.rerun()

        for role, text in st.session_state["chat_history"]:
            st.chat_message(role).write(text)

        user_message = st.chat_input("Ask anything about logs or errors...")
        if user_message:
            st.session_state["chat_history"].append(("user", user_message))
            st.chat_message("user").write(user_message)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    reply = chatbot_chat(
                        st.session_state["chat_history"][:-1], user_message, api_key=st.session_state["groq_api_key"]
                    )
                st.write(reply)
            st.session_state["chat_history"].append(("assistant", reply))

st.divider()
with st.expander("How does this work?"):
    st.markdown(
        """
1. **`LegacyCRM`** → RAG + LLM: retrieves similar labeled examples (yours, if uploaded, else a small built-in set)
   and gives them to the LLM as few-shot context.
2. **Other sources** → regex first, then BERT + Logistic Regression.
3. **BERT says "Unclassified"** and you've uploaded your own logs → RAG + LLM gets a second try using your data.
4. **Agent tab** → a tool-calling LLM agent that inspects your classified DataFrame (counts, searches, filters)
   and decides for itself which tools it needs to answer your question.
5. **Confidence & explanations** → every row gets a confidence score and a plain-language reason for its
   label; low-confidence rows (< 50%) are flagged for review.
6. **Manual corrections** → fix a wrong label directly in the results table and save it — corrected
   examples get added to your knowledge base to improve future classifications.
        """
    )