import sys
import os
import uuid
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from config.secrets import load_secrets

load_dotenv()

# set basic layout of the frontend
st.set_page_config(page_title="Research Agent", layout="centered")
st.title("Research Agent")

# load the agent and checkpointer only once per process, lazily on first use
@st.cache_resource
def get_agent():
    load_secrets()  # loads from Secrets Manager in production; no-op if AWS_SECRETS_NAME is not set
    from agent.agent import build_agent
    from checkpointing.checkpointer import get_checkpointer
    checkpointer = get_checkpointer()
    return build_agent(checkpointer=checkpointer)

# assign a stable thread_id for this browser session
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# empty list to store the full conversation history in
if "messages" not in st.session_state:
    st.session_state.messages = []

# render the full conversation history on every rerun
for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# append user message to conversation history
if user_input := st.chat_input("Ask a research question..."):
    human_msg = HumanMessage(content=user_input)
    st.session_state.messages.append(human_msg)

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            result = get_agent().invoke({"messages": [human_msg]}, config=config)
            response = result["messages"][-1].content

        # collect tool calls from messages after the last human message only
        all_messages = result["messages"]
        last_human_idx = max(i for i, m in enumerate(all_messages) if isinstance(m, HumanMessage))
        tool_calls = [
            tc["name"]
            for msg in all_messages[last_human_idx:]
            if hasattr(msg, "tool_calls")
            for tc in msg.tool_calls
        ]

        # display tool usage
        if tool_calls:
            with st.expander(f"Tools used: {', '.join(tool_calls)}"):
                for name in tool_calls:
                    st.markdown(f"- `{name}`")

        st.markdown(response)

    st.session_state.messages.append(AIMessage(content=response))
