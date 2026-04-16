import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
load_dotenv()

# set basic layout of the frontend
st.set_page_config(page_title="Research Agent", layout="centered")
st.title("Research Agent")

# load the agent only once per session
@st.cache_resource
def get_agent():
    from agent.agent import build_agent
    return build_agent()

# load the agent
agent = get_agent()

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
            # send full conversation history to the agent and run the agent
            result = agent.invoke({"messages": st.session_state.messages})
            response = result["messages"][-1].content

        # collect tool calls from intermediate messages (after the last human message)
        tool_calls = []
        for msg in result["messages"]:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls.append(tc["name"])

        # display tool usage
        if tool_calls:
            with st.expander(f"Tools used: {', '.join(tool_calls)}"):
                for name in tool_calls:
                    st.markdown(f"- `{name}`")

        st.markdown(response)

    st.session_state.messages.append(AIMessage(content=response))
