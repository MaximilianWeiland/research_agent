import sys
import os
import uuid
import io
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from config.secrets import load_secrets

load_dotenv()

st.set_page_config(page_title="Research Agent", layout="centered")
st.title("Research Agent")

# load the agent only once at startup (or upon re-initialization)
@st.cache_resource
def get_agent(model: str, temperature: float):
    load_secrets()
    from agent.agent import build_agent
    from checkpointing.checkpointer import get_checkpointer
    checkpointer = get_checkpointer()
    return build_agent(checkpointer=checkpointer, model=model, temperature=temperature)

# streamlit sidebar for configurations
with st.sidebar:
    st.header("Settings")

    # select model and temperature vua slider
    model = st.selectbox("Model", ["gpt-4o", "gpt-4o-mini", "gpt-4.1"])
    temperature = st.slider("Temperature", 0.0, 2.0, 1.0, 0.1)
    st.caption(f"Active: {model} · temp {temperature}")

    st.divider()

    # update the vectorstore with new pdfs
    st.header("Update Index")
    uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    if uploaded_files and st.button("Add to index"):
        import hashlib
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from rag.vectorstore import get_vectorstore
        from config.settings import DOCS_DIR

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
         # instantiate a new vectorstore object
        # important: agent points to the same vs url, so it can directly refer to new pdfs
        vs = get_vectorstore()

        with st.spinner("Indexing..."):
            for file in uploaded_files:
                # write the pdf to disk
                file_bytes = file.read()
                dest = DOCS_DIR / file.name
                dest.write_bytes(file_bytes)

                # update the vector database based on the file's hash
                file_hash = hashlib.md5(file_bytes).hexdigest()
                loader = PyPDFLoader(str(dest))
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = file.name
                    doc.metadata["file_hash"] = file_hash

                splits = splitter.split_documents(docs)
                vs.add_documents(splits)

        st.success(f"Added {len(uploaded_files)} file(s) to index.")


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

    # render the message in markdown format
    with st.chat_message("user"):
        st.markdown(user_input)

    # call agent which then runs in a loop
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            result = get_agent(model, temperature).invoke({"messages": [human_msg]}, config=config)
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

        # display the selected tools and the final text response
        if tool_calls:
            with st.expander(f"Tools used: {', '.join(tool_calls)}"):
                for name in tool_calls:
                    st.markdown(f"- `{name}`")
        st.markdown(response)

    # update streamlit's full conversation history
    st.session_state.messages.append(AIMessage(content=response))
