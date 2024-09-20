import streamlit as st
import uuid

from langchain_core.messages.ai import AIMessage
from langchain_core.messages.tool import ToolMessage

from agent import agent_graph


if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.thread_id = uuid.uuid4().hex
    st.session_state.messages.append(
        {"role": "assistant", "content": "Olá, como posso ajudar?"}
    )


for message in st.session_state.messages:

    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(message["content"])

    elif message["role"] == "assistant":
        if isinstance(message["content"], AIMessage):

            if len(message["content"].content) > 0:
                with st.chat_message("assistant"):
                    st.markdown(message["content"].content)

            if len(message["content"].tool_calls) > 0:
                call_results_hist = {}

                for tool_call in message["content"].tool_calls:
                    status = st.status(
                        f"""Tool Call: {tool_call["name"]}""", state="complete"
                    )
                    call_results_hist[tool_call["id"]] = status
                    status.write("Input:")
                    status.write(tool_call["args"])

        elif isinstance(message["content"], ToolMessage):
            status = call_results_hist[message["content"].tool_call_id]
            status.write("Output:")
            status.write(message["content"].content)

        else:
            with st.chat_message("assistant"):
                st.markdown(message["content"])


def clear_chat_history():
    st.session_state.messages = []
    st.session_state.messages.append(
        {"role": "assistant", "content": "Olá, como posso ajudar?"}
    )
    st.session_state.thread_id = uuid.uuid4().hex


if st.sidebar.button("Reset Chat"):
    clear_chat_history()
    st.rerun()

# st.sidebar.button("Limpar Chat", on_click=clear_chat_history)

if prompt := st.chat_input("Digite a sua mensagem..."):

    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    stream = agent_graph.stream(
        {"messages": [("user", prompt)]},
        stream_mode="values",
        config={"configurable": {"thread_id": st.session_state.thread_id}},
    )
    for event in stream:
        response = event["messages"][-1]

        if isinstance(response, AIMessage):

            st.session_state.messages.append({"role": "assistant", "content": response})

            if len(response.content) > 0:
                with st.chat_message("assistant"):
                    st.markdown(response.content)

            if len(response.tool_calls) > 0:
                call_results = {}

                for tool_call in response.tool_calls:
                    status = st.status(
                        f"""Tool Call: {tool_call["name"]}""", state="running"
                    )
                    call_results[tool_call["id"]] = status
                    status.write("Input:")
                    status.write(tool_call["args"])

        if isinstance(response, ToolMessage):
            for tool_message in event["messages"][-len(call_results) :]:
                status = call_results[tool_message.tool_call_id]
                status.write("Output:")
                status.write(tool_message.content)
                status.update(state="complete")

                st.session_state.messages.append(
                    {"role": "assistant", "content": tool_message}
                )
