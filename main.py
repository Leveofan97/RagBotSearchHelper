# app.py
import os
import tempfile
import time
import streamlit as st
from streamlit_chat import message
from rag import ChatPDF

st.set_page_config(page_title="Прототип ИИ чата")


def chat_history():
    st.subheader("История чата")
    for i, (msg, is_user) in enumerate(st.session_state["messages"]):
        message(msg, is_user=is_user, key=str(i))
    st.session_state["thinking_spinner"] = st.empty()


def process_input():
    if st.session_state["user_input"] and len(st.session_state["user_input"].strip()) > 0:
        user_text = st.session_state["user_input"].strip()
        with st.session_state["thinking_spinner"], st.spinner("Думаю..."):
            try:
                agent_text = st.session_state["assistant"].ask(
                    user_text,
                    k=st.session_state["retrieval_k"],
                    score_threshold=st.session_state["retrieval_threshold"],
                )
            except ValueError as e:
                agent_text = str(e)

        st.session_state["messages"].append((user_text, True))
        st.session_state["messages"].append((agent_text, False))


def read_and_save_file():
    st.session_state["assistant"].clear()
    st.session_state["messages"] = []
    st.session_state["user_input"] = ""

    for file in st.session_state["file_uploader"]:
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(file.getbuffer())
            file_path = tf.name

        with st.session_state["ingestion_spinner"], st.spinner(f"Идет загрузка документа {file.name}..."):
            t0 = time.time()
            st.session_state["assistant"].reading(file_path)
            t1 = time.time()

        st.session_state["messages"].append(
            (f"Время загрузки: {file.name} in {t1 - t0:.2f} seconds", False)
        )
        os.remove(file_path)


def page():
    if len(st.session_state) == 0:
        st.session_state["messages"] = []
        st.session_state["assistant"] = ChatPDF()

    st.header("Прототип ИИ чата")

    st.subheader("Загрузите документ")
    st.file_uploader(
        "Upload a PDF document",
        type=["pdf"],
        key="file_uploader",
        on_change=read_and_save_file,
        label_visibility="collapsed",
        accept_multiple_files=True,
    )

    st.session_state["ingestion_spinner"] = st.empty()

    st.subheader("Настройки")
    st.session_state["retrieval_k"] = st.slider(
        "Количество полученных результатов (k)", min_value=1, max_value=10, value=5
    )
    st.session_state["retrieval_threshold"] = st.slider(
        "Порог оценки сходства", min_value=0.0, max_value=1.0, value=0.2, step=0.05
    )

    # Display messages and text input
    chat_history()
    st.text_input("Напиши интересующий вас вопрос", key="user_input", on_change=process_input)

    # Clear chat
    if st.button("Очистить чат"):
        st.session_state["messages"] = []
        st.session_state["assistant"].clear()


if __name__ == "__main__":
    page()
