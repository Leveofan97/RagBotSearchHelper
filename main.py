# app.py
import os
import tempfile
import time
import streamlit as st
from streamlit_chat import message
from logic import ChatPDF
import logging

st.set_page_config(page_title="Прототип ИИ чата")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
            except Exception as e:
                agent_text = f"Ошибка: {str(e)}"

        st.session_state["messages"].append((user_text, True))
        st.session_state["messages"].append((agent_text, False))


def read_and_save_file():
    # st.session_state["assistant"].clear()
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

    # Настройки
    st.subheader("Настройки")

    # Настройка модели
    llm_model = st.text_input("Версия модели LLM", value="deepseek-r1:latest", help="Это строка, указывающая на конкретную версию модели для обработки текста (например, DeepSeek или другая модель). Модели могут быть разных типов или версий, которые могут изменять поведение вашего ИИ-помощника. Пример значения: deepseek-r1:8b или deepseek-r1:32b, что указывает на последнюю версию модели DeepSeek.")
    # Настройка векторного преобразователя
    embedding_model = st.text_input("Модель для встраивания", value="mxbai-embed-large", help="Этот параметр определяет модель, которая будет использоваться для преобразования текста в векторное представление (embedding). Эти векторы помогают системе искать схожие документы или фрагменты текста. По умолчанию используется mxbai-embed-large преобразователь, если на сервере Ollama установлены другие версии, ты  впишите в это поле альтернативную версию преобразователя")
    # Настройка размеров чанков
    chunk_size = st.slider("Размер чанков", min_value=256, max_value=4096, value=1024, step=256, help="При загрузке документа он считывается и преобразуется в вектор (двоичное представление файла). Считывание производится порционно, размер порции считывания и есть 1 чанк. В среднем размер одного чанка варьируется в пределах 100-1000 слов. Чем меньше чанк по размеру, тем точнее будет буквальный поиск, чем больше размер чанка тем больше поиск приближается к смысловому.")
    # Настройка перекрытия
    chunk_overlap = st.slider("Перекрытие чанков", min_value=0, max_value=1024, value=100, step=50, help="Если по простому то это общее количество символов которыми чанки пересекаюются между собой, то есть как бы накладываются друг на друга. Чанки должны перекрывать друг друга, чтобы у вас был шанс подать на вход последовательность чанков, которые следуют друг за другом вместе, а не просто вырванные из контекста куски. Начало и конец чанка должны быть осмысленными, в идеале должны совпадать с началом и концом предложения, а лучше абзаца, чтобы вся мысль была в чанке целиком.")
    # Настройка температуры
    temperature = st.session_state["temperature"] = st.slider(
        "Температура (temperature)", min_value=0.0, max_value=1.0, value=0.7, step=0.05,
        help="Температура контролирует степень случайности в ответах модели. Меньшие значения делают модель более детерминированной, а большие значения — более креативной и разнообразной."
    )
    system_prompt = st.text_area(
        "Системный промпт",
        value="Ты - полезный помощник, отвечающий на вопросы на основе загруженного документа.\nContext:\n{context}\n\nQuestion:\n{question}\n\nОтветьте кратко и точно в три предложения или меньше.",
        help="Шаблон для формирования запроса к модели. Используйте {context} для вставки контекста и {question} для вопроса пользователя."
    )

    current_params = {
        "llm_model": llm_model,
        "embedding_model": embedding_model,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "temperature": temperature,
        "system_prompt": system_prompt,
    }

    prev_params = {
        k: st.session_state.get(k, "")
        for k in ("llm_model", "embedding_model", "chunk_size", "chunk_overlap", "temperature", "system_prompt")
    }

    if current_params != prev_params:
        # Если изменился только промпт
        if (current_params["system_prompt"] != prev_params.get("system_prompt")
                and all(current_params[k] == prev_params[k] for k in
                        ["llm_model", "embedding_model", "chunk_size", "chunk_overlap", "temperature"])):

            st.session_state["assistant"].update_prompt(current_params["system_prompt"])
        else:
            # Полное пересоздание ассистента
            previous_assistant = st.session_state["assistant"]
            new_assistant = ChatPDF(**current_params)

            if hasattr(previous_assistant, "vector_store") and previous_assistant.vector_store:
                new_assistant.vector_store = previous_assistant.vector_store
                new_assistant.retriever = previous_assistant.retriever

            st.session_state["assistant"] = new_assistant

        st.session_state.update(current_params)

    st.session_state["retrieval_k"] = st.slider(
        "Количество полученных результатов (k)", min_value=1, max_value=10, value=5,
        help="Этот параметр определяет, сколько результатов (ответов или фрагментов текста) будет извлечено из документа для дальнейшей обработки. Чем больше значение, тем больше информации будет извлечено, но это также может замедлить процесс поиска. Обычно выбирают значение от 1 до 10, в зависимости от того, сколько релевантных фрагментов вы хотите получить в ответе."
    )
    st.session_state["retrieval_threshold"] = st.slider(
        "Порог оценки сходства", min_value=0.0, max_value=1.0, value=0.2, step=0.05,
        help="Этот параметр определяет минимальный порог сходства между запросом и фрагментом текста, который будет возвращен. Если оценка сходства (между запросом и документом) ниже этого порога, фрагмент текста не будет учтен. Меньшее значение позволяет возвращать больше фрагментов, даже если их сходство с запросом низкое, в то время как большее значение повышает точность, но уменьшает количество возвращаемых фрагментов."
    )

    # Display messages and text input
    chat_history()
    st.text_input("Напиши интересующий вопрос или действие", key="user_input", on_change=process_input)

    # Clear chat
    if st.button("Очистить чат", help="Очистка чата удаляет все ранее загруженные файлы, фактически это полный сброс системы "):
        st.session_state["messages"] = []
        st.session_state["assistant"].clear()


if __name__ == "__main__":
    page()
