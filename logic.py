# logic.py
import os
import shutil

from langchain_core.globals import set_verbose, set_debug
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.prompts import ChatPromptTemplate
import logging

set_debug(True)
set_verbose(True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChatPDF:

    ## Указать используемую версию дистиллята
    def __init__(self, llm_model: str = "deepseek-r1:latest", embedding_model: str = "mxbai-embed-large", chunk_size: int = 1024, chunk_overlap: int = 100, temperature: float = 0.7, system_prompt: str = None):
        # Дефолтный промпт, если не задан
        default_prompt = (
            "Ты - полезный помощник, отвечающий на вопросы на основе загруженного документа.\n"
            "Context:\n{context}\n\nQuestion:\n{question}\n\n"
            "Ответьте кратко и точно в три предложения или меньше."
        )

        """
        Initialize the ChatPDF instance with an LLM and embedding model.
        """
        self.model = ChatOllama(model=llm_model, temperature=temperature)
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.prompt_template = system_prompt or default_prompt
        self.prompt = ChatPromptTemplate.from_template(self.prompt_template)
        self.vector_store = None
        self.retriever = None

    def reading(self, pdf_file_path: str):
        logger.info(f"Считывание файла: {pdf_file_path}")
        docs = PyPDFLoader(file_path=pdf_file_path).load()
        chunks = self.text_splitter.split_documents(docs)
        chunks = filter_complex_metadata(chunks)

        if self.vector_store is None:
            self.vector_store = Chroma.from_documents(
                chunks, self.embeddings, persist_directory="chroma_db"
            )
        else:
            self.vector_store.add_documents(chunks)
        logger.info("Документ добавлен в хранилище.")

    def ask(self, query: str, k: int = 5, score_threshold: float = 0.2):
        if not self.vector_store:
            raise ValueError("Не удалось найти вектор в хранилище векторов. Загрузите документ заново.")

        if not self.retriever:
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"k": k, "score_threshold": score_threshold},
            )

        logger.info(f"Получение контекста для запроса: {query}")
        retrieved_docs = self.retriever.invoke(query)

        if not retrieved_docs:
            return "В документе нет соответствующего контекста, чтобы ответить на ваш вопрос."

        formatted_input = {
            "context": "\n\n".join(doc.page_content for doc in retrieved_docs),
            "question": query,
        }

        # Build the RAG chain
        chain = (
                RunnablePassthrough()  # Passes the input as-is
                | self.prompt  # Formats the input for the LLM
                | self.model  # Queries the LLM
                | StrOutputParser()  # Parses the LLM's output
        )

        logger.info("Генерирование ответа.")
        return chain.invoke(formatted_input)

    def clear(self):
        if self.vector_store:
            self.vector_store.delete_collection()
            self.vector_store = None
        self.retriever = None
        if os.path.exists("chroma_db"):
            shutil.rmtree("chroma_db")

    def update_prompt(self, new_prompt: str):
        """Обновление промпта без пересоздания всего класса"""
        self.prompt_template = new_prompt
        self.prompt = ChatPromptTemplate.from_template(new_prompt)