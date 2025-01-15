import streamlit as st
import time
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.callbacks.base import BaseCallbackHandler

st.set_page_config(
    page_title="Streamlit is",
    page_icon="🤖",
)

st.markdown(
    """
# Hello!
            
Welcome to the assignment Streamlit is 🔥!
            
(EN)
Migrate the RAG pipeline you implemented in the previous assignments to Streamlit.
Implement file upload and chat history.
Allow the user to use its own OpenAI API Key, load it from an st.input inside of st.sidebar
Using st.sidebar put a link to the Github repo with the code of your Streamlit app.

(KR)
이전 과제에서 구현한 RAG 파이프라인을 Streamlit으로 마이그레이션합니다.
파일 업로드 및 채팅 기록을 구현합니다.
사용자가 자체 OpenAI API 키를 사용하도록 허용하고, st.sidebar 내부의 st.input에서 이를 로드합니다.
st.sidebar를 사용하여 스트림릿 앱의 코드와 함께 깃허브 리포지토리에 링크를 넣습니다.

PLEASE put your API key on the sidebar and press enter. 👈🏻

"""
)

with st.sidebar:
    input_open_api_key = st.text_input("Please enter your Open API Key here")

if input_open_api_key :

    class ChatCallbackHandler(BaseCallbackHandler):
        # to make it look like the AI writes the answer itself

        message = ""
    
        def on_llm_start(self, *args, **kwargs):
            self.message_box = st.empty()
        def on_llm_end(self, *args, **kwargs):
            save_message(self.message, "ai")

        def on_llm_new_token(self, token, *args, **kwargs):
            self.message += token
            self.message_box.markdown(self.message)

    llm = ChatOpenAI(
        temperature = 0.1,
        streaming=True,
        callbacks=[
            ChatCallbackHandler(),
        ],
        openai_api_key = input_open_api_key,
    )

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # decorator 
    # streamlit will check if we have the same file
    # if we have it, it will return the previous value, not rerunning the whole thing
    @st.cache_data(show_spinner="Embedding File...")
    def embed_file(file):
        file_content = file.read() # read the file that user uploaded
        file_path = f"./.cache/files/{file.name}"
        with open(file_path, "wb") as f:
            f.write(file_content)
        cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
        splitter = CharacterTextSplitter.from_tiktoken_encoder(
            separator="\n",
            chunk_size=600,
            chunk_overlap=100,
        )
        loader = UnstructuredFileLoader(file_path)
        docs = loader.load_and_split(text_splitter=splitter)
        embeddings = OpenAIEmbeddings(
            openai_api_key = input_open_api_key
        )
        cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
        vectorstore = FAISS.from_documents(docs, cached_embeddings)
        retriever = vectorstore.as_retriever()
        return retriever

    def save_message(message, role):
        st.session_state["messages"].append({"message": message, "role": role})

    def send_message(message, role, save=True):
        with st.chat_message(role):
            st.markdown(message)

        if save:
            save_message(message, role)


    def paint_history():
        for message in st.session_state["messages"]:
            send_message(message["message"], message["role"], save=False)

    def format_docs(docs):
        return "/n/n".join(document.page_content for document in docs)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON'T make anything up.
                
                Context: {context}
                """,
            ),
            ("human", "{question}"),
        ]
    )


    with st.sidebar:
        file = st.file_uploader(
            "Upload a .txt .pdf or .docx file",
            type=["pdf", "txt", "docx"],
        )

    if file:
        retriever = embed_file(file)
        #s = retriever.invoke("winston")
        send_message("I'm ready! Ask away!", "ai" , save=False)
        # it won't be saved because it happens always

        paint_history()
        message = st.chat_input("Ask anything about your file...")

        if message:
            send_message(message, "human")

            chain = {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough()
            } | prompt | llm

            with st.chat_message("ai"):
                response = chain.invoke(message) 
            #send_message(response.content, "ai")
    else:
        st.session_state["messages"] = [] #will be initialized
        


