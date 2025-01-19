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
from langchain.prompts import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
import json

st.set_page_config(
    page_title="Streamlit is",
    page_icon="🤖",
)

st.markdown(
    """
(EN)
Implement QuizGPT but add the following features:

Use function calling.
Allow the user to customize the difficulty of the test and make the LLM generate hard or easy questions.
Allow the user to retake the test if not all answers are correct.
If all answers are correct use st.ballons.
Allow the user to use its own OpenAI API Key, load it from an st.input inside of st.sidebar
Using st.sidebar put a link to the Github repo with the code of your Streamlit app.

(KR)
QuizGPT를 구현하되 다음 기능을 추가합니다:

함수 호출을 사용합니다.
유저가 시험의 난이도를 커스터마이징 할 수 있도록 하고 LLM이 어려운 문제 또는 쉬운 문제를 생성하도록 합니다.
만점이 아닌 경우 유저가 시험을 다시 치를 수 있도록 허용합니다.
만점이면 st.ballons를 사용합니다.
유저가 자체 OpenAI API 키를 사용하도록 허용하고, st.sidebar 내부의 st.input에서 로드합니다.
st.sidebar를 사용하여 Streamlit app의 코드와 함께 Github 리포지토리에 링크를 넣습니다.

"""
)

def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})

#schema
function = {
    "name": "create_quiz",
    "description": "function that takes a list of questions and answers and returns a quiz",
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                        },
                        "answers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "answer": {
                                        "type": "string",
                                    },
                                    "correct": {
                                        "type": "boolean",
                                    },
                                },
                                "required": ["answer", "correct"],
                            },
                        },
                    },
                    "required": ["question", "answers"],
                },
            }
        },
        "required": ["questions"],
    },
}

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

    with st.sidebar:
        difficulty = st.selectbox(
        "Please select the difficulty of the test", ("Difficult", "Easy"), index=None, placeholder="Select the difficulty of the test...",
    )

    if difficulty:

        llm = ChatOpenAI(
            temperature = 0.1,
            streaming=True,
            callbacks=[
                ChatCallbackHandler(),
            ],
            openai_api_key = input_open_api_key,
        ).bind(
        function_call={
            "name": "create_quiz",
        },
        functions=[
            function,
        ],
    )
    # only work for gpt-3 and gpt-4

        #st.write(difficulty)
        prompt = PromptTemplate.from_template("Make a quiz about {city}." + "The difficulty level is " + difficulty)

        chain = prompt | llm

        response = chain.invoke({"city": "Seoul"})
        #st.balloons()

        response = response.additional_kwargs["function_call"]["arguments"]

        with st.form("questions_form"):
            correct_answer = 0
            total_questions = len(json.loads(response)["questions"])
            for question in json.loads(response)["questions"]:
                #print(question)
                st.write(question["question"]) #showing questions
                value = st.radio("Select an option.", [answer["answer"] for answer in question["answers"]], index=None)
                    #st.write(value)
                if {"answer" : value, "correct" : True} in question["answers"]:
                    st.success("Correct!")
                    correct_answer+=1
                elif value is not None:
                    st.error("Wrong..")
                    # check the answer
            button = st.form_submit_button("Submit")

            if button:
                st.session_state.submitted = True
                if correct_answer == total_questions:
                    st.balloons()

