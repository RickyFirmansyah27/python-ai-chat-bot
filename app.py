import streamlit as st
import os

from langchain.chains import LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq


def main():
    groq_api_key = os.getenv('GROQ_API_KEY')
    if not groq_api_key:
        st.error("Please set the 'GROQ_API_KEY' environment variable.")
        return

    st.title("PROGRAMMING CHATBOT")
    st.write("Teach your programming language with AI!")

    st.sidebar.title('Customization')
    system_prompt = st.sidebar.text_input("System prompt:")
    model = st.sidebar.selectbox(
        'Choose a model',
        ['llama3-8b-8192', 'mixtral-8x7b-32768', 'gemma-7b-it']
    )
    conversational_memory_length = st.sidebar.slider('Conversational memory length:', 1, 10, value=5)

    memory = ConversationBufferWindowMemory(k=conversational_memory_length, memory_key="chat_history", return_messages=True)

    # Change from st.text_input to st.text_area
    user_question = st.text_area("Ask a question:", height=100)

    if st.button("Clear"):
        st.session_state.chat_history = []
        st.session_state.user_question = ""

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.chat_history:
        memory.save_context(
            {'input': message['human']},
            {'output': message['AI']}
        )

    groq_chat = ChatGroq(
        groq_api_key=groq_api_key,
        model_name=model
    )

    if st.button("Submit"):
        if user_question:
            if system_prompt:
                prompt = ChatPromptTemplate.from_messages(
                    [
                        SystemMessage(content=system_prompt),
                        MessagesPlaceholder(variable_name="chat_history"),
                        HumanMessagePromptTemplate.from_template("{human_input}"),
                    ]
                )

                conversation = LLMChain(
                    llm=groq_chat,
                    prompt=prompt,
                    verbose=True,
                    memory=memory,
                )
                
                response = conversation.predict(human_input=user_question)
                message = {'human': user_question, 'AI': response}
                st.session_state.chat_history.append(message)
                st.write("Chatbot:", response)
            else:
                st.warning("Please enter a system prompt.")

if __name__ == "__main__":
    main()
