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

def chatConversation(user_question, system_prompt, groq_chat, memory):
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
    return response

def main():
    st.set_page_config(page_title="Chat Interface", layout="centered")

    st.markdown("""
        <style>
        .stTextArea textarea {
        
            color: white;
        
        }
        .stButton button {
            background-color: transparent;
            border: 1px solid white;
            color: white;
        }
        .response-area {
        
        
            border-radius: 4px;
            padding: 1rem;
            min-height: 200px;
        }
        </style>
    """, unsafe_allow_html=True)

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    groq_api_key = os.getenv('GROQ_API_KEY')
    if not groq_api_key:
        st.error("Please set the 'GROQ_API_KEY' environment variable.")
        return

    with st.sidebar:
        st.title('Customization')
        category_options = {
            "CODE": "Bantu menjawab pertanyaan tentang pemrograman.",
            "TECHNOLOGY": "Jelaskan tren teknologi terbaru.",
            "HEALTH": "Berikan saran kesehatan umum.",
            "KNOWLEDGE": "Bantu menjawab pertanyaan umum.",
            "BUSINESS": "Berikan nasihat tentang karir dan bisnis."
        }
        selected_category = st.selectbox("Pilih kategori:", list(category_options.keys()))
        system_prompt = category_options[selected_category]
        
        model = st.selectbox(
            'Choose a model',
            ['llama3-8b-8192', 'mixtral-8x7b-32768', 'gemma-7b-it']
        )
        
        conversational_memory_length = st.slider(
            'Conversational memory length:', 
            1, 10, value=5
        )

    st.title("PROGRAMMING CHATBOT")
    
    user_question = st.text_area("", height=100, placeholder="Enter your message here...")
    
    notification_placeholder = st.empty()
    col1, col2, col3 = st.columns([2, 3, 1])
    
    with col1:
        if st.button("Clear", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.user_question = ""
            notification_placeholder.success("Chat history cleared.")
            
    with col3:
        submit_button = st.button("Submit", use_container_width=True)

    response_placeholder = st.empty()
    
    memory = ConversationBufferWindowMemory(
        k=conversational_memory_length, 
        memory_key="chat_history", 
        return_messages=True
    )
    
    groq_chat = ChatGroq(
        groq_api_key=groq_api_key,
        model_name=model
    )

    if submit_button and user_question:
        response = chatConversation(user_question, system_prompt, groq_chat, memory)
        with response_placeholder:
            st.markdown(f"""
            <div class="response-area">
                {response}
            </div>
            """, unsafe_allow_html=True)

    if st.session_state.chat_history:
        st.markdown("### History Conservation")
        for message in st.session_state.chat_history:
            st.markdown(f"**User:** {message['human']}")
            st.markdown(f"**Chatbot:** {message['AI']}")

if __name__ == "__main__":
    main()