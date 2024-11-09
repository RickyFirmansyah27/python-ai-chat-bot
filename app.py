import streamlit as st
import pdfplumber
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
from langchain.text_splitter import RecursiveCharacterTextSplitter

def get_quick_actions(file_type, file_content):
    """Return quick actions berdasarkan tipe file dan konten"""
    common_actions = {
        "summarize": "Rangkum bagian ini secara lengkap",
        "main_points": "Identifikasi poin-poin utama",
        "keywords": "Ekstrak kata kunci penting"
    }
    
    # Deteksi konten berbasis coding
    code_indicators = [
        "def ", "class ", "import ", "function", "var ", "const ",
        "<html", "<?php", "#include", "package ", "using namespace"
    ]
    
    # Deteksi konten berbasis data
    data_indicators = [
        "SELECT ", "INSERT ", "CREATE TABLE", "DROP ", "ALTER ",
        "DataFrame", "pandas", "numpy", "matplotlib"
    ]
    
    # Deteksi konten akademis
    academic_indicators = [
        "Abstract", "Introduction", "Methodology", "Conclusion",
        "References", "et al.", "Fig.", "Table "
    ]
    
    # Cek indikator dalam konten
    is_code = any(indicator in file_content for indicator in code_indicators)
    is_data = any(indicator in file_content for indicator in data_indicators)
    is_academic = any(indicator in file_content for indicator in academic_indicators)
    
    specific_actions = {}
    
    if is_code:
        specific_actions.update({
            "explain_code": "Jelaskan kode ini",
            "code_review": "Review dan berikan saran perbaikan kode",
            "documentation": "Buatkan dokumentasi untuk kode ini",
            "optimize": "Sarankan optimisasi kode",
            "bug_check": "Cek potensi bug atau masalah",
            "test_cases": "Sarankan test cases"
        })
    
    if is_data:
        specific_actions.update({
            "data_analysis": "Analisis data ini",
            "query_explain": "Jelaskan query/operasi data ini",
            "data_validation": "Cek validitas data",
            "visualization": "Sarankan visualisasi yang sesuai",
            "data_cleaning": "Sarankan teknik cleaning data"
        })
    
    if is_academic:
        specific_actions.update({
            "research_summary": "Rangkum penelitian ini",
            "methodology_review": "Analisis metodologi",
            "literature_analysis": "Analisis kajian literatur",
            "findings_summary": "Rangkum temuan utama",
            "citation_check": "Periksa format sitasi"
        })
    
    # Jika tidak ada indikator spesifik, gunakan aksi umum untuk dokumen
    if not (is_code or is_data or is_academic):
        specific_actions.update({
            "extract_topics": "Ekstrak topik-topik utama",
            "generate_questions": "Buat pertanyaan-pertanyaan penting",
            "create_outline": "Buat outline dokumen",
            "sentiment_analysis": "Analisis sentimen dokumen"
        })
    
    return {**common_actions, **specific_actions}

def chunk_text(text, chunk_size=4000):
    """Split text into smaller chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=200,
        length_function=len,
    )
    return text_splitter.split_text(text)

def read_pdf(file):
    try:
        with pdfplumber.open(file) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None

def read_text_file(file):
    try:
        return file.read().decode("utf-8")
    except Exception as e:
        st.error(f"Error reading text file: {str(e)}")
        return None

def process_file_content(file_content, max_length=4000):
    """Process and truncate file content to prevent context length issues"""
    if not file_content:
        return None
    
    # Split content into chunks
    chunks = chunk_text(file_content)
    
    # Store all chunks in session state
    st.session_state.file_chunks = chunks
    
    # Return a summary of the file content
    return f"Dokumen berisi {len(chunks)} bagian dengan total {len(file_content)} karakter."

def chatConversation(user_question, system_prompt, groq_chat, memory, chunk_index=None):
    # Prepare the context based on the current chunk
    if chunk_index is not None and 'file_chunks' in st.session_state:
        current_chunk = st.session_state.file_chunks[chunk_index]
        context = f"\n\nBerikut adalah bagian {chunk_index + 1} dari dokumen:\n{current_chunk}"
    else:
        context = ""

    # Combine system prompt with context
    full_prompt = f"{system_prompt}{context}"

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=full_prompt),
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
        .stTextArea textarea { color: white; }
        .stButton button { background-color: transparent; border: 1px solid white; color: white; }
        .response-area { border-radius: 4px; padding: 1rem; min-height: 200px; }
        .file-content { background-color: #f0f2f6; padding: 1rem; border-radius: 4px; margin: 1rem 0; }
        .quick-actions { display: flex; flex-wrap: wrap; gap: 0.5rem; }
        .quick-action-button { flex: 1; min-width: 200px; }
        </style>
    """, unsafe_allow_html=True)

    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'file_chunks' not in st.session_state:
        st.session_state.file_chunks = []
    if 'current_chunk_index' not in st.session_state:
        st.session_state.current_chunk_index = 0
    if 'quick_actions' not in st.session_state:
        st.session_state.quick_actions = {}

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
            "BUSINESS": "Berikan nasihat tentang karir dan bisnis.",
            "DOCUMENT": "Analisis dan rangkum dokumen yang diunggah."
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
        
        # File uploader section
        uploaded_file = st.file_uploader("Upload a file (PDF or TXT)", type=["pdf", "txt"])
        if uploaded_file is not None:
            if uploaded_file.type == "application/pdf":
                file_content = read_pdf(uploaded_file)
            else:
                file_content = read_text_file(uploaded_file)
            
            if file_content:
                file_info = process_file_content(file_content)
                st.success("File berhasil diunggah!")
                st.info(file_info)
                
                # Update quick actions based on file content
                st.session_state.quick_actions = get_quick_actions(uploaded_file.type, file_content)
                
                if st.session_state.file_chunks:
                    chunk_index = st.number_input(
                        "Pilih bagian dokumen (0 sampai {}):".format(len(st.session_state.file_chunks) - 1),
                        0,
                        len(st.session_state.file_chunks) - 1,
                        st.session_state.current_chunk_index
                    )
                    st.session_state.current_chunk_index = chunk_index
                    
                    with st.expander("Lihat Bagian Dokumen Saat Ini"):
                        st.markdown(f"```\n{st.session_state.file_chunks[chunk_index][:500]}...\n```")
                
                # Quick action buttons
                if st.session_state.quick_actions:
                    st.subheader("Quick Actions")
                    cols = st.columns(2)
                    for i, (action_key, action_desc) in enumerate(st.session_state.quick_actions.items()):
                        col_idx = i % 2
                        with cols[col_idx]:
                            if st.button(action_desc, key=f"action_{action_key}"):
                                st.session_state.user_question = action_desc

    st.title("DOCUMENT CHAT ASSISTANT")
    
    # Main chat interface
    user_question = st.text_area(
        "Masukkan pertanyaan Anda:", 
        value=st.session_state.get("user_question", ""), 
        height=100,
        placeholder="Contoh: Tolong rangkum bagian ini, atau tanyakan hal spesifik tentang isi dokumen..."
    )

    # Control buttons
    col1, col2, col3 = st.columns([2, 3, 1])
    with col1:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.user_question = ""
            st.success("Chat history cleared.")
            
    with col3:
        submit_button = st.button("Submit", use_container_width=True)

    # Chat processing
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
        response = chatConversation(
            user_question,
            system_prompt,
            groq_chat,
            memory,
            st.session_state.current_chunk_index if st.session_state.file_chunks else None
        )
        with response_placeholder:
            st.markdown(f"""
            <div class="response-area">
                {response}
            </div>
            """, unsafe_allow_html=True)

    # Display chat history
    if st.session_state.chat_history:
        st.markdown("### Riwayat Percakapan")
        for message in st.session_state.chat_history:
            st.markdown(f"**User:** {message['human']}")
            st.markdown(f"**Assistant:** {message['AI']}")

if __name__ == "__main__":
    main()