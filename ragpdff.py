import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

from langchain.memory import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain_chroma import Chroma
from langchain.vectorstores import FAISS

from langchain.document_loaders import PyPDFLoader
from chromadb.config import Settings  # Make sure this is imported

# Load environment
load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")

# --- Streamlit Page Setup ---
st.set_page_config(
    page_title="📄 RAG PDF Chatbot",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
body {
    font-family: 'Segoe UI', sans-serif;
}
header, .css-18e3th9 {
    background-color: #0e1117;
    color: white;
}
[data-testid="stSidebar"] {
    background-color: #93a2c4;
}
.chat-message.user {
    background-color: #2c3e50;
    padding: 10px;
    border-radius: 10px;
    color: white;
}
.chat-message.assistant {
    background-color: #34495e;
    padding: 10px;
    border-radius: 10px;
    color: white;
}
.stButton>button {
    border-radius: 8px;
    background-color: #4CAF50;
    color: white;
}
.uploadedFile {
    color: #00c7ff;
}
</style>
""", unsafe_allow_html=True)

# --- App Title ---
st.title("📄 PDF RAG Chatbot with Memory")

# Sidebar
with st.sidebar:
    st.subheader("⚙️ Settings")
    api_key = st.text_input("🔑 Enter your GROQ API Key", type="password")
    st.markdown("**Instructions**:")
    st.markdown("- Upload PDF(s) on the main page\n- Ask questions\n- Review full conversation history")

# Ensure API key is present
if not api_key:
    st.warning("Please provide your GROQ API Key in the sidebar to proceed.")
    st.stop()

# --- Embedding Model ---
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

# --- LLM ---
llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.1-8b-instant")

# --- PDF Upload ---
uploaded_files = st.file_uploader("📁 Upload one or more PDFs", type="pdf", accept_multiple_files=True)

all_docs = []

if uploaded_files:
    with st.spinner("🔄 Reading and processing PDF(s)..."):
        for file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.getvalue())
                loader = PyPDFLoader(tmp.name)
                docs = loader.load()
                all_docs.extend(docs)
        st.success(f"✅ Loaded {len(all_docs)} pages.")

# --- Split Text into Chunks ---
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
splits = splitter.split_documents(all_docs)

if not splits:
    st.warning("PDFs could not be parsed properly. Try a different file.")
    st.stop()



@st.cache_resource(show_spinner=False)
def get_vectorstore(_docs):
    return FAISS.from_documents(_docs, embeddings)


vectorstore = get_vectorstore(splits)
retriever = vectorstore.as_retriever()

# --- Prompt & Chain Setup ---
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given the chat history and the latest user question, decide what to retrieve."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use context to answer questions in 2-3 sentences.\n\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
qa_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

# --- Chat History Session ---
if "chathistory" not in st.session_state:
    st.session_state.chathistory = {}

def get_history(session_id: str):
    if session_id not in st.session_state.chathistory:
        st.session_state.chathistory[session_id] = ChatMessageHistory()
    return st.session_state.chathistory[session_id]

conversational_rag = RunnableWithMessageHistory(
    rag_chain,
    get_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer"
)

# --- Chat Interface ---
st.subheader("💬 Chat Interface")
session_id = st.text_input("🆔 Session ID", value="default")

user_input = st.chat_input("💭 Type your question here...")

if user_input:
    history = get_history(session_id)
    with st.spinner("🧠 Thinking..."):
        result = conversational_rag.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        )
    answer = result["answer"]

    st.chat_message("user").markdown(f"**You:** {user_input}")
    st.chat_message("assistant").markdown(f"**Assistant:** {answer}")

# --- Show Full History ---
with st.expander("📜 Full Chat History"):
    history = get_history(session_id)
    for msg in history.messages:
        role = getattr(msg, "role", msg.type).capitalize()
        content = msg.content
        st.markdown(f"**{role}:** {content}")
