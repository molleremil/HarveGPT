# STREAMLIT VERSION 2.1
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import WebBaseLoader, YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from PIL import Image
from PyPDF2 import PdfReader
from dotenv import load_dotenv

# Load secrets from .env file
load_dotenv()


def extract_data_from_url(url):
    '''
    Extract the url content and return as a list of Document objects -> [Document].

    args: url (str)
    '''
    loader = WebBaseLoader(url)
    doc = loader.load()

    return doc


def extract_transcript_from_youtube_url(youtube_url):
    '''
    Extract the transcript of a YouTube video and return as a list of Document objects -> [Document].

    args: url (str): The url of the YouTube video
    '''
    youtube_loader = YoutubeLoader.from_youtube_url(
        youtube_url, add_video_info=False)
    transcript = youtube_loader.load()

    return transcript


def create_vectorstore_from_pdf(uploaded_pdf):
    '''
    Extract the text content from a PDF file, embed it and store in a vector db.

    args: uploaded pdf (file)
    '''
    pdf_reader = PdfReader(uploaded_pdf)

    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n", "\n\n", "\r", "\t", " "],
        chunk_size=1000,
        chunk_overlap=0,
    )
    text_chunks = text_splitter.split_text(text)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_db = Qdrant.from_texts(
        text_chunks,
        embeddings,
        location=":memory:",  # Using in-memory storage
        collection_name="HarveDocs")

    return vector_db


def create_vectorstore_from_data(data):
    '''
    1. Split the text data into text chunks.
    2. Vectorize text chunks and store in a vector db.
    3. Return the vector db.

    args: data -> [document]: List of Document objects
    '''
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n", "\n\n", "\r", "\t", " "],
        chunk_size=1000,
        chunk_overlap=0,
    )
    text_chunks = text_splitter.split_documents(data)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_db = Qdrant.from_documents(
        text_chunks,
        embeddings,
        location=":memory:",  # Using in-memory storage
        collection_name="HarveDocs")

    return vector_db


def create_context_retriever_chain(vec_store):
    '''
    Get the context retriever chain to be used in the dialog chain.
    '''
    llm = ChatOpenAI(temperature=0.1, max_tokens=500)
    retriever = vec_store.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Based on the conversation above, create a search query that you will refer to, to get information that is relevant to the conversation.")
    ])

    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain


def create_dialog_rag_chain(retriever_chain):
    '''
    Get the conversation chain
    '''
    llm = ChatOpenAI(temperature=0.1, max_tokens=500)
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("system",
         "Answer the user's questions based on the context below:\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)

    return create_retrieval_chain(retriever_chain, stuff_documents_chain)


def get_response(query):
    '''
    Get response from the AI model
    '''
    # Dialog chain
    retrieval_chain = create_context_retriever_chain(
        st.session_state.vec_store)

    dialog_rag_chain = create_dialog_rag_chain(retrieval_chain)
    response = dialog_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })
    return response["answer"]


def chat(user_input):
    if user_input and user_input.strip() != "":
        response = get_response(user_input)
        st.session_state.chat_history.append(
            HumanMessage(content=user_input))
        st.session_state.chat_history.append(AIMessage(content=response))

    # Dialog flow
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)


def get_chat_history():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello! How can I help you?")
        ]
        return st.session_state.chat_history


# UI Config
logo = Image.open("assets/logo_harve.png")
st.set_page_config(page_title="HarveGPT", page_icon=logo, layout="wide")
st.title("HarveGPT")


# Sidebar
with st.sidebar:
    st.header("Options")
    url = st.text_input("Enter Website or YouTube URL")
    uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])
    start_button = st.button("Start Chat")

# Options to start chat
if not url or url.strip() == "" or url is None:
    if uploaded_pdf is not None:
        chat_history = get_chat_history()

        if "vec_store" not in st.session_state:
            st.session_state.vec_store = create_vectorstore_from_pdf(
                uploaded_pdf)

        user_input = st.chat_input("Type a message...")
        chat(user_input)

    else:
        st.success("👈  Please provide Harve with a source to start the chat.")

else:
    try:
        if "youtube.com" in url or "youtu.be" in url:
            data = extract_transcript_from_youtube_url(url)
        else:
            data = extract_data_from_url(url)

    except Exception as e:
        st.warning(
            f"An error occurred: {e} Enter a valid link to continue.")
        st.stop()

    # Use `st.session_state`` to store chat history and avoid reinitializing the entire session
    chat_history = get_chat_history()

    if "vec_store" not in st.session_state:
        st.session_state.vec_store = create_vectorstore_from_data(data)

    # Chat input
    user_input = st.chat_input("Type a message...")
    chat(user_input)
