from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import WebBaseLoader, YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv


# Load secrets from .env file
load_dotenv()


def extract_data_from_url(url):
    '''
    Extract the url content and return as a document.

    args: url (str): The url of the web page to extract content from
    '''
    loader = WebBaseLoader(url)
    doc = loader.load()

    return doc


def extract_transcript_from_youtube_url(youtube_url):
    '''
    Extract the transcript of a YouTube video.

    args: url (str): The url of the YouTube video
    '''
    youtube_loader = YoutubeLoader.from_youtube_url(
        youtube_url, add_video_info=False)
    transcript = youtube_loader.load()

    return transcript


def create_vectorstore_from_data(data):
    '''
    1. Split the text data into text chunks.
    2. Vectorize text chunks and store in a vector db.
    3. Return the vector db.

    args: data (str): The text data to be vectorized and stored in vector store.
    '''
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n", "\n\n", "\r", "\t", " "],
        chunk_size=1000,
        chunk_overlap=0,
    )
    text_chunks = text_splitter.split_documents(data)
    embeddings = OpenAIEmbeddings()
    vector_db = Qdrant.from_documents(
        text_chunks,
        embeddings,
        location=":memory:",  # Using in-memory storage
        collection_name="HarveDocs")

    return vector_db


def create_context_retriever_chain(vector_db):
    '''
    Get the context retriever chain to be used in the dialog chain.
    '''
    llm = HuggingFaceHub(repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
                         model_kwargs={"temperature": 0.5, "max_length": 1024})
    retriever = vector_db.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Based on the conversation above, create a search query that you will refer to, to get information that is relevant to the conversation.")
    ])

    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain


def create_dialog_rag_chain(retriever_chain):
    '''
    Get the dialog chain to be used in the chat loop.

    args: retriever_chain
    '''
    llm = HuggingFaceHub(repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
                         model_kwargs={"temperature": 0.5, "max_length": 1024})
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("system",
         "Answer the user's questions based on the context below:\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)

    return create_retrieval_chain(retriever_chain, stuff_documents_chain)


def get_response(query, vec_store, chat_history):
    '''
    Get response from the AI model

    args: query (str): The user's input
    '''
    # Dialog chain
    retrieval_chain = create_context_retriever_chain(vec_store)

    dialog_rag_chain = create_dialog_rag_chain(retrieval_chain)
    response = dialog_rag_chain.invoke({
        "chat_history": chat_history,
        "input": query
    })
    return response["answer"]
