from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv

# Load secrets from .env file
load_dotenv()


def create_vectordb(url):
    '''
    1. Extract the url content and return as a document,
    2. split document into text chunks,
    3. vectorize text chunks and store in a vector db

    args: url (str): The url of the web page to extract content from
    '''
    loader = WebBaseLoader(url)
    doc = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n", "\n\n", "\r", "\t", " "],
        chunk_size=1000,
        chunk_overlap=0,
    )
    text_chunks = text_splitter.split_documents(doc)
    embeddings = OpenAIEmbeddings()
    vector_db = Qdrant.from_documents(
        text_chunks,
        embeddings,
        location=":memory:",  # Using in-memory storage
        collection_name="WebChatDocuments")

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
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation.")
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
         "Answer the user's questions based on the below context:\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)

    return create_retrieval_chain(retriever_chain, stuff_documents_chain)


def get_response(query):
    '''
    Get response from the AI model

    args: query (str): The user's input
    '''
    # Dialog chain
    retrieval_chain = create_context_retriever_chain(vec_store)

    dialog_rag_chain = create_dialog_rag_chain(retrieval_chain)
    response = dialog_rag_chain.invoke({
        "chat_history": chat_history,
        "input": user_input
    })
    return response["answer"]


# Start the chat
print("Welcome to HarveGPT!")


# Get the URL
url = input("Please enter a link (URL): ")


# User input
if url.strip() != "" or url is not None:
    chat_history = []
else:
    print("Please enter a link to start the chat.")

vec_store = create_vectordb(url)

# Chat loop
keep_chatting = True
while keep_chatting:
    user_input = input("Type a message (or type 'exit' to quit):\n")
    if user_input.lower().strip() == "exit":
        keep_chatting = False
    elif user_input.strip() != "":
        response = get_response(user_input)
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response))
    else:
        print("Please enter a message to continue the chat.")

    # Dialog flow
    for message in chat_history:
        if isinstance(message, HumanMessage):
            print(f"\nQUESTION: {message.content}\n")
        elif isinstance(message, AIMessage):
            print(f"\nANSWER: {message.content}\n")
