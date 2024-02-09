# Import harve funcs
from harve_funcs import extract_data_from_url, extract_transcript_from_youtube_url, create_vectorstore_from_data, get_response
from langchain_core.messages import AIMessage, HumanMessage
import os
import sys


# Start the chat
print("Welcome to HarveGPT!")


# Get the URL
def input_url():
    url = input(
        "Please enter a link (URL) to start the chat or type 'exit' to quit: ")

    # User input
    if url == "" or url is None:
        print("Invalid URL. Try again.")
        input_url()

    else:
        chat_history = []
        try:
            if "exit" in url:
                exit()
            elif "youtube.com" in url or "youtu.be" in url:
                data = extract_transcript_from_youtube_url(url)
            else:
                data = extract_data_from_url(url)
        except Exception as e:
            print(f"Error: {e}")
            input_url()

        vec_store = create_vectorstore_from_data(data)

        # Chat loop
        keep_chatting = True
        while keep_chatting:
            user_input = input(
                "\nOPTIONS:\n> Type 'exit' to quit\n> Type 'restart' to enter a new URL\n\nType your message: ")
            if user_input.lower().strip() == "exit":
                keep_chatting = False
            elif user_input.lower().strip() == "restart":
                os.execl(sys.executable, sys.executable, *sys.argv)
            elif user_input.strip() != "":
                response = get_response(user_input, vec_store, chat_history)
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


input_url()
