*UPDATE: For the time being, the model used in the app is ChatOpenAI for smoother interaction. Currently working on reimplementing an open source model for similar performance.* 

--------------------------------------------------------------------------------------------------------------------------------------------

**HarveGPT:** *Harvest data from websites and chat with it*

A RAG-pipeline powered app designed to scrape data from URLs, embed and store it in a vectorstore, capture the semantics of a user query and perform a similarity search against the stored data.

--------------------------------------------------------------------------------------------------------------------------------------------

**SETUP:**

- Create a .env file to hold your secrets as:
OPENAI_API_KEY = "*token*"
- Download `app.py` and `harve_funcs.py`
- Execute `app.py` to start the program.

--------------------------------------------------------------------------------------------------------------------------------------------

**How it works:**
Initialize the chat by pasting in a URL and ask questions. At the moment, HarveGPT is able to distinguish between websites with text based content and YouTube URL's and retrieve data from given source.

**DEMO:**

*Start message:*

<img width="234" alt="Screenshot 2024-02-08 at 13 57 14" src="https://github.com/molleremil/HarveGPT/assets/139823248/6ee1bf25-c7d6-488e-a0df-bc186567c6bf">

*After inputting URL:*

<img width="1140" alt="Screenshot 2024-02-09 at 17 33 45" src="https://github.com/molleremil/HarveGPT/assets/139823248/0d8eb602-d71f-489a-bbad-254ed3af9923">

*Query & Response:*

<img width="1188" alt="Screenshot 2024-02-09 at 17 35 18" src="https://github.com/molleremil/HarveGPT/assets/139823248/e6b5b5c0-978e-4548-b556-d00682e60184">





--------------------------------------------------------------------------------------------------------------------------------------------
**PREVIOUS VERSION**

The app utilizes the open source model mistralai/Mixtral-8x7B-Instruct-v0.1 from HuggingFaceHub in connection with Qdrant vectorstore, running on local mode (with in-memory storage only). 
It's already giving realitively fast and precise results, when being asked questions about long documents:

<img width="946" alt="Screenshot 2024-02-08 at 00 34 12" src="https://github.com/molleremil/URLinkGPT/assets/139823248/77954d9f-d0dc-4182-a757-e0fbe9b23bde">

--------------------------------------------------------------------------------------------------------------------------------------------

Next goals: 
1. Enable the app to take any type of data as input (text documents, audio, video etc.).
2. Implement STT->TTS functionality.
3. Implement agents for browsing functionality. 
