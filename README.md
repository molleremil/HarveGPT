*UPDATE: For the time being, the model used in the app is ChatOpenAI for smoother interaction. Currently working on reimplementing an open source model for similar performance.* 

**: PDF functionality added. Harve can now chat about your PDF files.** *Check it out on HuggingFace spaces:* https://huggingface.co/spaces/emgoggles/HarveGPT

Demo:

<img width="1378" alt="Screenshot 2024-02-11 at 19 25 01" src="https://github.com/molleremil/HarveGPT/assets/139823248/d0601d3e-c149-4ec1-b780-c1863c9a9661">

--------------------------------------------------------------------------------------------------------------------------------------------

**HarveGPT:** *Harvest a source and chat with it*

A RAG-pipeline powered app designed to scrape data from URLs (including YouTube video transcripts), embed and store it in a vectorstore, capture the semantics of a user query and perform a similarity search against the stored data.

--------------------------------------------------------------------------------------------------------------------------------------------

**SETUP:**

- Create a .env file to hold your secrets as:
OPENAI_API_KEY = "*token*"

- Install the requirements by typing following in your CLI:
  
      pip install -r requirements.txt

- Then finally run the script by typing following in your CLI:

      streamlit run app.py 

--------------------------------------------------------------------------------------------------------------------------------------------

**How it works:**
Initialize the chat by providing the app a data source in form of a URL, YouTube URL or by uploading a PDF file. Then start asking questions. 

<img width="1379" alt="Screenshot 2024-02-11 at 22 21 45" src="https://github.com/molleremil/HarveGPT/assets/139823248/efd3d292-bb02-4a09-b8d7-7f5b50b2da4e">




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
