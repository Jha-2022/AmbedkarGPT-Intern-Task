# AmbedkarGPT-Intern-Task

This repository contains the solution for Assignment 1: AI Intern Hiring at Kalpit Pvt Ltd.
It implements a Retrieval-Augmented Generation (RAG) pipeline that ingests a speech by Dr. B.R. Ambedkar and answers user questions based solely on that context. The system runs entirely locally using open-source tools, requiring no API keys or cloud accounts.

## 🚀 Features
* Document Loading: Ingests text from speech.txt.
* Text Splitting: Chunks text for efficient processing.
*Embeddings: Uses sentence-transformers/all-MiniLM-L6-v2 via HuggingFace.
*Vector Store: Stores embeddings locally using ChromaDB.
*LLM: Uses Ollama running Mistral 7B for answer generation.
*Framework: Orchestrated using LangChain.
## 🛠️ Prerequisites
Before running the Python code, you must have the following installed:
Python 3.8+
Ollama (for running the LLM locally)


1. Setup Ollama
Download and install Ollama from ollama.ai. Once installed, open your terminal and pull the Mistral model:
ollama pull mistral


## 📦 Installation
Clone the repository
git clone [https://github.com/YOUR_USERNAME/AmbedkarGPT-Intern-Task.git](https://github.com/YOUR_USERNAME/AmbedkarGPT-Intern-Task.git)
cd AmbedkarGPT-Intern-Task


Create a Virtual Environment (Optional but Recommended)
# Windows
python -m venv venv
.\venv\Scripts\activate


# macOS/Linux
python3 -m venv venv
source venv/bin/activate


Install Dependencies
pip install -r requirements.txt




```sh 
pip install -r requirements.txt
python main.py

```
## This System can perform

1. Loading the provided text file (speech.txt).
2. Spliting the text into manageable chunks.
3. Creatting Embeddings and store them in a local vector store.
4. Retrieving relevant chunks based on a user's question.
5. Generating an answer by feeding the retrieved context and the question to an LLM.
