# AmbedkarGPT-Intern-Task

This repository contains the solution for Assignment 1: AI Intern Hiring at Kalpit Pvt Ltd.
It implements a Retrieval-Augmented Generation (RAG) pipeline that ingests a speech by Dr. B.R. Ambedkar and answers user questions based solely on that context. The system runs entirely locally using open-source tools, requiring no API keys or cloud accounts.

## 🚀 Features
* Document Loading: Ingests text from speech.txt.
* Text Splitting: Chunks text for efficient processing.
* Embeddings: Uses sentence-transformers/all-MiniLM-L6-v2 via HuggingFace.
* Vector Store: Stores embeddings locally using ChromaDB.
* LLM: Uses Ollama running Mistral 7B for answer generation.
* Framework: Orchestrated using LangChain.

## 🛠️ Prerequisites
Before running the Python code, you must have the following installed:
* Python 3.8+
* Ollama (for running the LLM locally)


1. Setup Ollama <br>
 Download and install Ollama from ollama.ai. Once installed, open your terminal and pull the Mistral model:
```
ollama pull mistral
```

## 📦 Installation
1. Clone the repository
```
git clone https://github.com/Jha-2022/AmbedkarGPT-Intern-Task.git
cd AmbedkarGPT-Intern-Task
```

2. Create a Virtual Environment (Optional but Recommended)
   
### Windows
```
python -m venv venv
.\venv\Scripts\activate
```

### macOS/Linux
```
python3 -m venv venv
source venv/bin/activate
```

Install Dependencies

```sh 
pip install -r requirements.txt
python main.py

```


🏃‍♂️ Usage

1. Ensure the ```speech.txt``` file is present in the root directory (included in this repo).

2. Run the main script:
```
python main.py
```

3. The system will initialize the vector store and LLM. Once ready, type your question when prompted.

* Type ```exit``` or ```quit``` to close the application.

Example Interaction <br>
```
🤖 AmbedkarGPT Ready! (Type 'exit' to quit)
========================================

Your Question: What is the real enemy?

Answer: According to the text, the real enemy is the belief in the shastras.
```

## 📂 Project Structure

* ```main.py```: The core logic containing the LangChain RAG pipeline.

* ```speech.txt```: The source text file containing the excerpt from "Annihilation of Caste".

* ```requirements.txt```: List of Python dependencies.

README.md: Project documentation.

## 🧩 Tech Stack

* **LangChain:** Framework for LLM applications.

* **ChromaDB:** Vector database for retrieving context.

* **HuggingFace Embeddings:** For converting text to vectors (all-MiniLM-L6-v2).

* **Ollama:** Local LLM runner.
