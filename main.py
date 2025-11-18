import sys
from langchain.document_loaders import TextLoader
from langchain.text_splitters import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import Ollama
from langchain.chains import RetrievalQA

def create_rag_pipeline():
    """
    Builds the full RAG (Retrieval-Augmented Generation) pipeline.
    """
    try:
        # 1. Load the provided text file
        loader = TextLoader('./speech.txt')
        documents = loader.load()
        
        if not documents:
            print("Error: Could not load speech.txt. Make sure the file exists and is not empty.")
            return None

        # 2. Split the text into manageable chunks
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(documents)
        
        if not chunks:
            print("Error: Failed to split the document into chunks.")
            return None

        # 3. Create Embeddings
        # Using the specified model: sentence-transformers/all-MiniLM-L6-v2
        # This runs locally and requires no API keys.
        print("Loading embedding model (all-MiniLM-L6-v2)...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}  # Use CPU for broader compatibility
        )
        print("Embedding model loaded.")

        # 4. Store chunks in a local vector store (ChromaDB)
        # This creates an in-memory vector store.
        print("Creating vector store...")
        vector_store = Chroma.from_documents(chunks, embeddings)
        print("Vector store created.")

        # 5. Initialize the LLM (Ollama with Mistral)
        # Assumes Ollama is running and has the 'mistral' model pulled.
        print("Initializing Ollama with Mistral model...")
        try:
            llm = Ollama(model="mistral")
            # Test connection
            llm.invoke("Hello") 
            print("Ollama connection successful.")
        except Exception as e:
            print(f"\n--- ERROR ---")
            print(f"Failed to connect to Ollama: {e}")
            print("Please ensure Ollama is running and you have pulled the 'mistral' model.")
            print("You can run it using: `ollama serve`")
            print("And pull the model using: `ollama pull mistral`")
            print("---------------\n")
            return None

        # 6. Create the RetrievalQA chain
        # This chain links the LLM with the retriever (our vector store)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",  # "stuff" puts all retrieved chunks into the prompt
            retriever=vector_store.as_retriever()
        )
        
        print("Q&A system is ready.")
        return qa_chain

    except ImportError:
        print("\n--- ERROR ---")
        print("One or more required Python packages are not installed.")
        print("Please install all dependencies from requirements.txt:")
        print("pip install -r requirements.txt")
        print("---------------\n")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during setup: {e}")
        return None

def main():
    """
    Main function to run the command-line Q&A interface.
    """
    qa_chain = create_rag_pipeline()
    
    if qa_chain is None:
        print("Failed to initialize the Q&A system. Exiting.")
        return

    print("\n--- AmbedkarGPT Q&A System ---")
    print("Ask questions based on the provided speech. Type 'exit' to quit.")
    
    while True:
        try:
            query = input("\nYour question: ")
            
            if query.lower().strip() == 'exit':
                print("Exiting. Goodbye!")
                break
                
            if not query.strip():
                continue

            print("Generating answer...")
            
            # 5. Generate an answer
            # The chain automatically handles retrieval and generation
            result = qa_chain.invoke({"query": query})
            
            print("\n--- Answer ---")
            print(result['result'])
            print("--------------")

        except KeyboardInterrupt:
            print("\nExiting. Goodbye!")
            break
        except Exception as e:
            print(f"An error occurred while processing your question: {e}")

if __name__ == "__main__":
    main()
