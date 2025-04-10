# RAG-Based QA System

This project demonstrates a comparison between **Naive Retrieval-Augmented Generation (RAG)** and **Re-ranked RAG** for question-answering tasks. It uses a local language model, embeddings, and a vector store to retrieve and rank relevant documents for generating concise answers.

![image](https://github.com/user-attachments/assets/535656bc-b788-411a-9a9c-ab148d4a52b8)

![image](https://github.com/user-attachments/assets/5ea5f4aa-455c-4155-8f82-abaebcb2f528)



## Features

- **Naive RAG**: Retrieves documents and generates answers directly.
- **Re-ranked RAG**: Re-ranks retrieved documents using a cross-encoder for improved relevance before generating answers.
- **Interactive Chat Interface**: Built using Gradio for user interaction.
- **Dynamic Vector Store Management**: Automatically builds or loads a vector store for document retrieval.

## Requirements

- Python 3.8 or higher
- Required Python libraries:
  - `gradio`
  - `langchain-ollama`
  - `langchain-nomic`
  - `chromadb`
  - `sentence-transformers`
  - `numpy`
- **Ollama**: A local language model server.

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone `https://github.com/kdshreyas/rag-comparative-demo`
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install Ollama:
   - Visit the [Ollama website](https://ollama.ai) and follow the installation instructions for your operating system.

4. Serve the Ollama model:
   - Start the Ollama server with the required model:
     ```bash
     ollama serve gemma3:latest
     ```

5. Ensure the vector store is built:
   - The script will automatically build the vector store if it doesn't exist in the `vectorstore` directory.

6. Run the chatbot:
   ```bash
   python main.py
   ```

## Usage

- Launch the chatbot interface in your browser.
- Ask questions related to insurance policies, such as:
  - "What is Insured's Declared Value (IDV)?"
  - "How is No Claim Bonus (NCB) calculated?"
- The chatbot will provide answers using both Naive RAG and Re-ranked RAG approaches.

## Project Structure

- `main.py`: Main script containing the chatbot logic.
- `build_vector_store.py`: Script to build the vector store for document retrieval.
- `vectorstore/`: Directory to store the persistent vector store.

## Example Questions

- "What are some general exceptions under the policy?"
- "What is the geographical scope of the policy?"
- "What coverage is provided under the Personal Accident Cover for the Owner-Driver?"

## Acknowledgments

This project uses the following models and libraries:
- **ChatOllama** for local language model inference.
- **Nomic Embeddings** for document embeddings.
- **CrossEncoder** for re-ranking retrieved documents.
- **Gradio** for building the interactive chatbot interface.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
