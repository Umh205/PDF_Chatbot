# Chat with Multiple PDFs

A Streamlit-powered chatbot that allows users to interactively query multiple PDF documents. Powered by LangChain, OpenAI embeddings, and FAISS vector stores, this app processes PDFs, splits text into chunks, and retrieves relevant answers from your documents in real-time.

---

## Features

- Upload multiple PDF files and process them in a single interface.
- Automatic text extraction and chunking for efficient retrieval.
- Embedding-based semantic search using **OpenAIEmbeddings** (or HuggingFace Instructor embeddings).
- Conversational interface with memory support, so the chatbot remembers context within a session.
- Streamlit-based GUI for a clean, interactive experience.

---

## Installation

1. Clone the repository:  
```bash
git clone <repository_url>
cd <repository_folder>

## Install dependencies:

pip install -r requirements.txt

## Or install individually:

pip install streamlit pypdf2 langchain python-dotenv faiss-cpu openai huggingface_hub InstructorEmbedding sentence_transformers

---

## Usage

1. Run the Streamlit app: streamlit run app.py
2. Open the displayed local URL in your browser.
3. Upload one or more PDFs via the sidebar.
4. Click "Process" to extract text, create embeddings, and initialize the chatbot.
5. Ask questions about your uploaded documents in the text input field.
6. View answers in the chat interface, with the conversation history preserved during the session.

---

## Dependencies

Python 3.11+
Streamlit
LangChain
PyPDF2
FAISS
OpenAI API / HuggingFace Instructor embeddings
python-dotenv
huggingface_hub
InstructorEmbedding
sentence_transformers

---

## Configuration
Store your OpenAI API key in a .env file: embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")

---

## How It Works

---

## PDF Text Extraction
All uploaded PDFs are read and combined into a single text string.

## Text Chunking
Text is split into overlapping chunks using LangChain’s `CharacterTextSplitter` for better semantic retrieval.

## Vector Store Creation
- Chunks are converted into embeddings using OpenAI or HuggingFace Instructor embeddings.  
- FAISS is used to store embeddings for fast similarity search.

## Conversation Chain
- LangChain’s `ConversationalRetrievalChain` integrates the vector store with a chatbot model (`ChatOpenAI`) to provide context-aware answers.  
- Memory support allows the chatbot to maintain conversation history during the session.

## Future Improvements
- Add support for larger PDF collections with optimized memory management.  
- Implement caching to speed up repeated queries.  
- Provide a download option for chat history.  
- Enhance UI with Streamlit components or custom styling.

