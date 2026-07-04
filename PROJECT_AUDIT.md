# Chatbot RAG Pipeline Project Audit

## 1. Project Structure
Below is a tree view of the essential files and folders relevant to the RAG chatbot pipeline:

```text
my-rag-based-chatbot/
├── .env                  # Environment variables
├── app.py                # Main Flask backend, RAG logic, and API endpoints
├── requirements.txt      # Python dependencies
├── leads.db              # SQLite database storing lead data
├── data/
│   └── source_documents/ # Markdown files defining company services and info (ingestion data)
├── frontend/
│   ├── index.html        # Main landing page for chatbot
│   ├── widget.html       # Minimal iFrame widget UI
│   └── widget.js         # Frontend interactions and API calls
├── vector_store/
│   └── self_publishing_consultant_faiss_index/
│       ├── index.faiss   # Vector store data (created dynamically)
│       └── index.pkl     # FAISS metadata/index
└── leads/
    └── lead_capture.json # JSON backup of collected leads (created dynamically)
```

## 2. Data Flow
Step-by-step trace of how a user query moves through the system:
1. **User Input:** The user types a message in the frontend UI and hits submit. A POST request is sent to the backend `/ask` endpoint.
2. **Intent & Interest Extraction:** The backend performs lightweight keyword matching to track which publishing services the user mentioned.
3. **Retrieval:** The backend queries the FAISS vector store using the `retriever.invoke(question)` method to find relevant chunks from the ingested markdown files.
4. **Context Building:** The top 2 retrieved document chunks are joined into a single string (capped at 1,000 characters) to form the context. The user's name and recent conversation history are also retrieved from memory.
5. **Model Call:** A prompt is formatted with instructions to return a JSON object (containing a `reply` string and a `lead_required` boolean). This prompt is sent synchronously to the configured LLM.
6. **Response Generation:** The LLM's raw response is parsed to extract the JSON. If valid JSON isn't detected, it falls back to a regex search. The final response is appended to the session history and returned to the frontend.
7. **Lead Capture (Optional):** If the LLM sets `lead_required = true`, the frontend prompts the user for contact details. Submitting this sends a POST to `/lead`, where a second LLM call summarizes the conversation, and the lead is saved to the SQLite DB.

## 3. Connected Models & APIs
- **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (Running locally via HuggingFaceEmbeddings)
  - Initialized in `app.py` (line 56)
- **Local LLM (Development):** `gemma3:4b` running on `localhost:11434` via Ollama
  - Initialized in `app.py` (line 112)
- **Production LLM (Third-Party API):** `llama-3.3-70b-versatile` via Groq (`https://api.groq.com/openai/v1`)
  - Initialized in `app.py` (line 105)

**Where LLMs are invoked in the code:**
- `app.py` > `ask()` (line 378): Generates the chatbot's response to the user query.
- `app.py` > `lead()` (line 469): Analyzes the conversation transcript to generate a 1-2 sentence project summary and extract interested services.

## 4. Environment Variables / API Keys
- `LLM_PROVIDER`: Toggles between 'ollama' and 'groq' (`app.py`)
- `OLLAMA_MODEL`: Sets the local model name (`app.py`)
- `OLLAMA_BASE_URL`: Sets the Ollama local host URL (`app.py`)
- `GROQ_API_BASE`: Sets the base URL for Groq's OpenAI-compatible API (`app.py`)
- `GROQ_MODEL`: Sets the model to use on Groq (`app.py`)
- `GROQ_API_KEY`: Authentication key for Groq (`app.py`)
- `ADMIN_PASSWORD`: Used as the Flask secret key and for protecting the lead dashboard (`app.py`)

## 5. Vector Store & Retrieval Setup
- **Technology:** FAISS (Facebook AI Similarity Search) via `langchain_community`
- **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Chunking Strategy:** `RecursiveCharacterTextSplitter` with `chunk_size=1000` and `chunk_overlap=200`
- **Top-K Setting:** The retriever is configured as `k=3`, but during actual execution in `/ask`, the code manually slices the response to only use the top 2 documents (`docs[:2]`) and heavily truncates the final context string to a maximum of 1,000 characters.

## 6. AI/RAG Dependencies
Extracted from `requirements.txt`:
- `langchain`
- `langchain-community`
- `sentence-transformers`
- `faiss-cpu`
- `pypdf`
- `ollama`
- `langchain-huggingface`
- `langchain-ollama`
- `langchain-openai`

## 7. Known Bottlenecks & Potential Issues
- **Localhost Dependency & Ollama:** If `LLM_PROVIDER` is set to `ollama`, the server routes calls to `http://localhost:11434`. This will cause extremely slow responses if the host machine lacks a dedicated GPU.
- **Synchronous Execution:** LLM invocations (`llm.invoke`) run synchronously inside the Flask routes. When one user makes a request, the entire server thread is blocked until the LLM replies. This severely limits concurrent users.
- **In-Memory Chat History:** The `chat_sessions` dict stores chat history in memory. If the app is restarted, all active user contexts vanish. If running long-term without cleanup, it could cause memory leaks.
- **Aggressive Context Truncation:** The retrieved context is hard-sliced to only 1,000 characters total (`context[:1000]`). If chunk 1 is exactly 1,000 chars, chunk 2 is completely discarded, which might hide relevant information from the LLM.
- **LLM JSON Parsing Reliance:** The prompt explicitly asks for JSON, but the backend parsing relies heavily on custom regex fallbacks. If the LLM generates conversational text instead of raw JSON, the regex parser might fail to extract `reply` or `lead_required` properly.
- **SQLite Concurrency:** Saving leads to SQLite (`leads.db`) lacks proper thread pooling, which could result in "database locked" errors under simultaneous load.
