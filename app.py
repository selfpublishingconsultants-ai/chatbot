import sys
sys.stdout.reconfigure(encoding='utf-8')
from flask import Flask, request, jsonify, send_from_directory, session, redirect, url_for, render_template_string
from flask_cors import CORS
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaLLM
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import re
import json
import os
import glob
import requests
import sqlite3
from datetime import datetime
from dotenv import load_dotenv
from collections import defaultdict

load_dotenv()

app = Flask(__name__)
CORS(app)
app.secret_key = os.getenv("ADMIN_PASSWORD", "default-secret-key-123456")

# -------------------------
# SQLite Database Initialization
# -------------------------
DB_PATH = "leads.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at TEXT,
            name TEXT,
            email TEXT,
            phone TEXT,
            interested_services TEXT,
            transcript TEXT,
            summary TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

# -------------------------
# Load modular documents and create vectorstore
# -------------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

VECTOR_STORE_PATH = "vector_store/self_publishing_consultant_faiss_index"
SOURCE_DOCS_DIR = "data/source_documents"

def initialize_vector_store():
    if os.path.exists(VECTOR_STORE_PATH):
        print("[INFO] Loading existing vector store...")
        return FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        print("[INFO] Creating new vector store from modular markdown documents...")
        all_docs = []
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        
        if os.path.exists(SOURCE_DOCS_DIR):
            md_files = glob.glob(os.path.join(SOURCE_DOCS_DIR, "*.md"))
            for file_path in md_files:
                try:
                    loader = TextLoader(file_path, encoding="utf-8")
                    documents = loader.load()
                    docs = text_splitter.split_documents(documents)
                    all_docs.extend(docs)
                    print(f"[OK] Loaded {len(docs)} chunks from {os.path.basename(file_path)}")
                except Exception as e:
                    print(f"[ERROR] Failed to load {file_path}: {e}")
        else:
            print(f"[WARN] Source documents directory not found: {SOURCE_DOCS_DIR}")
        
        if all_docs:
            vectorstore = FAISS.from_documents(all_docs, embeddings)
            os.makedirs(os.path.dirname(VECTOR_STORE_PATH), exist_ok=True)
            vectorstore.save_local(VECTOR_STORE_PATH)
            print(f"[OK] Vector store saved with {len(all_docs)} total chunks")
            return vectorstore
        else:
            raise Exception("No documents found to create vector store")

vectorstore = initialize_vector_store()

# SIMPLE - Just one retriever like your original
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# -------------------------
# LLM setup - Support Ollama and Groq (OpenAI-compatible)
# -------------------------
llm_provider = os.getenv("LLM_PROVIDER", "ollama").lower()

if llm_provider == "groq":
    print("[INFO] Initializing ChatOpenAI client for Groq...")
    llm = ChatOpenAI(
        openai_api_base=os.getenv("GROQ_API_BASE", "https://api.groq.com/openai/v1"),
        openai_api_key=os.getenv("GROQ_API_KEY"),
        model_name=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        temperature=0.7
    )
else:
    print("[INFO] Initializing OllamaLLM client...")
    llm = OllamaLLM(
        model=os.getenv("OLLAMA_MODEL", "gemma3:4b"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )



conversation_prompt = PromptTemplate(
    input_variables=["context", "history", "question"],
    template="""You are a helpful assistant for Self Publishing Consultant.

You must respond ONLY in a valid JSON format with the following structure:
{{
  "reply": "Your clear, concise, and helpful response text here",
  "lead_required": true/false
}}

Do not include any explanation, markdown wrappers (like ```json), or text outside of the JSON block.

Rules for answering:
1. If the user says "you" or "you guys" it means Self Publishing Consultant.
2. Always answer clearly, concisely, and with a human tone. Never use asterisks (*), use dashes (-) or numbers instead.
3. If the question is about pricing or packages:
   - Explain that our services are modular and custom-tailored (a la carte), so authors only pay for what they need instead of rigid bundles.
   - Outline the primary service pathways:
     1. Editorial & Prep (Manuscript Assessment, Developmental Editing)
     2. Design & Production (Cover Design, universal Interior Formatting)
     3. Launch & Distribution (Global Distribution Setup, Book Launch Marketing)
   - Explain that pricing is based on the specific manuscript scope (word count, design complexity).
   - After explaining, ask: "Would you like to schedule a free consultation to get a tailored quote for your book?"
4. If the question is about services:
   - First analyze what specific type of service the user is asking about from their question.
   - If they mention a specific service category (like "editing", "marketing", "design"), filter and show only relevant services from the context.
   - If they ask generally about "services", show all available services.
   - Extract only the short service names (one per line, no descriptions).
   - Ensure each service name starts with a dash and a space.
   - After listing, ask: "Which of these services are you most interested in for your project?"
5. If the user directly mentions or selects a service (e.g., "I want to publish my book", "I need cover design", "I want ghost writing"):
   - Do not list all services.
   - Treat this as a confirmed interest in that service.
   - First, acknowledge their interest warmly (e.g., "That's great! We'd be happy to help you with [service].")
   - Ask one or two gentle, service-specific questions to better understand their requirements.
       Example: For "publish my book", ask: "Do you already have your manuscript ready, or are you still working on it?"
   - Once they respond, naturally introduce the modular options and explain that we can provide a custom quote for it.
6. If the user asks "what is [service]" or "what does [service] include" or similar informational questions:
   - Provide a clear, concise explanation of what that specific service includes based on the context.
   - Mention the key benefits and what's typically involved.
   - After explaining, ask: "Does this sound like something that would help with your project?"
   - Do not immediately jump to qualification questions.
7. If the user selects or mentions a specific service from a previous list:
   - Do not immediately push for a sale or listing.
   - First, acknowledge their interest warmly.
   - Ask one or two gentle, service-specific questions to better understand their requirements.
       Example: For "formatting", ask: "What type of document do you need formatted - manuscript, ebook, or print book?"
   - Once they respond, naturally introduce the custom-quoted modular options that fit their needs and offer a free consultation.
8. If the user expresses interest in scheduling a consultation, getting a quote, or moving forward:
   - Set "lead_required" to true in your JSON output so the frontend can display the lead capture form automatically.
   - Do not ask for contact details in text format.
9. When a user responds to clarifying questions about their service needs:
   - Analyze their response and match it to the most relevant service pathways from the context.
   - Explain briefly how these services align with their goals.
   - After explaining, ask: "Would you like to schedule a free consultation to get a custom proposal for these services?"
10. If the user greets you (hi, hello), reply politely without contact details unless they ask.
11. Provide contact details only if directly asked for them.
12. Never add "according to the context" or similar filler.
13. Always guide the conversation forward — never repeat the same question if it has already been answered.
14. If the user is exploring and not ready to commit, give concise helpful information and ask the most relevant next question.

Context: {context}

Previous conversation: {history}

Question: {question}

JSON Response:"""
)


# Store conversation history and user names
chat_sessions = defaultdict(list)
user_names = {}   # session_id -> user's first name

# Track which services/packages a user has shown interest in during conversation
session_interests = defaultdict(set)  # session_id -> set of service/package names

# -------------------------
# -------------------------
# SIMPLE intent detection - not complex
# -------------------------
def detect_intent(question, history=""):
    """Simple intent detection"""
    q_lower = question.lower()
    
    # Human agent / call / contact signals - trigger lead form immediately
    human_signals = [
        "human agent", "human", "real person", "talk to someone", "speak to someone",
        "talk to agent", "talk on call", "want a call", "want to call", "call me",
        "phone call", "whatsapp", "contact you", "reach you", "speak with",
        "connect me", "get in touch", "i want to talk", "talk to a person",
        "is there any agent", "any agent", "live agent", "live support",
        "support agent", "agent"
    ]
    if any(signal in q_lower for signal in human_signals):
        return "human"

    # Informational questions about services
    if any(phrase in q_lower for phrase in ["what is", "what does", "tell me about", "explain"]):
        return "service_info"
    
    # Quote/consultation and new buy signals (robust pattern matching)
    buy_patterns = [
        lambda q: "get" in q and "quote" in q,
        lambda q: "want" in q and "quote" in q,
        lambda q: "interest" in q and "quote" in q,
        lambda q: "book" in q and "consult" in q,
        lambda q: "schedule" in q and "consult" in q,
        lambda q: "book" in q and "call" in q,
        lambda q: "schedule" in q and "call" in q,
        lambda q: "let" in q and "start" in q,
        lambda q: "sign" in q and "up" in q,
        lambda q: "ready" in q and "start" in q,
        lambda q: "proceed" in q,
        lambda q: "let's go" in q,
    ]
    
    if any(pattern(q_lower) for pattern in buy_patterns):
        return "buy"
    
    return "general"



def extract_interests_from_message(question):
    """
    Extract mentioned services/packages from a user message to track their interests.
    Returns a set of interest strings.
    """
    interests = set()
    q_lower = question.lower()

    service_keywords = {
        "Manuscript Assessment": ["manuscript assessment", "assess my manuscript", "manuscript review"],
        "Developmental Editing": ["developmental editing", "developmental edit", "editing"],
        "Cover Design": ["cover design", "book cover", "cover"],
        "Interior Formatting": ["formatting", "interior format", "kdp format", "ebook format"],
        "Global Distribution": ["distribution", "global distribution", "distribute"],
        "Book Marketing": ["marketing", "book marketing", "launch strategy", "advertis"],
        "Royalty Accounting": ["royalty", "royalties", "royalty tracking", "royalty accounting"],
        "Ghostwriting": ["ghostwriting", "ghost writing", "ghostwrite"],
        "Copyright Registration": ["copyright", "copyright registration", "rights protection"],
        "Metadata & SEO": ["metadata", "seo", "discoverability"],
        "Audiobook Production": ["audiobook", "audio book", "audio production"],
        "Video Trailer": ["video trailer", "book trailer"],
        "Proofreading": ["proofreading", "proofread", "copyediting"],
    }

    for service, keywords in service_keywords.items():
        if any(kw in q_lower for kw in keywords):
            interests.add(service)

    return interests

# -------------------------
# Routes
# -------------------------
@app.route("/")
def index():
    return send_from_directory('frontend', 'index.html')

def parse_llm_json_response(llm_output):
    """
    Parses the JSON response from the LLM.
    Returns (reply_text, lead_required_boolean).
    """
    clean_output = llm_output.strip()
    
    # Strip markdown code block wrappers if present
    if clean_output.startswith("```"):
        # Remove opening ```json or ```
        clean_output = re.sub(r'^```(?:json)?\n', '', clean_output, flags=re.IGNORECASE)
        # Remove closing ```
        clean_output = re.sub(r'\n```$', '', clean_output)
        clean_output = clean_output.strip()
        
    try:
        data = json.loads(clean_output)
        reply = data.get("reply", "").strip()
        lead_required = bool(data.get("lead_required", False))
        if reply:
            return reply, lead_required
    except Exception as e:
        print(f"[WARN] Failed to parse LLM response as JSON: {e}. Output was: {llm_output}")
        
    # Fallback: regex search for keys
    reply_match = re.search(r'"reply"\s*:\s*"((?:[^"\\]|\\.)*)"', clean_output)
    lead_match = re.search(r'"lead_required"\s*:\s*(true|false)', clean_output, re.IGNORECASE)
    
    if reply_match:
        try:
            reply = reply_match.group(1).encode().decode('unicode-escape')
        except Exception:
            reply = reply_match.group(1)
        lead_required = False
        if lead_match:
            lead_required = lead_match.group(1).lower() == 'true'
        return reply, lead_required
        
    # Worst case fallback: treat whole output as reply
    return llm_output, False

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "").strip()
    session_id = data.get("session_id", "default")
    
    if not question:
        return jsonify({"error": "No message provided"}), 400

    # Track service interests from this message
    new_interests = extract_interests_from_message(question)
    session_interests[session_id].update(new_interests)

    # Get session history
    session_history = chat_sessions[session_id]
    
    # Simple history text (last 3 exchanges only)
    history_text = "\n".join([f"User: {msg['user']}\nBot: {msg['bot']}" 
                              for msg in session_history[-3:]])
    
    # Get context - SIMPLE, not multiple searches
    try:
        docs = retriever.invoke(question)
        context = "\n\n".join([doc.page_content for doc in docs[:2]])  # Only 2 docs
        context = context[:1000]  # Limit context size
    except Exception as e:
        context = ""
    
    # Check if this is the second message and we're waiting to capture user's name
    is_awaiting_name = session_id in user_names and user_names[session_id] == "__awaiting__"
    if is_awaiting_name:
        words = question.strip().split()
        if 1 <= len(words) <= 4 and "?" not in question and "!" not in question:
            captured_name = words[0].capitalize()
            user_names[session_id] = captured_name
            answer_text = f"Nice to meet you, {captured_name}! How can I help you with your publishing journey today?"
            session_history.append({"user": question, "bot": answer_text})
            return jsonify({"lead_required": False, "answer": answer_text, "session_id": session_id})
        else:
            user_names[session_id] = ""

    # Inject the user's known name into the history context
    known_name = user_names.get(session_id, "")
    name_context = f"The user's name is {known_name}. Use their name naturally in your replies.\n" if known_name and known_name != "__awaiting__" else ""

    # Build simple prompt
    prompt = conversation_prompt.format(
        context=name_context + context,
        history=history_text,
        question=question
    )
    
    lead_required = False
    try:
        # Get LLM response
        raw_answer = llm.invoke(prompt)
        raw_answer = raw_answer.strip()
        
        # Parse JSON response
        answer_text, lead_required = parse_llm_json_response(raw_answer)
        
    except Exception as e:
        print(f"[ERROR] LLM Invocation failed: {e}")
        import traceback
        traceback.print_exc()
        answer_text = "I'm here to help with your publishing needs. What would you like to know?"
        lead_required = False
    
    # After very first exchange: ask for user's name
    is_first_message = len(session_history) == 0
    if is_first_message and session_id not in user_names:
        user_names[session_id] = "__awaiting__"
        answer_text = answer_text + "\n\nBy the way, may I know your name? I'd love to address you personally throughout our conversation."

    # Save to session history
    session_history.append({
        "user": question,
        "bot": answer_text
    })

    
    # Keep history small
    if len(session_history) > 10:
        session_history.pop(0)
    
    return jsonify({
        "lead_required": lead_required,
        "answer": answer_text,
        "session_id": session_id
    })

# -------------------------
# Lead capture endpoint
# -------------------------
@app.route("/lead", methods=["POST"])
def lead():
    data = request.get_json()
    if not data:
        return jsonify({"errors": ["No data provided"]}), 400

    name = data.get("name", "").strip()
    email = data.get("email", "").strip()
    phone = data.get("phone", "").strip()
    session_id = data.get("session_id", "default")
    interested_services = data.get("interested_services", [])

    # Merge server-side tracked interests
    server_interests = list(session_interests.get(session_id, set()))
    all_interests = list(set(interested_services + server_interests))

    errors = []
    if not name:
        errors.append("Name is required")
    if not email and not phone:
        errors.append("Email or phone is required")

    if errors:
        return jsonify({"errors": errors}), 400

    try:
        # Get session history transcript
        session_history = chat_sessions.get(session_id, [])
        transcript_text = "\n".join([f"User: {msg['user']}\nBot: {msg['bot']}" for msg in session_history])

        # LLM Lead Extraction
        extracted_services = ", ".join(all_interests)
        extracted_summary = "General Inquiry"

        if transcript_text:
            extraction_prompt = f"""You are a helpful assistant for Self Publishing Consultant.
Analyze the following conversation between an author and our consultant chatbot:

{transcript_text}

Identify:
1. A concise 1-2 sentence summary of the author's book project, manuscript status, or goals.
2. The specific services they showed interest in.

You must respond ONLY in a valid JSON format with the following keys:
{{
  "summary": "1-2 sentence summary of their project/manuscript",
  "interested_services": "Comma-separated list of services discussed"
}}

Do not include any explanation, markdown wrappers (like ```json), or text outside of the JSON block."""
            try:
                raw_extraction = llm.invoke(extraction_prompt)
                # Parse JSON
                clean_output = raw_extraction.strip()
                if clean_output.startswith("```"):
                    clean_output = re.sub(r'^```(?:json)?\n', '', clean_output, flags=re.IGNORECASE)
                    clean_output = re.sub(r'\n```$', '', clean_output)
                    clean_output = clean_output.strip()
                
                parsed_data = json.loads(clean_output)
                extracted_summary = parsed_data.get("summary", "General Inquiry").strip()
                extracted_services = parsed_data.get("interested_services", "").strip()
                if not extracted_services:
                    extracted_services = ", ".join(all_interests)
            except Exception as e:
                print(f"[WARN] Failed to extract details via LLM: {e}. Falling back to default values.")

        # Save lead to SQLite
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO chats (started_at, name, email, phone, interested_services, transcript, summary)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            name,
            email,
            phone,
            extracted_services if extracted_services else "General Inquiry",
            transcript_text,
            extracted_summary
        ))
        conn.commit()
        conn.close()

        # Save lead to JSON file (backup)
        lead_data = {
            "name": name,
            "email": email,
            "phone": phone,
            "interested_services": extracted_services if extracted_services else "General Inquiry",
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "summary": extracted_summary,
            "transcript": transcript_text
        }
        save_lead_to_json(lead_data)

        # Clear session
        if session_id in session_interests:
            del session_interests[session_id]
        if session_id in chat_sessions:
            chat_sessions[session_id].clear()
        if session_id in user_names:
            del user_names[session_id]

        return jsonify({
            "success": True,
            "message": "Thank you! Your information has been received.",
            "next_steps": "Our team will contact you within 24 hours."
        })

    except Exception as e:
        print(f"[ERROR] Failed to save lead: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"errors": ["Failed to save. Please try again."]}), 500


def save_lead_to_json(lead_data):
    """Save lead data to JSON file"""
    LEADS_DIR = "leads"
    LEADS_JSON = os.path.join(LEADS_DIR, "lead_capture.json")
    os.makedirs(LEADS_DIR, exist_ok=True)
    try:
        existing_data = []
        if os.path.exists(LEADS_JSON):
            with open(LEADS_JSON, "r", encoding="utf-8") as f:
                content = f.read()
                if content:
                    existing_data = json.loads(content)
        
        existing_data.append(lead_data)
        
        with open(LEADS_JSON, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, indent=2)
    except Exception as e:
        print(f"[ERROR] Failed to write lead to json: {e}")
        raise


# -------------------------
# Sales team: View leads (JSON backup)
# -------------------------
@app.route("/leads/view", methods=["GET"])
def view_leads():
    if not session.get("admin_logged_in"):
        return jsonify({"error": "Unauthorized"}), 401
    
    LEADS_JSON = os.path.join("leads", "lead_capture.json")
    if not os.path.exists(LEADS_JSON):
        return jsonify({"leads": [], "total": 0})
    
    try:
        with open(LEADS_JSON, "r", encoding="utf-8") as f:
            content = f.read()
            leads = json.loads(content) if content else []
        return jsonify({"leads": leads, "total": len(leads)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -------------------------
# Admin Dashboard HTML Templates
# -------------------------
ADMIN_LOGIN_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Admin Login - Self Publishing Consultant</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400..700;1,400..700&family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-page: #0b0f19;
            --bg-surface: rgba(15, 23, 42, 0.7);
            --bg-card: rgba(30, 41, 59, 0.4);
            --border: rgba(255, 255, 255, 0.08);
            --border-glow: rgba(217, 119, 6, 0.3);
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --primary: #d97706;
            --primary-light: #fbbf24;
            --error: #ef4444;
            --font-serif: 'Lora', serif;
            --font-sans: 'Plus Jakarta Sans', sans-serif;
            --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: var(--font-sans);
            background: radial-gradient(circle at top right, #1e1b4b, var(--bg-page) 65%);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        .card {
            background: var(--bg-surface);
            backdrop-filter: blur(20px);
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 40px;
            width: 100%;
            max-width: 400px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.5);
            text-align: center;
        }
        h1 { font-family: var(--font-serif); font-size: 22px; color: var(--primary-light); margin-bottom: 8px; }
        p { font-size: 13px; color: var(--text-muted); margin-bottom: 24px; }
        .input-group { margin-bottom: 20px; text-align: left; }
        label { display: block; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; color: var(--text-muted); margin-bottom: 6px; font-weight: 600; }
        input {
            width: 100%;
            padding: 12px 16px;
            background: rgba(15, 23, 42, 0.8);
            border: 1px solid var(--border);
            border-radius: 10px;
            color: var(--text-main);
            font-size: 14px;
            outline: none;
            transition: var(--transition);
        }
        input:focus { border-color: var(--primary); box-shadow: 0 0 10px rgba(217, 119, 6, 0.15); }
        .btn {
            width: 100%;
            padding: 12px;
            background: linear-gradient(135deg, var(--primary) 0%, #b45309 100%);
            border: none;
            color: white;
            font-weight: 600;
            font-size: 14px;
            border-radius: 10px;
            cursor: pointer;
            transition: var(--transition);
            margin-top: 10px;
        }
        .btn:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(217, 119, 6, 0.3); }
        .error-msg { color: var(--error); font-size: 12px; margin-top: 12px; line-height: 1.4; }
    </style>
</head>
<body>
    <div class="card">
        <h1>Dashboard Login</h1>
        <p>Access the Self Publishing Consultant Lead Portal</p>
        <form method="POST">
            <div class="input-group">
                <label>Password</label>
                <input type="password" name="password" placeholder="Enter administrative password" required autofocus>
            </div>
            <button type="submit" class="btn">Authenticate</button>
            {% if error %}
            <div class="error-msg">{{ error }}</div>
            {% endif %}
        </form>
    </div>
</body>
</html>"""

ADMIN_CHATS_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Lead Dashboard - Self Publishing Consultant</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400..700;1,400..700&family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-page: #0b0f19;
            --bg-surface: rgba(15, 23, 42, 0.7);
            --bg-sidebar: #0f172a;
            --bg-card: rgba(30, 41, 59, 0.4);
            --bg-card-hover: rgba(30, 41, 59, 0.7);
            --border: rgba(255, 255, 255, 0.08);
            --border-glow: rgba(217, 119, 6, 0.3);
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --primary: #d97706;
            --primary-light: #fbbf24;
            --font-serif: 'Lora', serif;
            --font-sans: 'Plus Jakarta Sans', sans-serif;
            --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: var(--font-sans);
            background: radial-gradient(circle at top right, #1e1b4b, var(--bg-page) 65%);
            color: var(--text-main);
            min-height: 100vh;
            padding: 40px 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: var(--bg-surface);
            backdrop-filter: blur(20px);
            border: 1px solid var(--border);
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.5);
            overflow: hidden;
        }
        header {
            padding: 24px 40px;
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: rgba(15, 23, 42, 0.4);
        }
        h1 { font-family: var(--font-serif); font-size: 24px; color: var(--primary-light); }
        .logout-btn {
            background: transparent;
            border: 1px solid var(--border);
            color: var(--text-muted);
            padding: 8px 16px;
            border-radius: 8px;
            cursor: pointer;
            text-decoration: none;
            font-size: 13px;
            font-weight: 500;
            transition: var(--transition);
        }
        .logout-btn:hover { border-color: var(--primary); color: var(--primary-light); background: rgba(217, 119, 6, 0.1); }
        .table-container { padding: 30px; overflow-x: auto; }
        table { width: 100%; border-collapse: collapse; text-align: left; }
        th {
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            color: var(--text-muted);
            padding: 12px 18px;
            border-bottom: 1px solid var(--border);
            font-weight: 700;
        }
        td { padding: 18px; border-bottom: 1px solid var(--border); font-size: 13.5px; vertical-align: top; }
        tr:hover td { background: rgba(255,255,255,0.015); }
        .chat-row { cursor: pointer; transition: var(--transition); }
        .chat-row:hover { background: var(--bg-card-hover); }
        .name { font-weight: 600; color: var(--text-main); }
        .contact { font-size: 12.5px; color: var(--text-muted); line-height: 1.4; }
        .services { display: flex; flex-wrap: wrap; gap: 4px; }
        .badge {
            background: rgba(217,119,6,0.1);
            border: 1px solid rgba(217,119,6,0.25);
            color: var(--primary-light);
            border-radius: 12px;
            padding: 2px 8px;
            font-size: 11px;
            font-weight: 600;
        }
        .summary { color: var(--text-muted); font-size: 13px; max-width: 320px; line-height: 1.4; }
        .date { color: var(--text-muted); font-size: 12.5px; white-space: nowrap; }
        .action-link {
            color: var(--primary-light);
            text-decoration: none;
            font-weight: 600;
            display: inline-flex;
            align-items: center;
            gap: 4px;
            transition: var(--transition);
        }
        .action-link:hover { color: var(--primary); text-decoration: underline; }
        .no-leads { text-align: center; padding: 60px; color: var(--text-muted); font-size: 15px; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Lead Dashboard — Self Publishing Consultant</h1>
            <a href="/admin/logout" class="logout-btn">Log Out</a>
        </header>
        <div class="table-container">
            {% if chats %}
            <table>
                <thead>
                    <tr>
                        <th>Date & Time</th>
                        <th>Author Name</th>
                        <th>Contact Details</th>
                        <th>Requested Services</th>
                        <th>Project Summary</th>
                        <th>Actions</th>
                    </tr>
                </thead>
                <tbody>
                    {% for chat in chats %}
                    <tr class="chat-row" onclick="window.location.href='/admin/chats/{{ chat.id }}'">
                        <td class="date">{{ chat.started_at }}</td>
                        <td class="name">{{ chat.name }}</td>
                        <td class="contact">
                            <div>📧 {{ chat.email }}</div>
                            {% if chat.phone %}<div>📞 {{ chat.phone }}</div>{% endif %}
                        </td>
                        <td class="services">
                            {% if chat.interested_services %}
                                {% for svc in chat.interested_services.split(',') %}
                                    <span class="badge">{{ svc.strip() }}</span>
                                {% endfor %}
                            {% else %}
                                <span class="badge" style="background:rgba(255,255,255,0.05);border-color:transparent;color:var(--text-muted);">General</span>
                            {% endif %}
                        </td>
                        <td class="summary">{{ chat.summary }}</td>
                        <td><a href="/admin/chats/{{ chat.id }}" class="action-link">View Details →</a></td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
            {% else %}
            <div class="no-leads">No leads captured yet. Keep testing!</div>
            {% endif %}
        </div>
    </div>
</body>
</html>"""

ADMIN_DETAIL_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Lead Details - Self Publishing Consultant</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400..700;1,400..700&family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-page: #0b0f19;
            --bg-surface: rgba(15, 23, 42, 0.7);
            --bg-sidebar: #0f172a;
            --bg-card: rgba(30, 41, 59, 0.4);
            --bg-card-hover: rgba(30, 41, 59, 0.7);
            --border: rgba(255, 255, 255, 0.08);
            --border-glow: rgba(217, 119, 6, 0.3);
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --primary: #d97706;
            --primary-light: #fbbf24;
            --font-serif: 'Lora', serif;
            --font-sans: 'Plus Jakarta Sans', sans-serif;
            --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: var(--font-sans);
            background: radial-gradient(circle at top right, #1e1b4b, var(--bg-page) 65%);
            color: var(--text-main);
            min-height: 100vh;
            padding: 40px 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .back-link {
            color: var(--text-muted);
            text-decoration: none;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            font-size: 14px;
            font-weight: 500;
            margin-bottom: 24px;
            transition: var(--transition);
        }
        .back-link:hover { color: var(--primary-light); }
        .grid {
            display: grid;
            grid-template-columns: 420px 1fr;
            gap: 30px;
        }
        .card {
            background: var(--bg-surface);
            backdrop-filter: blur(20px);
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.5);
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        .card h2 { font-family: var(--font-serif); font-size: 22px; color: var(--primary-light); border-bottom: 1px solid var(--border); padding-bottom: 10px; }
        .field { display: flex; flex-direction: column; gap: 4px; }
        .field-label { font-size: 10px; text-transform: uppercase; letter-spacing: 1px; color: var(--text-muted); font-weight: 700; }
        .field-value { font-size: 14px; color: var(--text-main); line-height: 1.4; }
        .services { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 4px; }
        .badge {
            background: rgba(217,119,6,0.1);
            border: 1px solid rgba(217,119,6,0.25);
            color: var(--primary-light);
            border-radius: 12px;
            padding: 2px 8px;
            font-size: 11px;
            font-weight: 600;
        }
        .chat-feed {
            background: var(--bg-surface);
            backdrop-filter: blur(20px);
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.5);
            display: flex;
            flex-direction: column;
            gap: 20px;
            max-height: 700px;
            overflow-y: auto;
        }
        .chat-feed h2 { font-family: var(--font-serif); font-size: 20px; color: var(--primary-light); margin-bottom: 10px; }
        .message { display: flex; flex-direction: column; gap: 6px; max-width: 80%; padding: 12px 16px; border-radius: 14px; font-size: 13.5px; line-height: 1.5; }
        .message.user {
            align-self: flex-end;
            background: linear-gradient(135deg, var(--primary) 0%, #b45309 100%);
            color: white;
            border-bottom-right-radius: 2px;
        }
        .message.bot {
            align-self: flex-start;
            background: rgba(30, 41, 59, 0.6);
            border: 1px solid var(--border);
            color: var(--text-main);
            border-bottom-left-radius: 2px;
        }
        .message-sender { font-size: 10px; font-weight: 700; text-transform: uppercase; opacity: 0.8; letter-spacing: 0.5px; }
        .message-text { word-wrap: break-word; }
        @media (max-width: 900px) {
            .grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="container">
        <a href="/admin/chats" class="back-link">← Back to Lead Dashboard</a>
        <div class="grid">
            <!-- Left Panel: Extracted Details -->
            <div class="card">
                <h2>Lead Details</h2>
                <div class="field">
                    <div class="field-label">Date Captured</div>
                    <div class="field-value">{{ chat.started_at }}</div>
                </div>
                <div class="field">
                    <div class="field-label">Author Name</div>
                    <div class="field-value" style="font-size: 16px; font-weight:600; color: var(--primary-light);">{{ chat.name }}</div>
                </div>
                <div class="field">
                    <div class="field-label">Email Address</div>
                    <div class="field-value">{{ chat.email }}</div>
                </div>
                {% if chat.phone %}
                <div class="field">
                    <div class="field-label">Phone Number</div>
                    <div class="field-value">{{ chat.phone }}</div>
                </div>
                {% endif %}
                <div class="field">
                    <div class="field-label">Interested Services</div>
                    <div class="services">
                        {% if chat.interested_services %}
                            {% for svc in chat.interested_services.split(',') %}
                                <span class="badge">{{ svc.strip() }}</span>
                            {% endfor %}
                        {% else %}
                            <span class="badge" style="background:rgba(255,255,255,0.05);border-color:transparent;color:var(--text-muted);">General</span>
                        {% endif %}
                    </div>
                </div>
                <div class="field">
                    <div class="field-label">Project Summary</div>
                    <div class="field-value" style="font-style: italic; color: var(--text-muted); border-left: 2px solid var(--primary); padding-left: 10px;">{{ chat.summary }}</div>
                </div>
            </div>

            <!-- Right Panel: Full Conversation Transcript -->
            <div class="chat-feed">
                <h2>Conversation Transcript</h2>
                {% if transcript_list %}
                    {% for msg in transcript_list %}
                        {% if msg.sender.lower() == 'user' %}
                            <div class="message user">
                                <div class="message-sender">Author</div>
                                <div class="message-text">{{ msg.text }}</div>
                            </div>
                        {% else %}
                            <div class="message bot">
                                <div class="message-sender">Assistant</div>
                                <div class="message-text">{{ msg.text }}</div>
                            </div>
                        {% endif %}
                    {% endfor %}
                {% else %}
                    <div style="color:var(--text-muted); font-size: 14px;">No conversation logs found.</div>
                {% endif %}
            </div>
        </div>
    </div>
</body>
</html>"""


# -------------------------
# Admin Portal Routes
# -------------------------
@app.route("/admin", methods=["GET", "POST"])
def admin_login():
    error = None
    if request.method == "POST":
        password = request.form.get("password")
        if password == os.getenv("ADMIN_PASSWORD", "admin123"):
            session["admin_logged_in"] = True
            return redirect(url_for("admin_chats"))
        else:
            error = "Invalid administrative password"
    return render_template_string(ADMIN_LOGIN_HTML, error=error)

@app.route("/admin/chats", methods=["GET"])
def admin_chats():
    if not session.get("admin_logged_in"):
        return redirect(url_for("admin_login"))
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM chats ORDER BY started_at DESC")
        chats = cursor.fetchall()
        conn.close()
    except Exception as e:
        print(f"[ERROR] Failed to query leads: {e}")
        chats = []
    return render_template_string(ADMIN_CHATS_HTML, chats=chats)

@app.route("/admin/chats/<int:chat_id>", methods=["GET"])
def admin_chat_detail(chat_id):
    if not session.get("admin_logged_in"):
        return redirect(url_for("admin_login"))
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM chats WHERE id = ?", (chat_id,))
        chat = cursor.fetchone()
        conn.close()
        
        if not chat:
            return "Chat not found", 404
            
        transcript_list = []
        raw_transcript = chat["transcript"]
        if raw_transcript:
            lines = raw_transcript.split('\n')
            for line in lines:
                if line.startswith("User: "):
                    transcript_list.append({"sender": "User", "text": line[6:]})
                elif line.startswith("Bot: "):
                    transcript_list.append({"sender": "Bot", "text": line[5:]})
    except Exception as e:
        print(f"[ERROR] Failed to fetch lead detail: {e}")
        return "Internal server error", 500
        
    return render_template_string(ADMIN_DETAIL_HTML, chat=chat, transcript_list=transcript_list)

@app.route("/admin/logout", methods=["GET"])
def admin_logout():
    session.pop("admin_logged_in", None)
    return redirect(url_for("admin_login"))

# -------------------------
# Widget Route
# -------------------------
@app.route("/widget", methods=["GET"])
def widget():
    """
    Serves the minimal chat-only interface for embedding in standard iFrames.
    """
    return send_from_directory('frontend', 'widget.html')

# -------------------------
# Debug endpoints
# -------------------------
@app.route("/test-llm", methods=["GET"])
def test_llm():
    try:
        response = llm.invoke("Say hello")
        return jsonify({"status": "success", "response": response})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)})

@app.route("/clear_session", methods=["POST"])
def clear_session():
    data = request.get_json()
    session_id = data.get("session_id", "default")
    
    if session_id in chat_sessions:
        chat_sessions[session_id].clear()
    
    # Also clear the stored name so it can be captured fresh
    if session_id in user_names:
        del user_names[session_id]

    # Clear tracked interests
    if session_id in session_interests:
        del session_interests[session_id]
    
    return jsonify({"message": "Session cleared"})

if __name__ == "__main__":
    app.run(debug=True, port=5000)

