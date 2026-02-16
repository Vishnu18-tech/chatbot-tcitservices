from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from dotenv import load_dotenv
from google import genai
from collections import deque
import os
import time

# Load Environment Variables
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
pinecone_key = os.getenv("PINECONE_API_KEY")
pinecone_index = os.getenv("PINECONE_INDEX")

print(f"✓ .env loaded - Gemini key present: {bool(api_key)}")

client = genai.Client(api_key=api_key)
embedder = SentenceTransformer("all-MiniLM-L6-v2")  # 384 dims
pc = Pinecone(api_key=pinecone_key)
index = pc.Index(pinecone_index)

# 6-Hour Session Memory
SESSION_DURATION = 6 * 60 * 60  # 6 hours
sessions = {}

def get_memory(session_id: str):
    now = time.time()

    if session_id in sessions:
        session = sessions[session_id]

        # If session still valid
        if now - session["last_active"] < SESSION_DURATION:
            session["last_active"] = now
            return session["memory"]

    # Create new session
    sessions[session_id] = {
        "memory": deque(maxlen=6),
        "last_active": now
    }

    return sessions[session_id]["memory"]

# Retrieval Function
def retrieve_context(query: str, top_k: int = 4) -> str:
    query_vector = embedder.encode(query).tolist()

    res = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )

    matches = res.get("matches", [])
    if not matches:
        return ""

    return "\n\n".join(
        match["metadata"]["text"]
        for match in matches
        if "text" in match["metadata"]
    )

# Intent Detection
def is_vague_query(query: str) -> bool:
    vague_patterns = [
        "tell me more",
        "more details",
        "elaborate",
        "continue",
        "expand",
        "explain more"
    ]
    q = query.lower()
    return any(p in q for p in vague_patterns)

def is_memory_question(query: str) -> bool:
    memory_keywords = [
        "previous question",
        "what did i ask",
        "what was my last question",
        "repeat my question",
        "earlier question"
    ]
    q = query.lower()
    return any(k in q for k in memory_keywords)

# Chat Function
def chat(query: str, session_id: str = "default") -> str:

    memory = get_memory(session_id)

    # Memory-only questions
    if is_memory_question(query):
        for role, message in reversed(memory):
            if role == "User":
                return f"Your previous question was: {message}"
        return "You haven't asked any previous question yet."

    # Follow-up (only if vague)
    if is_vague_query(query) and memory:
        last_user_question = None
        for role, message in reversed(memory):
            if role == "User":
                last_user_question = message
                break

        if last_user_question:
            combined_query = f"{last_user_question}. {query}"
            context = retrieve_context(combined_query)
        else:
            context = retrieve_context(query)
    else:
        # Pure RAG
        context = retrieve_context(query)

    # Fallback if nothing found
    if not context.strip():
        return "I don’t have that information based on the TC IT Services knowledge base."

    prompt = f"""
You are an AI assistant for TC IT Services.

Use ONLY the provided context to answer the question.

If the answer is explicitly stated OR can be reasonably inferred from the context,
provide a clear and structured answer.

Do not add external knowledge.
Do not guess beyond the given context.

Context:
{context}

Question:
{query}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash-lite-preview-09-2025",
        contents=prompt
    )

    answer = response.text.strip()

    # Store in session memory
    memory.append(("User", query))
    memory.append(("Bot", answer))
    print(f"Memory call - {memory}")
    return answer

# ---------------------------
# CLI Testing
# ---------------------------
if __name__ == "__main__":
    print("\n🤖 TC IT Services RAG Chatbot (Gemini)")
    print("Type 'exit' or 'quit' to stop\n")

    session_id = "local-user"  # CLI single user session

    while True:
        user_input = input("User: ").strip()

        if user_input.lower() in ["exit", "quit"]:
            print("👋 Chat ended.")
            break

        if user_input.lower() in ["hi", "hello", "hey"]:
            print("Bot: Hello! I'm here to give information about TC IT Services.\n")
            continue

        if user_input.lower() in ["what can you do", "who are you", "how can you help me"]:
            print("Bot: I'm here to give information about TC IT Services.\n")
            continue

        print("Bot:", chat(user_input, session_id=session_id), "\n")