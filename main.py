import streamlit as st
import os
import faiss
import numpy as np
import pickle
from anthropic import Anthropic
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
from PIL import Image
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# # Set API Keys
ANTHROPIC_API_KEY = st.secrets("ANTHROPIC_API_KEY")
OPENAI_API_KEY = st.secrets("OPENAI_API_KEY")




# Set page config
st.set_page_config(page_title="HAI RAG Chat", layout="centered")

# Load avatars
user_avatar = Image.open("Hilka.JPG")
bot_avatar = Image.open("Hilka.JPG")

st.image(bot_avatar, width=100)
st.title("HAI Therapy 🐧")
st.markdown("**Ask me anything about CBT. I'm here to help.**")

TEXT_FOLDER = "Resources\\Cognitive Therapy Behavior"
INDEX_FILE = "faiss_index_Cognitive_Therapy_Behavior.bin"
METADATA_FILE = "metadata_Cognitive_Therapy_Behavior.pkl"

# Load documents
all_texts, file_names = [], []
for filename in os.listdir(TEXT_FOLDER):
    if filename.endswith(".txt"):
        with open(os.path.join(TEXT_FOLDER, filename), "r", encoding="utf-8", errors="ignore") as file:
            all_texts.append(file.read())
            file_names.append(filename)

class Document(BaseModel):
    content: str
    filename: str

documents = [Document(content=text, filename=name) for text, name in zip(all_texts, file_names)]

def save_faiss_index(index, file_path):
    faiss.write_index(index, file_path)

def load_faiss_index(file_path):
    return faiss.read_index(file_path)

def save_metadata(metadata, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(metadata, f)

def load_metadata(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)

model = SentenceTransformer("all-MiniLM-L6-v2")

# FAISS index
if os.path.exists(INDEX_FILE) and os.path.exists(METADATA_FILE):
    index = load_faiss_index(INDEX_FILE)
    documents = load_metadata(METADATA_FILE)
else:
    embeddings = model.encode([doc.content for doc in documents])
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings, dtype=np.float32))
    save_faiss_index(index, INDEX_FILE)
    save_metadata(documents, METADATA_FILE)

client = Anthropic(api_key=ANTHROPIC_API_KEY)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "feedback_log" not in st.session_state:
    st.session_state.feedback_log = []

def retrieve_and_answer(query):
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding, dtype=np.float32), k=3)
    retrieved_texts = [documents[i].content for i in I[0] if i < len(documents)]
    context = "\n\n".join(retrieved_texts)

    messages = st.session_state.chat_history + [
        {"role": "user", "content": f"The question is in Persian. Translate to English and answer using this context:\n\n{context}\n\nQuestion: {query}"}
    ]

    response = client.messages.create(
        model="claude-2",
        max_tokens=1000,
        system="You are an AI assistant that provides accurate responses based on retrieved documents.",
        messages=messages
    )

    reply = "".join(part.text for part in response.content).strip()
    st.session_state.chat_history.append({"role": "user", "content": query})
    st.session_state.chat_history.append({"role": "assistant", "content": reply})
    return reply

def reward_model(feedback):
    return 1 if feedback == "👍 Yes" else -1

def save_feedback_log(log, file_path="feedback_log.jsonl"):
    with open(file_path, "a", encoding="utf-8") as f:
        for entry in log:
            json.dump(entry, f)
            f.write("\n")

# Show chat history
for i, msg in enumerate(st.session_state.chat_history):
    with st.chat_message("user" if msg["role"] == "user" else "assistant",
                         avatar=user_avatar if msg["role"] == "user" else bot_avatar):
        st.markdown(msg["content"])

    if msg["role"] == "assistant" and i == len(st.session_state.chat_history) - 1:
        feedback = st.radio("Was this response helpful?", ["👍 Yes", "👎 No"], horizontal=True, key=f"feedback_{i}")
        st.session_state.feedback_log.append({
            "query": st.session_state.chat_history[-2]["content"],
            "response": msg["content"],
            "feedback": feedback,
            "reward": reward_model(feedback)
        })

# Input
query = st.chat_input("Ask me something...")

if query:
    with st.chat_message("user", avatar=user_avatar):
        st.markdown(query)

    response = retrieve_and_answer(query)

    with st.chat_message("assistant", avatar=bot_avatar):
        st.markdown(response)

    # Immediate feedback widget after assistant reply
    feedback = st.radio("Was this response helpful?", ["👍 Yes", "👎 No"], horizontal=True, key=f"feedback_{len(st.session_state.chat_history)}")
    reward = 1 if feedback == "👍 Yes" else -1
    st.session_state.feedback_log.append({
        "query": query,
        "response": response,
        "feedback": feedback,
        "reward": reward
    })

# Save feedback manually
if st.button("💾 Save All Feedback"):
    if st.session_state.feedback_log:
        save_feedback_log(st.session_state.feedback_log)
        st.success("Feedback saved!")
        st.session_state.feedback_log = []


















# import streamlit as st
# import os
# import faiss
# import numpy as np
# import pickle
# from openai import OpenAI
# from anthropic import Anthropic
# from sentence_transformers import SentenceTransformer
# from pydantic import BaseModel
# from PIL import Image
# 
# 
# 
# 
# 
# 
# # Set API Keys
# ANTHROPIC_API_KEY="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"
# OPENAI_API_KEY="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"
# 
# # Set page config
# st.set_page_config(page_title="MindBuddy RAG Chat", layout="centered")
# 
# # Load Avatars
# user_avatar = Image.open("Hilka.JPG")  # Optional user image
# bot_avatar = Image.open("Hilka.JPG")        # Penguin mascot or similar
# 
# # Title and description
# st.image(bot_avatar, width=100)
# st.title("MindBuddy 🐧")
# st.markdown("**Ask me anything about CBT. I'm here to help.**")
# 
# # Folder containing text files
# TEXT_FOLDER = "Resources\\Cognitive Therapy Behavior"  # Change to your folder path
# INDEX_FILE = f"faiss_index_Cognitive_Therapy_Behavior.bin"
# METADATA_FILE = f"metadata_Cognitive_Therapy_Behavior.pkl"
# 
# # Load text data
# all_texts = []
# file_names = []
# for filename in os.listdir(TEXT_FOLDER):
#     if filename.endswith(".txt"):
#         with open(os.path.join(TEXT_FOLDER, filename), "r", encoding="utf-8", errors="ignore") as file:
#             all_texts.append(file.read())
#             file_names.append(filename)
# 
# # Validate input data using Pydantic
# class Document(BaseModel):
#     content: str
#     filename: str
# 
# documents = [Document(content=text, filename=name) for text, name in zip(all_texts, file_names)]
# 
# # Save/load FAISS index and metadata
# def save_faiss_index(index, file_path):
#     faiss.write_index(index, file_path)
# 
# def load_faiss_index(file_path):
#     return faiss.read_index(file_path)
# 
# def save_metadata(metadata, file_path):
#     with open(file_path, "wb") as f:
#         pickle.dump(metadata, f)
# 
# def load_metadata(file_path):
#     with open(file_path, "rb") as f:
#         return pickle.load(f)
# 
# # Initialize Sentence Transformer
# model = SentenceTransformer("all-MiniLM-L6-v2")
# 
# # Load or create FAISS index
# if os.path.exists(INDEX_FILE) and os.path.exists(METADATA_FILE):
#     st.write("Loading existing FAISS index...")
#     index = load_faiss_index(INDEX_FILE)
#     documents = load_metadata(METADATA_FILE)
# else:
#     st.write("Creating FAISS index from scratch...")
#     embeddings = model.encode([doc.content for doc in documents])
#     dimension = embeddings.shape[1]
#     index = faiss.IndexFlatL2(dimension)
#     index.add(np.array(embeddings, dtype=np.float32))
#     save_faiss_index(index, INDEX_FILE)
#     save_metadata(documents, METADATA_FILE)
#     st.write("FAISS index saved for future use.")
# 
# # Initialize Anthropic client
# client = Anthropic(api_key=ANTHROPIC_API_KEY)
# 
# # Chat history
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []
# 
# # RAG function
# def retrieve_and_answer(query):
#     query_embedding = model.encode([query])
#     D, I = index.search(np.array(query_embedding, dtype=np.float32), k=3)
#     retrieved_texts = [documents[i].content for i in I[0] if i < len(documents)]
#     combined_text = "\n\n".join(retrieved_texts)
# 
#     messages = st.session_state.chat_history + [
#         {"role": "user", "content": f"The question is in persian translate to english first, and then use the following retrieved texts to answer the question: {query}\n\nRetrieved Texts:\n{combined_text}"}
#     ]
# 
#     response = client.messages.create(
#         model="claude-2",
#         max_tokens=200,
#         system="You are an AI assistant that provides accurate responses based on retrieved documents.",
#         messages=messages
#     )
# 
#     reply = response.content[0].text.replace('\n', '').strip()
# 
#     # Update chat history
#     st.session_state.chat_history.append({"role": "user", "content": query})
#     st.session_state.chat_history.append({"role": "assistant", "content": reply})
# 
#     return reply
# 
# # Display chat history
# for msg in st.session_state.chat_history:
#     with st.chat_message("user" if msg["role"] == "user" else "assistant", avatar=user_avatar if msg["role"] == "user" else bot_avatar):
#         st.markdown(msg["content"])
# 
# # Chat input
# query = st.chat_input("Ask me something...")
# if query:
#     with st.chat_message("user", avatar=user_avatar):
#         st.markdown(query)
# 
#     response = retrieve_and_answer(query)
# 
#     with st.chat_message("assistant", avatar=bot_avatar):
#         st.markdown(response)

