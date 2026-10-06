"""
RAG Chatbot - upload a PDF/TXT and chat with it.
Run:  streamlit run app.py
"""
import os
import tempfile
import uuid
import warnings

import streamlit as st
import torch
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------------
st.set_page_config(page_title="RAG Document Chatbot", page_icon="📄", layout="wide")
st.title("📄 RAG Document Chatbot")
st.caption("Upload a document, then ask questions. Answers come only from your document.")

# ------------------------------------------------------------------
# SIDEBAR: all the tunable factors (model, temperature, chunking ...)
# ------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    st.subheader("LLM")
    model_id = st.selectbox(
        "Model",
        [
            "Qwen/Qwen2.5-0.5B-Instruct",   # fastest, runs fine on CPU
            "Qwen/Qwen2.5-1.5B-Instruct",   # sir's original model
            "Qwen/Qwen2.5-3B-Instruct",     # better answers, needs GPU / more RAM
            "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        ],
        index=0,
    )
    temperature = st.slider("Temperature", 0.0, 1.5, 0.3, 0.05,
                            help="0 = deterministic/factual, higher = more creative")
    top_p = st.slider("Top-p", 0.1, 1.0, 0.9, 0.05)
    max_new_tokens = st.slider("Max new tokens", 50, 800, 300, 50)
    repetition_penalty = st.slider("Repetition penalty", 1.0, 1.5, 1.1, 0.05)

    st.subheader("Chunking")
    chunk_size = st.slider("Chunk size", 200, 2000, 600, 50)
    chunk_overlap = st.slider("Chunk overlap", 0, 500, 100, 25)

    st.subheader("Retrieval")
    embedding_model = st.selectbox(
        "Embedding model",
        ["sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5"],
    )
    top_k = st.slider("Chunks retrieved (k)", 1, 8, 4)

    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ------------------------------------------------------------------
# CACHED LOADERS (so models are loaded only once)
# ------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embeddings(name: str):
    return HuggingFaceEmbeddings(model_name=name)


@st.cache_resource(show_spinner="Loading LLM (first time downloads the model)...")
def load_llm(name: str):
    tokenizer = AutoTokenizer.from_pretrained(name)
    use_gpu = torch.cuda.is_available()
    model = AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=torch.bfloat16 if use_gpu else torch.float32,
        device_map="auto" if use_gpu else None,
    )
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return pipe, tokenizer


def build_vector_store(file_bytes: bytes, filename: str, c_size: int, c_overlap: int, emb):
    """Load -> split -> embed -> store in an in-memory Chroma collection."""
    suffix = os.path.splitext(filename)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        loader = PyPDFLoader(tmp_path) if suffix == ".pdf" else TextLoader(tmp_path, encoding="utf-8")
        docs = loader.load()
    finally:
        os.remove(tmp_path)

    splitter = RecursiveCharacterTextSplitter(chunk_size=c_size, chunk_overlap=c_overlap)
    splits = splitter.split_documents(docs)
    store = Chroma.from_documents(
        documents=splits,
        embedding=emb,
        collection_name=f"doc_{uuid.uuid4().hex[:8]}",  # fresh collection per upload
    )
    return store, len(docs), len(splits)


def answer_question(question, store, pipe, tokenizer):
    retrieved = store.similarity_search(question, k=top_k)
    context = "\n\n".join(" ".join(d.page_content.split()) for d in retrieved)

    messages = [
        {"role": "system", "content": (
            "You are a helpful assistant. Answer the question strictly using the provided context. "
            "If the context does not contain the answer, say \"I don't know based on the document.\" "
            "Keep the answer clear and concise.")},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]
    # apply_chat_template adds the right special tokens for whichever model is chosen
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty,
        return_full_text=False,
    )
    if temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
    else:
        gen_kwargs.update(do_sample=False)

    out = pipe(prompt, **gen_kwargs)
    return out[0]["generated_text"].strip(), retrieved


# ------------------------------------------------------------------
# FILE UPLOAD
# ------------------------------------------------------------------
uploaded = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

if uploaded is None:
    st.info("👆 Upload a document to get started. (Try the included `sample_document.pdf`.)")
    st.stop()

# Rebuild the index only when the file or chunk/embedding settings change
index_key = (uploaded.name, uploaded.size, chunk_size, chunk_overlap, embedding_model)
if st.session_state.get("index_key") != index_key:
    with st.spinner("Reading and indexing your document..."):
        emb = load_embeddings(embedding_model)
        store, n_pages, n_chunks = build_vector_store(
            uploaded.getvalue(), uploaded.name, chunk_size, chunk_overlap, emb
        )
    st.session_state.update(index_key=index_key, store=store, n_pages=n_pages,
                            n_chunks=n_chunks, messages=[])

st.success(f"✅ **{uploaded.name}** indexed — {st.session_state.n_pages} page(s), "
           f"{st.session_state.n_chunks} chunks.")

pipe, tokenizer = load_llm(model_id)

# ------------------------------------------------------------------
# CHAT UI
# ------------------------------------------------------------------
st.session_state.setdefault("messages", [])

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander("📚 Sources"):
                for i, s in enumerate(m["sources"], 1):
                    st.markdown(f"**Chunk {i}** (page {s['page']}): {s['text']}")

if question := st.chat_input("Ask something about your document..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, docs = answer_question(question, st.session_state.store, pipe, tokenizer)
        st.markdown(answer)
        sources = [{"page": d.metadata.get("page", 0) + 1 if "page" in d.metadata else "-",
                    "text": " ".join(d.page_content.split())[:300] + "..."} for d in docs]
        with st.expander("📚 Sources"):
            for i, s in enumerate(sources, 1):
                st.markdown(f"**Chunk {i}** (page {s['page']}): {s['text']}")

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
