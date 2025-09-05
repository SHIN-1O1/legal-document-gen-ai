import os
import sys
import json
import time
import hashlib
import logging
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---- Third-party deps ----
from PIL import Image, ImageEnhance
import pytesseract
from PyPDF2 import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.embeddings import HuggingFaceEmbeddings

import pickle

# ----------------- CONFIG -----------------
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"D:\ayooooooo\gemini-api-key.json"
VERTEX_VECTORSTORE_PATH = "vertex_vectorstore.faiss"
LEGAL_VECTORSTORE_PATH = "legal_vectorstore.faiss"

# ----------------- PDF Extraction -----------------
def extract_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text.strip()
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

# ----------------- Build Dual Vectorstores -----------------
def build_dual_vectorstores(document_text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.create_documents([document_text])

    # Vertex AI embeddings
    vertex_embeddings = VertexAIEmbeddings(model_name="text-embedding-005")
    vertex_store = FAISS.from_documents(docs, vertex_embeddings)
    vertex_store.save_local(VERTEX_VECTORSTORE_PATH)

    # Legal-BERT embeddings
    legal_embeddings = HuggingFaceEmbeddings(model_name="nlpaueb/legal-bert-base-uncased")
    legal_store = FAISS.from_documents(docs, legal_embeddings)
    legal_store.save_local(LEGAL_VECTORSTORE_PATH)

    return vertex_store, legal_store

def load_dual_vectorstores():
    vertex_embeddings = VertexAIEmbeddings(model_name="text-embedding-005")
    legal_embeddings = HuggingFaceEmbeddings(model_name="nlpaueb/legal-bert-base-uncased")

    vertex_store = None
    legal_store = None

    if os.path.exists(VERTEX_VECTORSTORE_PATH):
        vertex_store = FAISS.load_local(VERTEX_VECTORSTORE_PATH, vertex_embeddings, allow_dangerous_deserialization=True)

    if os.path.exists(LEGAL_VECTORSTORE_PATH):
        legal_store = FAISS.load_local(LEGAL_VECTORSTORE_PATH, legal_embeddings, allow_dangerous_deserialization=True)

    return vertex_store, legal_store

# ----------------- Hybrid Retriever -----------------
def hybrid_retrieve(vertex_store, legal_store, query, top_k=3):
    results = []

    if vertex_store:
        v_embed = VertexAIEmbeddings(model_name="text-embedding-005").embed_query(query)
        v_dist, v_idx = vertex_store.index.search(np.array(v_embed, dtype="float32").reshape(1, -1), top_k)
        for idx in v_idx[0]:
            doc_id = vertex_store.index_to_docstore_id(idx)
            doc = vertex_store.docstore.search(doc_id)[0]
            results.append(doc)

    if legal_store:
        l_embed = HuggingFaceEmbeddings(model_name="nlpaueb/legal-bert-base-uncased").embed_query(query)
        l_dist, l_idx = legal_store.index.search(np.array(l_embed, dtype="float32").reshape(1, -1), top_k)
        for idx in l_idx[0]:
            doc_id = legal_store.index_to_docstore_id(idx)
            doc = legal_store.docstore.search(doc_id)[0]
            results.append(doc)

    # Deduplicate by content
    seen, unique_docs = set(), []
    for d in results:
        if d.page_content not in seen:
            unique_docs.append(d)
            seen.add(d.page_content)

    return unique_docs[:top_k]

# ----------------- Ask Document -----------------
def ask_document(vertex_store, legal_store, user_question, language="English"):
    llm = ChatVertexAI(model="gemini-2.5-flash-lite", temperature=0)
    relevant_docs = hybrid_retrieve(vertex_store, legal_store, user_question)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    prompt = PromptTemplate(
        input_variables=["context", "question", "language"],
        template="""
        You are a legal assistant. Use the following context to answer the question.
        Respond in {language}.

        Context:
        {context}

        Question: {question}

        Answer clearly and concisely.
        """
    )
    chain = LLMChain(llm=llm, prompt=prompt)

    try:
        response = chain.invoke({
            "context": context,
            "question": user_question,
            "language": language
        })
        return response.get("text", "").strip()
    except Exception as e:
        return f"Error generating response: {str(e)}"

# ----------------- GUI -----------------
class PDFChatGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Legal Assistant — Lite")
        self.root.geometry("1000x700")

        self.extractor = Extractor(Path(CONFIG.CACHE_DIR))
        self.vs = VectorStore(CONFIG.EMBEDDING_BACKEND, Path(CONFIG.DATA_DIR))
        self.qa = QAEngine()

        self.current_doc_text: Optional[str] = None
        self.current_doc_fid: Optional[str] = None
        self.busy = False
        self.lang = "en"

        top = tk.Frame(root)
        top.pack(fill="x", padx=8, pady=8)

        tk.Button(top, text="Load PDF", command=self.load_pdf).pack(side="left")
        tk.Button(top, text="Load Image", command=self.load_image).pack(side="left", padx=6)
        tk.Button(top, text="Clear", command=self.clear_chat).pack(side="left", padx=6)

        self.lang_var = tk.StringVar(value="English")
        lang_menu = tk.OptionMenu(top, self.lang_var, *LANGUAGES.keys(), command=self.on_language_change)
        lang_menu.pack(side="left", padx=10)

        self.status = tk.StringVar(value="Ready")
        tk.Label(top, textvariable=self.status, anchor="w").pack(side="left", padx=20)

        self.chat_box = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=90, height=20)
        self.chat_box.pack(pady=10)
        self.chat_box.config(state=tk.NORMAL)
        if self.vertex_store and self.legal_store:
            self.chat_box.insert(tk.END, "✅ Loaded saved vectorstores (Vertex + Legal-BERT).\n")
        else:
            self.chat_box.insert(tk.END, "⚠️ No saved vectorstores found. Please load a PDF.\n")
        self.chat_box.config(state=tk.DISABLED)

        self.user_input = tk.Entry(root, width=90)
        self.user_input.pack(pady=5)
        self.user_input.bind("<Return>", self.ask_question)

        tk.Button(root, text="Ask", command=self.ask_question).pack(pady=5)

    def clear_chat(self):
        self.chat.configure(state="normal")
        self.chat.delete("1.0", "end")
        self.chat.configure(state="disabled")
        self.chat_insert_system("Cleared. Load a document to begin.")

    def load_pdf(self):
        if self.busy:
            return
        path = filedialog.askopenfilename(title="Select PDF", filetypes=[("PDF", "*.pdf"), ("All", "*.*")])
        if not path:
            return
        threading.Thread(target=self._load_pdf_worker, args=(path,), daemon=True).start()

    def _load_pdf_worker(self, path: str):
        self.set_busy(True, "Extracting PDF text...")
        text = self.extractor.from_pdf(path)
        self._after_load(path, text)

    def load_image(self):
        if self.busy:
            return
        path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff;*.gif"), ("All", "*.*")],
        )
        if not path:
            return

        self.chat_box.config(state=tk.NORMAL)
        self.chat_box.insert(tk.END, "⏳ Processing PDF and building vectorstores...\n")
        self.chat_box.config(state=tk.DISABLED)

        self.vertex_store, self.legal_store = build_dual_vectorstores(self.pdf_text)

        self.chat_box.config(state=tk.NORMAL)
        self.chat_box.insert(tk.END, "✅ PDF processed and vectorstores saved!\n")
        self.chat_box.config(state=tk.DISABLED)

    def ask_question(self, event=None):
        question = self.user_input.get()
        if not question.strip():
            return
        if not self.vs.store:
            messagebox.showwarning("No document", "Load a PDF or image first.")
            return

        language = self.language_var.get()
        answer = ask_document(self.vertex_store, self.legal_store, question, language)

        self.chat_box.config(state=tk.NORMAL)
        self.chat_box.insert(tk.END, f"You ({language}): {question}\n")
        self.chat_box.insert(tk.END, f"Assistant: {answer}\n\n")
        self.chat_box.config(state=tk.DISABLED)

        self.user_input.delete(0, tk.END)
        self.chat_box.yview(tk.END)

# ----------------- Main -----------------
if __name__ == "__main__":
    main()
