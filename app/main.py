from fastapi import FastAPI
from pydantic import BaseModel
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from sentence_transformers import CrossEncoder

# === Inisialisasi FastAPI ===
app = FastAPI(title="Yupien AI - TEST Gold Assistant")

# === Load Embedding Model ===
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# === Load Vector Database ===
vector_db = Chroma(
    persist_directory="db_faq_baru_4",
    embedding_function=embedding_model
)
retriever = vector_db.as_retriever(search_kwargs={"k": 10})

# === Load CrossEncoder untuk Reranker ===
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device='cpu')

# === Load Chat Model dari Ollama ===
llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0.0,
)

# === Prompt Template ===
chat_prompt = ChatPromptTemplate.from_messages([
("system", 
     "Kamu adalah Yupien AI, asisten akademik. Tugasmu adalah menjawab pertanyaan berdasarkan potongan dokumen yang diberikan."
     "\n\nINSTRUKSI:"
     "\n1. Jawab hanya berdasarkan teks yang ada di dalam tag <context> dan </context> di bawah."
     "\n2. Jangan gunakan pengetahuan luar."
     "\n3. Jika jawaban tidak ditemukan di dalam <context>, katakan 'Maaf, saya tidak menemukan informasi tersebut dalam data kami.'."
     "\n4. Jawablah dengan singkat dan langsung pada intinya."),
    ("human", 
     "<context>"
     "\n{context}"
     "\n</context>"
     "\n\nBerdasarkan context di atas, jawablah pertanyaan ini:"
     "\nPertanyaan: {question}"
     "\nJawaban:")
])

# === Build LLM Chain (tanpa deprecated LLMChain) ===
chain: RunnableSequence = chat_prompt | llm

# === Reranking Function ===
def rerank_documents(question, documents, top_k=1): 
    if not documents:
        return []
    
    pairs = []
    for doc in documents:
        # Prioritaskan metadata question jika ada, kalau tidak pakai content
        doc_text = doc.metadata.get("question", doc.page_content)
        pairs.append([question, doc_text])

    scores = reranker.predict(pairs)
    
    # Urutkan berdasarkan skor tertinggi
    reranked = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
    
    # Kembalikan dokumen terbaik
    return [doc for _, doc in reranked[:top_k]]


# === Skema Request ===
class QuestionRequest(BaseModel):
    question: str

# === Endpoint untuk Bertanya ===
@app.post("/ask")
def ask_roxy(request: QuestionRequest):
    question = request.question

    # Khusus pertanyaan "jumlah data"
    if "jumlah data" in question.lower():
        total_docs = vector_db._collection.count()
        return {
            "pertanyaan": question,
            "jawaban": f"Jumlah data RAG saya saat ini adalah {total_docs} dokumen."
        }

    # Ambil dokumen
    retrieved_docs = retriever.invoke(question)
    if not retrieved_docs or all(len(doc.page_content.strip()) == 0 for doc in retrieved_docs):
        return {
            "pertanyaan": question,
            "jawaban": "Maaf, saya belum menemukan jawaban untuk pertanyaan tersebut."
        }

    # Rerank dokumen
    top_docs = rerank_documents(question, retrieved_docs, top_k=1)
    doc_texts = [f"Dokumen {i+1}: {doc.page_content}" for i, doc in enumerate(top_docs)]
    context = "\n\n".join(doc_texts)

    # Generate Jawaban dengan LLM
    ai_response = chain.invoke({"question": question, "context": context})


    # Jalankan pipeline prompt | llm
    answer = chain.invoke({"question": question, "context": context})

    return {
        "pertanyaan": question,
        "jawaban": ai_response.content,
        "dokumen_terkait": doc_texts
    }

# === Root Endpoint ===
@app.get("/")
def root():
    return {"message": "Yupien AI API with ChatOllama is running"}
