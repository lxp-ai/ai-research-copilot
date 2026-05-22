import os
import json
import shutil
import numpy as np

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

load_dotenv()

api_key = os.getenv("DEEPSEEK_API_KEY")

if not api_key:
    raise ValueError("DEEPSEEK_API_KEY 没有读取到，请检查 backend/.env 文件")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com"
)

# 第一次运行会自动下载模型，可能需要等一会儿
embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

UPLOAD_DIR = "uploads"
DOCUMENTS_FILE = "documents.json"
CHAT_HISTORY_FILE = "chat_history.json"

class ChatRequest(BaseModel):
    message: str


class DocumentQuestionRequest(BaseModel):
    filename: str
    question: str

class DocumentSummaryRequest(BaseModel):
    filename: str

class ChatHistoryItemRequest(BaseModel):
    mode: str
    filename: str | None = None
    question: str
    answer: str
    chunks: list = []

def load_documents():
    if not os.path.exists(DOCUMENTS_FILE):
        return {}

    with open(DOCUMENTS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_documents(documents):
    with open(DOCUMENTS_FILE, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)

def load_chat_history():
    if not os.path.exists(CHAT_HISTORY_FILE):
        return []

    with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_chat_history(history):
    with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def split_text_into_chunks(text: str, chunk_size: int = 800, overlap: int = 150):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]

        chunks.append({
            "chunk_id": len(chunks),
            "text": chunk_text
        })

        start += chunk_size - overlap

    return chunks


def create_embeddings_for_chunks(chunks: list):
    """
    给每个 chunk 生成 embedding 向量。
    注意：json 不能直接保存 numpy array，所以要转成 list。
    """
    texts = [chunk["text"] for chunk in chunks]

    embeddings = embedding_model.encode(
        texts,
        normalize_embeddings=True
    )

    for index, chunk in enumerate(chunks):
        chunk["embedding"] = embeddings[index].tolist()

    return chunks


def semantic_retrieve(question: str, chunks: list, top_k: int = 5):
    """
    语义检索：
    1. 把用户问题转成向量
    2. 和每个 chunk 的向量做相似度计算
    3. 返回最相似的 top_k 个 chunk
    """
    question_embedding = embedding_model.encode(
        question,
        normalize_embeddings=True
    )

    scored_chunks = []

    for chunk in chunks:
        if "embedding" not in chunk:
            continue

        chunk_embedding = np.array(chunk["embedding"])

        score = float(np.dot(question_embedding, chunk_embedding))

        scored_chunks.append({
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "score": score
        })

    scored_chunks.sort(key=lambda x: x["score"], reverse=True)

    return scored_chunks[:top_k]


@app.get("/")
def health_check():
    return {
        "status": "AI Research Copilot backend is running"
    }


@app.post("/chat")
def chat(request: ChatRequest):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                "role": "system",
                "content": "你是一个专业的 AI Research Copilot，擅长解释论文、报告和技术文档。回答要清晰、结构化、适合初学者理解。"
            },
            {
                "role": "user",
                "content": request.message
            }
        ]
    )

    return {
        "reply": response.choices[0].message.content
    }


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        return {
            "error": "目前只支持上传 PDF 文件"
        }

    os.makedirs(UPLOAD_DIR, exist_ok=True)

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    reader = PdfReader(file_path)

    all_text = ""

    for page_index, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            all_text += f"\n\n--- Page {page_index + 1} ---\n"
            all_text += page_text

    chunks = split_text_into_chunks(all_text)

    chunks = create_embeddings_for_chunks(chunks)

    documents = load_documents()

    documents[file.filename] = {
        "filename": file.filename,
        "pages": len(reader.pages),
        "text": all_text,
        "chunks": chunks
    }

    save_documents(documents)

    return {
        "filename": file.filename,
        "pages": len(reader.pages),
        "text_length": len(all_text),
        "chunks_count": len(chunks),
        "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
        "text_preview": all_text[:500],
        "message": "PDF 上传、解析、切块、向量化成功"
    }


@app.get("/documents")
def list_documents():
    documents = load_documents()

    result = []

    for filename, doc in documents.items():
        chunks = doc.get("chunks", [])

        result.append({
            "filename": filename,
            "pages": doc["pages"],
            "text_length": len(doc["text"]),
            "chunks_count": len(chunks),
            "has_embeddings": bool(chunks and "embedding" in chunks[0])
        })

    return {
        "documents": result
    }


@app.post("/ask-document")
def ask_document(request: DocumentQuestionRequest):
    documents = load_documents()

    if request.filename not in documents:
        return {
            "error": "没有找到这个文档，请先上传 PDF"
        }

    document = documents[request.filename]

    chunks = document.get("chunks", [])

    if not chunks:
        chunks = split_text_into_chunks(document["text"])

    if chunks and "embedding" not in chunks[0]:
        chunks = create_embeddings_for_chunks(chunks)
        document["chunks"] = chunks
        documents[request.filename] = document
        save_documents(documents)

    retrieved_chunks = semantic_retrieve(
        question=request.question,
        chunks=chunks,
        top_k=5
    )

    context = ""

    for item in retrieved_chunks:
        context += f"\n\n[Chunk {item['chunk_id']} | Similarity {item['score']:.4f}]\n"
        context += item["text"]

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                "role": "system",
                "content": """
你是一个专业的 AI Research Copilot。

你必须只基于给定的文档片段回答问题。
如果文档片段里没有相关信息，你要明确说：
“当前检索到的文档片段中没有找到相关信息”。

不要编造。
回答要清晰、结构化，适合初学者理解。
回答最后必须列出你参考了哪些 Chunk。
"""
            },
            {
                "role": "user",
                "content": f"""
以下是系统从文档中语义检索到的相关片段：

{context}

用户问题：
{request.question}
"""
            }
        ]
    )

    return {
        "filename": request.filename,
        "question": request.question,
        "retrieval_method": "semantic_embedding_search",
        "retrieved_chunks": [
            {
                "chunk_id": item["chunk_id"],
                "similarity": round(item["score"], 4),
                "preview": item["text"][:200]
            }
            for item in retrieved_chunks
        ],
        "answer": response.choices[0].message.content
    }





@app.post("/summarize-document")
def summarize_document(request: DocumentSummaryRequest):
    documents = load_documents()

    if request.filename not in documents:
        return {
            "error": "没有找到这个文档，请先上传 PDF"
        }

    document = documents[request.filename]
    document_text = document["text"]

    # 先取前 12000 字做摘要，后面可以升级成 map-reduce summary
    context = document_text[:12000]

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                "role": "system",
                "content": """
你是一个专业的 AI Research Copilot。
请基于用户提供的文档内容，生成结构化研究摘要。
不要编造文档中没有的信息。
回答使用中文。
"""
            },
            {
                "role": "user",
                "content": f"""
请基于以下文档内容生成摘要：

{context}

请按下面结构输出：

1. 文档主题
2. 核心观点
3. 关键结论
4. 重要细节
5. 一句话总结
"""
            }
        ]
    )

    return {
        "filename": request.filename,
        "summary": response.choices[0].message.content
    }

@app.post("/ask-all-documents")
def ask_all_documents(request: ChatRequest):
    documents = load_documents()

    if not documents:
        return {
            "error": "当前没有任何文档，请先上传 PDF"
        }

    all_scored_chunks = []

    question_embedding = embedding_model.encode(
        request.message,
        normalize_embeddings=True
    )

    for filename, document in documents.items():
        chunks = document.get("chunks", [])

        if not chunks:
            continue

        if chunks and "embedding" not in chunks[0]:
            chunks = create_embeddings_for_chunks(chunks)
            document["chunks"] = chunks
            documents[filename] = document

        for chunk in chunks:
            if "embedding" not in chunk:
                continue

            chunk_embedding = np.array(chunk["embedding"])
            score = float(np.dot(question_embedding, chunk_embedding))

            all_scored_chunks.append({
                "filename": filename,
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "score": score
            })

    save_documents(documents)

    all_scored_chunks.sort(key=lambda x: x["score"], reverse=True)

    retrieved_chunks = all_scored_chunks[:5]

    if not retrieved_chunks:
        return {
            "error": "没有检索到相关文档片段"
        }

    context = ""

    for item in retrieved_chunks:
        context += f"\n\n[File: {item['filename']} | Chunk {item['chunk_id']} | Similarity {item['score']:.4f}]\n"
        context += item["text"]

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                "role": "system",
                "content": """
你是一个专业的 AI Research Copilot。

你现在需要基于多个文档中检索到的相关片段回答问题。
你必须只基于给定片段回答。
如果片段中没有相关信息，请明确说明“当前检索到的文档片段中没有找到相关信息”。
不要编造。
回答要清晰、结构化。
回答最后必须列出参考的文件名和 Chunk。
"""
            },
            {
                "role": "user",
                "content": f"""
以下是从多个文档中检索到的相关片段：

{context}

用户问题：
{request.message}
"""
            }
        ]
    )

    return {
        "question": request.message,
        "retrieval_method": "multi_document_semantic_search",
        "retrieved_chunks": [
            {
                "filename": item["filename"],
                "chunk_id": item["chunk_id"],
                "similarity": round(item["score"], 4),
                "preview": item["text"][:200]
            }
            for item in retrieved_chunks
        ],
        "answer": response.choices[0].message.content
    }

@app.delete("/documents/{filename}")
def delete_document(filename: str):
    documents = load_documents()

    if filename not in documents:
        return {
            "error": "没有找到这个文档"
        }

    # 1. 从 documents.json 中删除记录
    del documents[filename]
    save_documents(documents)

    # 2. 删除 uploads 里的 PDF 文件
    file_path = os.path.join(UPLOAD_DIR, filename)

    if os.path.exists(file_path):
        os.remove(file_path)

    return {
        "message": "文档删除成功",
        "filename": filename
    }


@app.post("/summarize-all-documents")
def summarize_all_documents():
    documents = load_documents()

    if not documents:
        return {
            "error": "当前没有任何文档，请先上传 PDF"
        }

    combined_context = ""

    for filename, document in documents.items():
        text = document.get("text", "")

        # 每个文档最多取前 6000 字，避免 prompt 过长
        preview_text = text[:6000]

        combined_context += f"""

==============================
文档名称：{filename}
==============================

{preview_text}
"""

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                "role": "system",
                "content": """
你是一个专业的 AI Research Copilot。
你需要基于多个文档内容，生成跨文档综述和对比分析。
你必须只基于用户提供的文档内容回答，不要编造。
回答使用中文，结构清晰，适合用于研究汇报或面试项目展示。
"""
            },
            {
                "role": "user",
                "content": f"""
以下是多个文档的内容节选：

{combined_context}

请基于这些文档，按下面结构输出：

1. 所有文档共同关注的主题
2. 每个文档的主要内容
3. 文档之间的相同点
4. 文档之间的差异点
5. 综合结论
6. 适合汇报的一句话总结
"""
            }
        ]
    )

    return {
        "document_count": len(documents),
        "summary": response.choices[0].message.content
    }


@app.get("/chat-history")
def get_chat_history():
    history = load_chat_history()

    return {
        "history": history
    }


@app.post("/chat-history")
def add_chat_history(item: ChatHistoryItemRequest):
    history = load_chat_history()

    history_item = {
        "mode": item.mode,
        "filename": item.filename,
        "question": item.question,
        "answer": item.answer,
        "chunks": item.chunks
    }

    # 最新记录放前面
    history.insert(0, history_item)

    # 最多保存 50 条，避免文件越来越大
    history = history[:50]

    save_chat_history(history)

    return {
        "message": "聊天历史保存成功",
        "history_count": len(history)
    }


@app.delete("/chat-history")
def clear_chat_history():
    save_chat_history([])

    return {
        "message": "聊天历史已清空"
    }