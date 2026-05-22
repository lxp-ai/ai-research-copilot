import { useEffect, useState } from "react";
import axios from "axios";
import "./App.css";
import ReactMarkdown from "react-markdown";

const API_BASE_URL = "http://127.0.0.1:8000";

type DocumentItem = {
  filename: string;
  pages: number;
  text_length: number;
  chunks_count: number;
  has_embeddings: boolean;
};

type RetrievedChunk = {
  filename?: string;
  chunk_id: number;
  similarity: number;
  preview: string;
};

type ChatHistoryItem = {
  question: string;
  answer: string;
  chunks: RetrievedChunk[];
};

function App() {
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [selectedFile, setSelectedFile] = useState("");
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [chunks, setChunks] = useState<RetrievedChunk[]>([]);
  const [summary, setSummary] = useState("");
  const [chatHistory, setChatHistory] = useState<ChatHistoryItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [uploadMessage, setUploadMessage] = useState("");
  const [askMode, setAskMode] = useState<"single" | "all">("single");

  async function fetchDocuments() {
    try {
      const res = await axios.get(`${API_BASE_URL}/documents`);
      setDocuments(res.data.documents || []);
    } catch (error) {
      console.error(error);
    }
  }

  async function handleUpload() {
    if (!uploadFile) {
      alert("请先选择一个 PDF 文件");
      return;
    }

    const formData = new FormData();
    formData.append("file", uploadFile);

    setLoading(true);
    setUploadMessage("正在上传、解析、切块、向量化，请稍等...");

    try {
      const res = await axios.post(`${API_BASE_URL}/upload`, formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });

      if (res.data.error) {
        setUploadMessage(res.data.error);
        return;
      }

      setUploadMessage(
        `${res.data.message}：${res.data.filename}，共 ${res.data.chunks_count} 个 chunks`
      );

      await fetchDocuments();
      setSelectedFile(res.data.filename);
    } catch (error) {
      console.error(error);
      setUploadMessage("上传失败，请查看后端终端报错");
    } finally {
      setLoading(false);
    }
  }

  async function handleAsk() {

    if (!question.trim()) {
      alert("请输入问题");
      return;
    }

    setLoading(true);
    setAnswer("");
    setChunks([]);

    try {
      let res;

      if (askMode === "single") {
        if (!selectedFile) {
          alert("请先选择一个文档");
          return;
        }

        res = await axios.post(`${API_BASE_URL}/ask-document`, {
          filename: selectedFile,
          question,
        });
      } else {
        res = await axios.post(`${API_BASE_URL}/ask-all-documents`, {
          message: question,
        });
      }

      if (res.data.error) {
        setAnswer(res.data.error);
        return;
      }

      const newItem: ChatHistoryItem = {
        question,
        answer: res.data.answer,
        chunks: res.data.retrieved_chunks || [],
      };

      setChatHistory((prev) => [newItem, ...prev]);

      setAnswer(res.data.answer);
      setChunks(res.data.retrieved_chunks || []);
      setQuestion("");
    } catch (error) {
      console.error(error);
      setAnswer("提问失败，请查看后端终端报错");
    } finally {
      setLoading(false);
    }
  }

  async function handleSummary() {
  if (!selectedFile) {
    alert("请先选择一个文档");
    return;
  }

  setLoading(true);
  setSummary("");

  try {
    const res = await axios.post(`${API_BASE_URL}/summarize-document`, {
      filename: selectedFile,
    });

    if (res.data.error) {
      setSummary(res.data.error);
      return;
    }

    setSummary(res.data.summary);
  } catch (error) {
    console.error(error);
    setSummary("生成摘要失败，请查看后端终端报错");
  } finally {
    setLoading(false);
  }
}

  useEffect(() => {
    fetchDocuments();
  }, []);

  return (
    <div className="app">
      <header className="header">
        <div>
          <h1>AI Research Copilot</h1>
          <p>上传 PDF，基于语义检索 RAG 进行文档问答</p>
        </div>
        <span className="badge">RAG Demo</span>
      </header>

      <main className="layout">
        <section className="panel">
          <h2>1. 上传文档</h2>

          <input
            type="file"
            accept="application/pdf"
            onChange={(e) => setUploadFile(e.target.files?.[0] || null)}
          />

          <button onClick={handleUpload} disabled={loading}>
            {loading ? "处理中..." : "上传 PDF"}
          </button>

          {uploadMessage && <p className="message">{uploadMessage}</p>}

          <h2>2. 文档列表</h2>

          {documents.length === 0 ? (
            <p className="empty">暂无文档，请先上传 PDF。</p>
          ) : (
            <div className="document-list">
              {documents.map((doc) => (
                <button
                  key={doc.filename}
                  className={
                    selectedFile === doc.filename
                      ? "document-card active"
                      : "document-card"
                  }
                  onClick={() => setSelectedFile(doc.filename)}
                >
                  <strong>{doc.filename}</strong>
                  <span>{doc.pages} 页</span>
                  <span>{doc.chunks_count} chunks</span>
                  <span>{doc.has_embeddings ? "已向量化" : "未向量化"}</span>
                </button>
              ))}
            </div>
          )}
        </section>

        <section className="panel main-panel">
          <h2>3. 向文档提问</h2>

          <div className="selected">
            当前文档：
            <strong>{selectedFile || "未选择"}</strong>
          </div>
          
          <div className="mode-switch">
            <button
              className={askMode === "single" ? "mode-button active" : "mode-button"}
              onClick={() => setAskMode("single")}
            >
              单文档问答
            </button>

            <button
              className={askMode === "all" ? "mode-button active" : "mode-button"}
              onClick={() => setAskMode("all")}
            >
              全部文档问答
            </button>
          </div>

          <div className="status-grid">
            <div className="status-card">
              <span>检索方式</span>
              <strong>Semantic RAG</strong>
            </div>
            <div className="status-card">
              <span>Embedding</span>
              <strong>MiniLM-L12-v2</strong>
            </div>
            <div className="status-card">
              <span>LLM</span>
              <strong>DeepSeek Chat</strong>
            </div>
          </div>

          <textarea
            placeholder="例如：这篇文档主要讲了什么？核心贡献是什么？有什么结论？"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
          />

          <button onClick={handleAsk} disabled={loading}>
            {loading ? "AI 思考中..." : "提问"}
          </button>

          <button
          className="secondary-button"
          onClick={() => {
            setChatHistory([]);
            setAnswer("");
            setChunks([]);
          }}
        >
          清空聊天
        </button>

          <button onClick={handleSummary} disabled={loading}>
          {loading ? "生成中..." : "生成文档摘要"}
          </button>

          {answer && (
          <div className="answer markdown-body">
          <h3>AI 回答</h3>
          <ReactMarkdown>{answer}</ReactMarkdown>
          </div>
          )}

          {chatHistory.length > 0 && (
            <div className="history">
              <h3>聊天历史</h3>

              {chatHistory.map((item, index) => (
                <div className="history-item" key={index}>
                  <div className="history-question">
                    <strong>Q:</strong> {item.question}
                  </div>

                  <div className="history-answer markdown-body">
                    <strong>A:</strong>
                    <ReactMarkdown>{item.answer}</ReactMarkdown>
                  </div>

                  {item.chunks.length > 0 && (
                    <div className="history-chunks">
                      <strong>引用 Chunks:</strong>
                      {item.chunks.map((chunk) => (
                        <div className="mini-chunk" key={chunk.chunk_id}>
                          {chunk.filename && `${chunk.filename} · `}
                          Chunk {chunk.chunk_id} · Similarity {chunk.similarity}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}

          {summary && (
          <div className="answer markdown-body">
          <h3>文档摘要</h3>
          <ReactMarkdown>{summary}</ReactMarkdown>
          </div>
          )}

          

          {chunks.length > 0 && (
            <div className="chunks">
              <h3>检索到的相关片段</h3>

              {chunks.map((chunk) => (
                <div className="chunk" key={chunk.chunk_id}>
                  <div className="chunk-title">
                    {chunk.filename && `${chunk.filename} · `}
                    Chunk {chunk.chunk_id} · Similarity {chunk.similarity}
                  </div>
                  <p>{chunk.preview}</p>
                </div>
              ))}
            </div>
          )}
        </section>
      </main>
    </div>
  );
}

export default App;