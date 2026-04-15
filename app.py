import streamlit as st
import os
import numpy as np
from pathlib import Path
from pypdf import PdfReader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import faiss

# ─────────────────────────────────────────────
#  설정
# ─────────────────────────────────────────────
API_KEY = os.environ.get("OPENAI_API_KEY", "")
DOCS_FOLDER = Path(__file__).parent          # app.py 와 같은 폴더
CHUNK_SIZE = 1_000
CHUNK_OVERLAP = 200
TOP_K = 5
MODEL_NAME = "gpt-4o-mini"

# ─────────────────────────────────────────────
#  문서 로딩
# ─────────────────────────────────────────────
def load_pdf(path: Path) -> list[Document]:
    reader = PdfReader(str(path))
    docs = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            docs.append(Document(
                page_content=text,
                metadata={"source": path.name, "page": i + 1},
            ))
    return docs


def load_txt(path: Path) -> list[Document]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return [Document(page_content=text, metadata={"source": path.name, "page": 1})]


def load_all_docs(folder: Path) -> list[Document]:
    loaders = {".pdf": load_pdf, ".txt": load_txt, ".md": load_txt}
    docs = []
    for ext, fn in loaders.items():
        for fp in sorted(folder.glob(f"*{ext}")):
            try:
                docs.extend(fn(fp))
            except Exception as e:
                st.warning(f"⚠️ {fp.name} 로드 실패: {e}")
    return docs


# ─────────────────────────────────────────────
#  벡터 스토어 (FAISS + OpenAI Embeddings)
# ─────────────────────────────────────────────
class VectorStore:
    def __init__(self, embeddings: OpenAIEmbeddings):
        self.embeddings = embeddings
        self.index: faiss.Index | None = None
        self.docs: list[Document] = []

    def build(self, documents: list[Document]):
        texts = [d.page_content for d in documents]
        self.docs = documents
        vecs = np.array(self.embeddings.embed_documents(texts), dtype=np.float32)
        dim = vecs.shape[1]
        self.index = faiss.IndexFlatIP(dim)          # 내적(코사인 유사도)
        faiss.normalize_L2(vecs)
        self.index.add(vecs)

    def search(self, query: str, k: int = TOP_K) -> list[Document]:
        q = np.array([self.embeddings.embed_query(query)], dtype=np.float32)
        faiss.normalize_L2(q)
        _, idxs = self.index.search(q, k)
        return [self.docs[i] for i in idxs[0] if i != -1]


# ─────────────────────────────────────────────
#  RAG 시스템 초기화 (캐시)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def init_rag():
    raw_docs = load_all_docs(DOCS_FOLDER)
    if not raw_docs:
        return None, None, []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(raw_docs)

    embeddings = OpenAIEmbeddings(api_key=API_KEY, model="text-embedding-3-small")
    store = VectorStore(embeddings)
    store.build(chunks)

    llm = ChatOpenAI(api_key=API_KEY, model=MODEL_NAME, temperature=0, streaming=True)

    # 로드된 파일 목록 (중복 제거)
    file_names = sorted({d.metadata["source"] for d in raw_docs})
    return store, llm, file_names


# ─────────────────────────────────────────────
#  응답 생성
# ─────────────────────────────────────────────
FALLBACK_MSG = "해당 사실은 시설팀에 문의하시기 바랍니다"

SYSTEM_PROMPT = """\
당신은 첨부된 문서만을 근거로 질문에 답변하는 전문 AI 어시스턴트입니다.

[규칙]
1. 반드시 아래 <문서 컨텍스트>를 기반으로 답변하세요.
2. 문서 컨텍스트에 없는 내용이거나, 답변하기 어려운 질문인 경우
   반드시 아래 문장을 그대로 출력하고 다른 말은 일절 덧붙이지 마세요:
   해당 사실은 시설팀에 문의하시기 바랍니다
3. 답변 가능한 경우 출처(파일명·페이지)를 답변 마지막에 명시하세요.
4. 항상 한국어로 답변하세요.

<문서 컨텍스트>
{context}
</문서 컨텍스트>"""


def build_context(docs: list[Document]) -> str:
    parts = []
    for d in docs:
        src = d.metadata.get("source", "알 수 없음")
        pg = d.metadata.get("page", "?")
        parts.append(f"[{src} / p.{pg}]\n{d.page_content}")
    return "\n\n---\n\n".join(parts)


def stream_response(query: str, history: list[dict], store: VectorStore, llm: ChatOpenAI):
    """스트리밍 방식으로 응답 반환"""
    relevant = store.search(query)
    context = build_context(relevant)

    messages = [{"role": "system", "content": SYSTEM_PROMPT.format(context=context)}]
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": query})

    for chunk in llm.stream(messages):
        yield chunk.content


# ─────────────────────────────────────────────
#  Streamlit UI
# ─────────────────────────────────────────────
def main():
    # ── 페이지 설정 ──
    st.set_page_config(
        page_title="RAG 챗봇",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── 커스텀 CSS ──
    st.markdown("""
    <style>
        /* ── 전체 배경: 흰색 ── */
        .stApp { background: #ffffff; color: #1a1a1a; }

        /* ── 메인 컨테이너 ── */
        .block-container { background: #ffffff; }

        /* ── 사이드바: 오렌지 계열 ── */
        [data-testid="stSidebar"] {
            background: #ff6b00;
            border-right: none;
        }
        [data-testid="stSidebar"] * {
            color: #ffffff !important;
        }
        [data-testid="stSidebar"] .stMarkdown h2,
        [data-testid="stSidebar"] .stMarkdown h3 {
            color: #ffffff !important;
        }
        [data-testid="stSidebar"] hr {
            border-color: rgba(255,255,255,0.35) !important;
        }
        /* 사이드바 버튼 */
        [data-testid="stSidebar"] button {
            background: #ffffff !important;
            color: #ff6b00 !important;
            border: none !important;
            font-weight: 700 !important;
            border-radius: 8px !important;
        }
        [data-testid="stSidebar"] button:hover {
            background: #ffe8d6 !important;
        }

        /* ── 채팅 입력창 ── */
        [data-testid="stChatInput"] textarea {
            background: #fff7f2 !important;
            color: #1a1a1a !important;
            border: 2px solid #ff6b00 !important;
            border-radius: 12px !important;
        }
        [data-testid="stChatInput"] textarea:focus {
            box-shadow: 0 0 0 3px rgba(255,107,0,0.18) !important;
        }

        /* ── 전송 버튼 ── */
        [data-testid="stChatInput"] button {
            background: #ff6b00 !important;
            color: #ffffff !important;
            border-radius: 8px !important;
        }

        /* ── user 말풍선 ── */
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
            background: #fff3ea;
            border-radius: 14px;
            padding: 12px 16px;
            margin-bottom: 8px;
            border-left: 4px solid #ff6b00;
            color: #1a1a1a;
        }

        /* ── assistant 말풍선 ── */
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
            background: #f9f9f9;
            border-radius: 14px;
            padding: 12px 16px;
            margin-bottom: 8px;
            border-left: 4px solid #e0e0e0;
            color: #1a1a1a;
        }

        /* ── 텍스트 색상 전반 ── */
        p, span, li, td, th, label, div { color: #1a1a1a; }
        h1, h2, h3, h4 { color: #1a1a1a !important; letter-spacing: -0.5px; }

        /* ── 타이틀 오렌지 강조 ── */
        h1 { color: #ff6b00 !important; }

        /* ── 문서 뱃지 (사이드바 내) ── */
        .badge {
            display: inline-block;
            background: rgba(255,255,255,0.2);
            color: #ffffff;
            border: 1px solid rgba(255,255,255,0.5);
            border-radius: 6px;
            padding: 3px 10px;
            font-size: 0.78rem;
            margin: 2px 0;
        }

        /* ── caption 색상 ── */
        [data-testid="stSidebar"] small,
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
            color: rgba(255,255,255,0.8) !important;
        }

        /* ── 구분선 ── */
        hr { border-color: #f0e0d4 !important; }

        /* ── spinner ── */
        .stSpinner > div { border-top-color: #ff6b00 !important; }

        /* ── 경고/에러 박스 ── */
        [data-testid="stAlertContainer"] { border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

    # ── RAG 초기화 ──
    with st.spinner("📖 문서 색인(인덱싱) 중..."):
        store, llm, file_names = init_rag()

    # ── 사이드바 ──
    with st.sidebar:
        st.markdown("## 📚 RAG 챗봇")
        st.caption("업로드된 문서를 기반으로 질의응답합니다.")
        st.divider()

        if file_names:
            st.markdown("### 🗂️ 색인된 문서")
            for fn in file_names:
                st.markdown(f'<div class="badge">📄 {fn}</div>', unsafe_allow_html=True)
        else:
            st.warning("폴더에 문서가 없습니다.")

        st.divider()
        st.markdown(f"**모델**: `{MODEL_NAME}`")
        st.markdown(f"**청크 크기**: `{CHUNK_SIZE}` / **오버랩**: `{CHUNK_OVERLAP}`")
        st.markdown(f"**검색 Top-K**: `{TOP_K}`")

        st.divider()
        if st.button("🗑️ 대화 초기화", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # ── 메인 영역 ──
    st.markdown("# 📚 RAG Document Chatbot")
    st.caption("문서에 대해 자유롭게 질문하세요. 한국어 / English 모두 지원합니다.")

    if store is None:
        st.error("❌ 문서를 찾을 수 없습니다. project_rag 폴더에 PDF/TXT 파일을 추가 후 재시작하세요.")
        return

    # 세션 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 이전 메시지 렌더링
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 입력
    if prompt := st.chat_input("질문을 입력하세요..."):
        # 사용자 메시지 표시
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 어시스턴트 응답 (스트리밍)
        with st.chat_message("assistant"):
            response = st.write_stream(
                stream_response(
                    prompt,
                    st.session_state.messages[:-1],
                    store,
                    llm,
                )
            )

        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
