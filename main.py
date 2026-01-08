from pathlib import Path
from src.utils import setup_logging
from src.document_parser import DocumentParser
from src.chunker import HybridChunker
from src.embedder import MultilingualEmbedder, EmbeddingCache
from src.retrieval import rerank_search, load_default_reranker

def main():
    setup_logging()

    pdf_path = Path("data/HR-Guide_-Policy-and-Procedure-Template.pdf")

    doc = DocumentParser().parse(pdf_path)
    chunks = HybridChunker().chunk_document(doc.content, pdf_path.stem, doc.metadata)

    embedder = MultilingualEmbedder()
    cache = EmbeddingCache()
    key = "hr_policy_chunks_e5_large"

    embs = cache.load(key)
    if embs is None:
        embs = embedder.embed_documents([c.text for c in chunks])
        cache.save(embs, key)

    reranker = load_default_reranker()
    results = rerank_search("What is considered overtime?", embedder, embs, chunks, reranker, top_k=5)

    for r in results:
        print(f"[{r.rank}] {r.score:.4f} | {r.section_header} | {r.chunk_id}")
        print(r.text_preview, "\n")

if __name__ == "__main__":
    main()
