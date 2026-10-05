"""Knowledge base locale: l'equivalente di file_search + vector store di OpenAI.

I file in kb/ vengono divisi in sezioni (per titolo), trasformati in embedding con
Ollama (bge-m3, multilingua) e salvati in ChromaDB. A ogni domanda si recuperano
le sezioni più simili.
"""

import re
from pathlib import Path

import chromadb

from gym_assistant.config import CHROMA_DIR, COLLECTION, EMBED_MODEL, client

def split_sections(path: Path) -> list[dict]:
    """Divide un file Markdown/testo in sezioni, una per titolo (#, ##, ###)."""
    sections, titles, buffer = [], {}, []

    def flush():
        text = "\n".join(buffer).strip()
        if text:
            heading = " > ".join(titles[k] for k in sorted(titles))
            sections.append({"text": f"{heading}\n{text}" if heading else text, "source": path.name})
        buffer.clear()

    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^(#{1,3})\s+(.*)", line)
        if match:
            flush()
            level = len(match.group(1))
            titles = {k: v for k, v in titles.items() if k < level}
            titles[level] = match.group(2).strip()
        else:
            buffer.append(line)
    flush()
    return sections


class KnowledgeBase:
    def __init__(self):
        self.db = chromadb.PersistentClient(path=str(CHROMA_DIR))

    def _collection(self):
        # Gli embedding li calcoliamo noi con Ollama: niente embedding function di Chroma
        return self.db.get_or_create_collection(
            COLLECTION, metadata={"hnsw:space": "cosine"}, embedding_function=None
        )

    @staticmethod
    def embed(texts: list[str]) -> list[list[float]]:
        response = client.embeddings.create(model=EMBED_MODEL, input=texts)
        return [item.embedding for item in response.data]

    def build(self, kb_dir: Path) -> int:
        """Ricrea la collezione a partire dai file della cartella kb/."""
        try:
            self.db.delete_collection(COLLECTION)
        except Exception:
            pass
        collection = self._collection()

        sections = [s for f in sorted(kb_dir.glob("*")) if f.suffix in (".md", ".txt") for s in split_sections(f)]
        collection.add(
            ids=[f"sec-{i}" for i in range(len(sections))],
            documents=[s["text"] for s in sections],
            metadatas=[{"source": s["source"]} for s in sections],
            embeddings=self.embed([s["text"] for s in sections]),
        )
        return len(sections)

    def count(self) -> int:
        return self._collection().count()

    def search(self, query: str, k: int = 4) -> list[dict]:
        """Le k sezioni più simili alla domanda (distanza coseno: più bassa = più pertinente)."""
        result = self._collection().query(query_embeddings=self.embed([query]), n_results=k)
        return [
            {"text": doc, "source": meta["source"], "distance": round(dist, 3)}
            for doc, meta, dist in zip(result["documents"][0], result["metadatas"][0], result["distances"][0])
        ]
