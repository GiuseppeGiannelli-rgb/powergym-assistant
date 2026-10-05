"""Da lanciare UNA SOLA VOLTA (e ogni volta che modifichi i file in kb/).

Equivalente di create_assistant.py della lezione:
- Ollama: crea la knowledge base locale (ChromaDB) dai file in kb/
- OpenAI: crea il vector store e carica i file in kb/ (da usare con file_search)

    poetry run python create_assistant.py
"""

from gym_assistant.config import EMBED_MODEL, KB_DIR, PROVIDER, client

KB_FILES = sorted(p for p in KB_DIR.glob("*") if p.suffix in (".md", ".txt"))


def create_openai_vector_store() -> str:
    # Il vector store è un database ottimizzato per la ricerca semantica, gestito da OpenAI
    vector_store = client.vector_stores.create(name="PowerGym Infos")
    streams = [open(path, "rb") for path in KB_FILES]
    try:
        batch = client.vector_stores.file_batches.upload_and_poll(
            vector_store_id=vector_store.id, files=streams
        )
    finally:
        for s in streams:
            s.close()
    print(f"File caricati: {batch.file_counts.completed}/{batch.file_counts.total}")
    return vector_store.id


def create_local_knowledge_base() -> int:
    from gym_assistant.knowledge_base import KnowledgeBase

    return KnowledgeBase().build(KB_DIR)


if __name__ == "__main__":
    print(f"Provider: {PROVIDER} | File: {[p.name for p in KB_FILES]}")
    if PROVIDER == "openai":
        vs_id = create_openai_vector_store()
        print(f"\n\nVECTOR_STORE_ID={vs_id}\n\nCopia questa riga nel file .env\n")
    else:
        n = create_local_knowledge_base()
        print(f"\nKnowledge base locale creata: {n} sezioni indicizzate con {EMBED_MODEL}\n")
