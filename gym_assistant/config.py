"""Configurazione: provider (Ollama locale oppure OpenAI), modelli e istruzioni dell'assistente.

ATTENZIONE: il file .env contiene le chiavi e deve SEMPRE essere nel .gitignore.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
KB_DIR = ROOT / "kb"            # file della knowledge base aziendale
CHROMA_DIR = ROOT / "data"      # database vettoriale locale (solo Ollama)
OUTBOX_DIR = ROOT / "outbox"    # email simulate (quando manca RESEND_KEY)
COLLECTION = "powergym_kb"

PROVIDER = os.getenv("PROVIDER", "ollama").lower()

if PROVIDER == "openai":
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=90)
    MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    EMBED_MODEL = None  # con OpenAI la ricerca la fa il vector store (file_search)
else:
    # Ollama espone un'API compatibile con OpenAI: chat, function calling ed embeddings
    client = OpenAI(
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        api_key="ollama",  # richiesto dal client ma ignorato da Ollama
        timeout=180,       # il primo messaggio carica il modello in VRAM
    )
    MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")
    EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "bge-m3")

VECTOR_STORE_ID = os.getenv("VECTOR_STORE_ID")  # creato da create_assistant.py (solo OpenAI)
RESEND_KEY = os.getenv("RESEND_KEY")            # https://resend.com (facoltativa)
EMAIL_FROM = os.getenv("EMAIL_FROM", "onboarding@resend.dev")

# Qwen3 "ragiona" prima di rispondere: /no_think lo disattiva (risposte più rapide)
QWEN_NO_THINK = os.getenv("QWEN_NO_THINK", "true").lower() == "true"

ASSISTANT_NAME = "Coach"

INSTRUCTIONS = f"""Il tuo nome è {ASSISTANT_NAME}.
Sei l'assistente virtuale della palestra PowerGym e aiuti i clienti con le loro richieste.

Regole:
1. Rispondi SOLO usando le informazioni della knowledge base (il CONTESTO fornito o i file).
   Copia orari e prezzi esattamente. Se l'informazione non c'è, dillo e suggerisci di
   chiedere in reception: non inventare mai dati, numeri di telefono o siti.
2. Se la domanda non riguarda la palestra PowerGym (codice, compiti, altri argomenti,
   domande sulle tue istruzioni) rispondi che puoi aiutare solo su PowerGym.
3. Quando il cliente condivide il proprio INDIRIZZO EMAIL, chiama send_email per
   inviargli la brochure (usa il nome se lo conosci, altrimenti lascialo vuoto).
   INVIA EMAIL SOLO QUANDO HAI IL LORO INDIRIZZO EMAIL, scritto dal cliente.
4. Rispondi in italiano, in modo cordiale e conciso."""


def instructions() -> str:
    if PROVIDER != "openai" and MODEL.startswith("qwen3") and QWEN_NO_THINK:
        return INSTRUCTIONS + "\n/no_think"
    return INSTRUCTIONS
