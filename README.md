# Selfwork OpenAI Assistant – PowerGym Virtual Assistant

Replica del progetto della video-lezione (*VerdePower Italia – assistente Sofia*) su un argomento a scelta: **l'assistente virtuale della palestra PowerGym**.

Come nell'originale, l'assistente:

- risponde usando una **knowledge base aziendale** (cartella `kb/`) tramite ricerca semantica (*file_search*);
- quando il cliente scrive il proprio **indirizzo email**, chiama la funzione `send_email` (**function calling**) e gli invia una brochure con **Resend**;
- mantiene la **conversazione** (thread) per tutta la sessione;
- ha un'interfaccia **Chainlit** che mostra le fonti consultate e le funzioni eseguite.

## Nota: Assistants API dismessa

L'API usata nella lezione (`client.beta.assistants`, `threads`, `runs`) è stata **dismessa da OpenAI il 26 agosto 2026**. Il progetto quindi realizza lo stesso assistente con:

| Lezione (Assistants API) | Questo progetto – OpenAI | Questo progetto – locale (Ollama) |
|---|---|---|
| `assistants.create(instructions, tools)` | `responses.create(instructions, tools)` | system prompt + `chat.completions` |
| `vector_stores` + `file_search` | `vector_stores` + `file_search` | ChromaDB + embedding `bge-m3` |
| `threads` / `runs` | `previous_response_id` | cronologia della sessione |
| `requires_action` → `submit_tool_outputs` | `function_call` → `function_call_output` | `tool_calls` → messaggio `tool` |

## Struttura

```
create_assistant.py          # da lanciare una volta: crea la knowledge base / il vector store
app.py                       # interfaccia Chainlit
gym_assistant/
    config.py                # provider, modelli, istruzioni dell'assistente
    chat_logic.py            # ChatLogic: LocalChatLogic (Ollama) e OpenAIChatLogic (Responses API)
    knowledge_base.py        # indicizzazione e ricerca nella kb con ChromaDB
    emailer.py               # invio email con Resend (o simulato in outbox/)
kb/                          # knowledge base: info.md, regolamento_faq.md
public/                      # tema, logo, avatar
```

## Sicurezza

- `send_email` parte solo se l'indirizzo è valido **ed è stato scritto davvero dal cliente** (il modello non può inventarlo).
- Se il cliente scrive un'email ma il modello non chiama la funzione, il codice la esegue comunque.
- Le chiavi stanno solo nel `.env`, escluso da git.

## Avvio (locale con Ollama)

```bash
poetry install
cp .env.example .env
ollama pull qwen3:8b
ollama pull bge-m3
poetry run python create_assistant.py     # una sola volta (e quando modifichi kb/)
poetry run chainlit run app.py -w
```

Con OpenAI: `PROVIDER=openai` e `OPENAI_API_KEY` nel `.env`, poi `create_assistant.py` stampa il `VECTOR_STORE_ID` da copiare nel `.env`.

Email reali: inserisci `RESEND_KEY` (account gratuito su resend.com; con il mittente di test `onboarding@resend.dev` si può inviare solo all'email del proprio account). Senza chiave le email vengono salvate come HTML in `outbox/`.
