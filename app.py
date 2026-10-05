"""PowerGym Virtual Assistant (Chainlit) – replica del progetto "virtual_assistant" della lezione.

Prima volta:  poetry run python create_assistant.py
Avvio:        poetry run chainlit run app.py -w
"""

import chainlit as cl
from openai import APIConnectionError, APIError, APITimeoutError

from gym_assistant.chat_logic import create_chat_logic
from gym_assistant.config import ASSISTANT_NAME, MODEL, PROVIDER

AUTHOR = ASSISTANT_NAME


@cl.set_starters
async def set_starters():
    return [
        cl.Starter(label="Orari di apertura", message="Quali sono gli orari della palestra?"),
        cl.Starter(label="Abbonamenti", message="Quanto costano gli abbonamenti?"),
        cl.Starter(label="Certificato medico", message="Serve il certificato medico per iscriversi?"),
        cl.Starter(label="Ricevi la brochure", message="Mi chiamo Luca, la mia email è luca@example.com"),
    ]


@cl.on_chat_start
async def on_chat_start():
    """Inizializza una nuova sessione: ogni chat ha la sua ChatLogic (e il suo thread)."""
    try:
        chat_logic = create_chat_logic()
    except Exception as e:
        await cl.Message(author=AUTHOR, content=f"Errore di avvio: {e}").send()
        return

    if PROVIDER != "openai" and chat_logic.kb.count() == 0:
        await cl.Message(
            author=AUTHOR,
            content="⚠️ La knowledge base è vuota: esegui `poetry run python create_assistant.py` e riavvia.",
        ).send()

    cl.user_session.set("chat_logic", chat_logic)
    print(f"[PowerGym] provider={PROVIDER} model={MODEL}")


@cl.on_message
async def main(message: cl.Message):
    chat_logic = cl.user_session.get("chat_logic")
    try:
        # process_message è sincrona: make_async evita di bloccare l'interfaccia
        reply = await cl.make_async(chat_logic.process_message)(message.content)
    except APITimeoutError:
        await cl.Message(author=AUTHOR, content="⏱️ Il modello non ha risposto in tempo, riprova.").send()
        return
    except APIConnectionError:
        await cl.Message(author=AUTHOR, content="🔌 Impossibile contattare il modello: Ollama è avviato?").send()
        return
    except APIError as e:
        await cl.Message(author=AUTHOR, content=f"Errore del modello: {e}").send()
        return

    # Fonti consultate nella knowledge base (equivalente di file_search)
    async with cl.Step(name="Knowledge base", type="retrieval") as step:
        step.input = message.content
        step.output = "\n\n".join(
            f"**{s['source']}** (distanza {s['distance']})\n{s['text'][:300]}" for s in reply.sources
        ) or "Nessuna sezione pertinente."

    # Funzioni eseguite (function calling)
    for call in reply.tool_calls:
        async with cl.Step(name=call["name"], type="tool") as step:
            step.input = call["arguments"]
            step.output = call["output"]
        print("*" * 80, "\nDEBUG", call, "\n" + "*" * 80)

    await cl.Message(author=AUTHOR, content=reply.text).send()
