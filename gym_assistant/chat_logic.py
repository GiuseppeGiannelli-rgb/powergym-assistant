"""Logica della chat (stessa struttura della classe ChatLogic della lezione).

L'API Assistants di OpenAI (client.beta.assistants / threads / runs) usata nella lezione
è stata dismessa il 26 agosto 2026. Lo stesso assistente è realizzato qui in due modi:

- OpenAIChatLogic: Responses API, l'erede ufficiale degli Assistants
  (file_search sul vector store + function calling + conversazione con previous_response_id)
- LocalChatLogic: tutto in locale con Ollama
  (knowledge base ChromaDB al posto di file_search + function calling + cronologia come "thread")
"""

import json
import re
from dataclasses import dataclass, field

from gym_assistant.config import MODEL, PROVIDER, VECTOR_STORE_ID, client, instructions
from gym_assistant.emailer import send_email

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(\.[\w-]+)+")
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

# Definizione del tool di invio email (function calling), come in create_assistant.py del prof
SEND_EMAIL_PARAMETERS = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "nome del potenziale cliente"},
        "email": {"type": "string", "description": "email del potenziale cliente"},
    },
    "required": ["name", "email"],
    "additionalProperties": False,
}
SEND_EMAIL_DESCRIPTION = "Invia al potenziale cliente un'email con la brochure quando fornisce il proprio indirizzo email"


@dataclass
class Reply:
    """Risposta dell'assistente con le informazioni da mostrare nella UI."""

    text: str
    sources: list[dict] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)


class ChatLogic:
    """Parte comune: esecuzione delle funzioni richieste dal modello."""

    def __init__(self):
        self.user_texts: list[str] = []  # messaggi scritti dal cliente (per validare le email)

    def send_email(self, name: str = "", email: str = "") -> str:
        """Invia l'email SOLO se l'indirizzo è valido ed è stato scritto davvero dal cliente."""
        email = (email or "").strip()
        if not EMAIL_RE.fullmatch(email):
            return f"Email non inviata: '{email}' non è un indirizzo valido. Chiedi l'email al cliente."
        if not any(email.lower() in t.lower() for t in self.user_texts):
            return "Email non inviata: il cliente non ha scritto questo indirizzo. Chiedi l'email al cliente."
        return send_email(name.strip(), email)

    def execute(self, name: str, arguments: str) -> str:
        try:
            args = json.loads(arguments or "{}")
        except json.JSONDecodeError:
            return f"Argomenti non validi: {arguments}"
        if name == "send_email":
            return self.send_email(**{k: v for k, v in args.items() if k in ("name", "email")})
        return f"Unknown function: {name}"

    def process_message(self, user_message: str) -> Reply:
        raise NotImplementedError


class LocalChatLogic(ChatLogic):
    """Assistente locale: Ollama + knowledge base ChromaDB."""

    TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "send_email",
                "description": SEND_EMAIL_DESCRIPTION,
                "parameters": SEND_EMAIL_PARAMETERS,
            },
        }
    ]
    MAX_HISTORY = 10   # messaggi del "thread" inviati al modello
    MAX_ROUNDS = 3     # giri modello -> funzione -> modello

    def __init__(self):
        super().__init__()
        from gym_assistant.knowledge_base import KnowledgeBase

        self.kb = KnowledgeBase()
        self.thread: list[dict] = []  # equivalente locale del thread: solo utente/assistente

    def initialize_thread(self) -> list[dict]:
        return self.thread

    def process_message(self, user_message: str) -> Reply:
        thread = self.initialize_thread()
        self.user_texts.append(user_message)
        reply = Reply(text="")

        # 1. Ricerca nella knowledge base (sempre, come file_search) -> contesto
        reply.sources = self.kb.search(user_message)
        if reply.sources:
            context = "\n\n---\n\n".join(s["text"] for s in reply.sources)
        else:
            context = "Nessuna informazione pertinente trovata nella knowledge base."

        thread.append({"role": "user", "content": user_message})
        messages = [
            {"role": "system", "content": f"{instructions()}\n\nCONTESTO (knowledge base PowerGym):\n{context}"},
            *thread[-self.MAX_HISTORY:],
        ]

        # 2. Ciclo di function calling
        for _ in range(self.MAX_ROUNDS):
            completion = client.chat.completions.create(
                model=MODEL, messages=messages, tools=self.TOOLS, tool_choice="auto", temperature=0
            )
            message = completion.choices[0].message
            if not message.tool_calls:
                reply.text = THINK_RE.sub("", message.content or "").strip()
                break

            messages.append({
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in message.tool_calls
                ],
            })
            for tc in message.tool_calls:
                output = self.execute(tc.function.name, tc.function.arguments)
                reply.tool_calls.append({"name": tc.function.name, "arguments": tc.function.arguments, "output": output})
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": output})

        # 3. Rete di sicurezza: email scritta dal cliente ma funzione non chiamata dal modello
        email = EMAIL_RE.search(user_message)
        if email and not any(c["name"] == "send_email" for c in reply.tool_calls):
            arguments = json.dumps({"name": "", "email": email.group(0)})
            output = self.execute("send_email", arguments)
            reply.tool_calls.append({"name": "send_email", "arguments": arguments, "output": output})
            reply.text = (reply.text + "\n\n" if reply.text else "") + "📧 Ti ho inviato la brochure di PowerGym via email."

        reply.text = reply.text or "Scusa, non sono riuscito a rispondere. Puoi riformulare la domanda?"
        thread.append({"role": "assistant", "content": reply.text})
        return reply


class OpenAIChatLogic(ChatLogic):
    """Assistente OpenAI con la Responses API (sostituisce Assistants/Threads/Runs)."""

    def __init__(self):
        super().__init__()
        self.previous_response_id: str | None = None  # equivalente del thread_id
        self.tools = [
            {"type": "function", "name": "send_email", "description": SEND_EMAIL_DESCRIPTION,
             "parameters": SEND_EMAIL_PARAMETERS, "strict": True},
        ]
        if VECTOR_STORE_ID:
            self.tools.append({"type": "file_search", "vector_store_ids": [VECTOR_STORE_ID]})

    def _create(self, input_items: list[dict]):
        return client.responses.create(
            model=MODEL,
            instructions=instructions(),
            input=input_items,
            tools=self.tools,
            previous_response_id=self.previous_response_id,
            include=["file_search_call.results"],
        )

    def process_message(self, user_message: str) -> Reply:
        self.user_texts.append(user_message)
        reply = Reply(text="")

        response = self._create([{"role": "user", "content": user_message}])

        # Finché il modello richiede funzioni, le eseguiamo e restituiamo i risultati
        for _ in range(3):
            self.previous_response_id = response.id
            for item in response.output:
                if item.type == "file_search_call" and item.results:
                    reply.sources += [
                        {"text": r.text, "source": r.filename, "distance": round(1 - (r.score or 0), 3)}
                        for r in item.results
                    ]
            calls = [item for item in response.output if item.type == "function_call"]
            if not calls:
                break
            outputs = []
            for call in calls:
                output = self.execute(call.name, call.arguments)
                reply.tool_calls.append({"name": call.name, "arguments": call.arguments, "output": output})
                outputs.append({"type": "function_call_output", "call_id": call.call_id, "output": output})
            response = self._create(outputs)

        self.previous_response_id = response.id
        reply.text = response.output_text or "Scusa, non sono riuscito a rispondere."
        return reply


def create_chat_logic() -> ChatLogic:
    return OpenAIChatLogic() if PROVIDER == "openai" else LocalChatLogic()
