"""Invio dell'email di benvenuto con Resend (come nella lezione) o simulato su file."""

import html
from datetime import datetime

import resend

from gym_assistant.config import EMAIL_FROM, OUTBOX_DIR, RESEND_KEY

SUBJECT = "Benvenuto in PowerGym! Ecco la tua brochure"


def brochure_html(name: str) -> str:
    nome = html.escape(name or "")
    return f"""
<div style="font-family:Arial,sans-serif;max-width:560px">
  <h2 style="color:#f97316">Ciao {nome}, benvenuto in PowerGym!</h2>
  <p>Grazie per averci contattato. Ecco le informazioni principali:</p>
  <h3>Orari</h3>
  <ul>
    <li>Lun–Ven: 06:30 – 22:30</li><li>Sabato: 08:00 – 20:00</li><li>Domenica: 09:00 – 13:00</li>
  </ul>
  <h3>Abbonamenti</h3>
  <ul>
    <li>Mensile: 59,00 € (sala pesi + 1 corso)</li>
    <li>Trimestrale: 159,00 € (sala pesi + 2 corsi)</li>
    <li>Annuale: 549,00 € (tutto incluso + 2 sessioni con personal trainer)</li>
    <li>Open Student under 26: 45,00 €/mese</li>
  </ul>
  <p>Ti aspettiamo per la tua <b>giornata di prova gratuita</b>!</p>
  <p>Cordiali saluti,<br/>Lo staff PowerGym</p>
</div>"""


def send_email(name: str, email: str) -> str:
    """Invia la brochure. Senza RESEND_KEY l'email viene salvata in outbox/."""
    body = brochure_html(name)

    if RESEND_KEY:
        try:
            resend.api_key = RESEND_KEY
            resend.Emails.send({"from": EMAIL_FROM, "to": email, "subject": SUBJECT, "html": body})
            return f"Email inviata con successo a {email}"
        except Exception as e:
            return f"Invio email non riuscito: {e}"

    OUTBOX_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    file = OUTBOX_DIR / f"{stamp}_{email.replace('@', '_at_')}.html"
    file.write_text(f"<p><b>A:</b> {html.escape(email)}<br/><b>Oggetto:</b> {SUBJECT}</p>{body}", encoding="utf-8")
    return f"Email inviata con successo a {email} (simulata: salvata in outbox/{file.name})"
