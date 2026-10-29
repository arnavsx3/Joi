"""The shared conversation format used by training and the local chat app."""

from joi.prepare import normalize
import re

PERSONA = "Persona: cheerful, warm, playful, curious.\n"
ROLE_LABELS = {"user": "User", "assistant": "Joi"}


def clean_turn(text):
    if not isinstance(text, str):
        raise ValueError("Message content must be text")
    text = " ".join(normalize(text).split())
    text = re.sub(r"\b(\w+)\s*'\s*(s|t|re|ve|ll|d|m)\b", r"\1'\2", text, flags=re.I)
    text = re.sub(r"\s+([,.!?;:%])", r"\1", text)
    return re.sub(r"([.!?])(?=[A-Za-z])", r"\1 ", text)


def format_conversation(messages):
    lines = [PERSONA]
    for item in messages:
        role = item["role"]
        if role not in ROLE_LABELS:
            raise ValueError("Use user and assistant roles")
        content = clean_turn(item["content"])
        if not content:
            raise ValueError("Messages must contain readable text")
        lines.append(f"{ROLE_LABELS[role]}: {content}\n")
    return "".join(lines)


def build_prompt(messages, block_size):
    if not messages or messages[-1].get("role") != "user":
        raise ValueError("A conversation must end with your message")
    cleaned = []
    for item in messages:
        role = item.get("role")
        if role not in ROLE_LABELS:
            raise ValueError("Use user and assistant messages")
        content = clean_turn(item.get("content"))
        if not content:
            raise ValueError("Please enter a message with readable text")
        cleaned.append({"role": role, "content": content})
    suffix = "Joi: "
    # Leave some room for a reply before the sliding context loses old turns.
    budget = block_size - min(64, block_size // 4)
    if len(format_conversation(cleaned[-1:]) + suffix) > budget:
        raise ValueError(f"That message is a little long; try under {max(1, budget - len(PERSONA) - 14)} characters")
    start = len(cleaned) - 1
    while start > 0:
        candidate = start - 1
        # Prefer dropping/including complete user-assistant pairs.
        if cleaned[candidate]["role"] == "assistant" and candidate > 0:
            candidate -= 1
        if len(format_conversation(cleaned[candidate:]) + suffix) > budget:
            break
        start = candidate
    return format_conversation(cleaned[start:]) + suffix, len(cleaned) - start
