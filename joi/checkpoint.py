"""Local checkpoint persistence, with restricted PyTorch loading."""

from dataclasses import asdict
from pathlib import Path

import torch

from joi.model import Joi, ModelConfig
from joi.tokenizer import CharTokenizer


def save_checkpoint(path, model, tokenizer, **state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "model_config": asdict(model.config),
        "model_state": model.state_dict(),
        "tokenizer_chars": tokenizer.chars,
        **state,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint version")
    tokenizer = CharTokenizer(payload["tokenizer_chars"])
    config = ModelConfig(**payload["model_config"])
    if config.vocab_size != len(tokenizer.chars):
        raise ValueError("Checkpoint vocabulary does not match model")
    model = Joi(config)
    model.load_state_dict(payload["model_state"])
    return model, tokenizer, payload
