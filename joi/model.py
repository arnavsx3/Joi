"""A GPT-style, decoder-only transformer built from PyTorch components."""

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    vocab_size: int
    block_size: int = 128
    n_embd: int = 128
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1
    attention_dropout: float | None = None

    def __post_init__(self):
        for name in ("vocab_size", "block_size", "n_embd", "n_heads", "n_layers"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.n_embd % self.n_heads:
            raise ValueError("n_embd must be divisible by n_heads")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.attention_dropout is not None and not 0 <= self.attention_dropout < 1:
            raise ValueError("attention_dropout must be in [0, 1) or None")


class CausalAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.dropout = config.dropout if config.attention_dropout is None else config.attention_dropout
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [
            t.view(batch, length, self.n_heads, width // self.n_heads).transpose(1, 2)
            for t in (q, k, v)
        ]
        # A character can only attend to itself and earlier characters.
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(batch, length, width)
        return self.residual_dropout(self.proj(y))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attention = CausalAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class Joi(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.Sequential(*(Block(config) for _ in range(config.n_layers)))
        self.final_norm = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        # Reuse the character embeddings as the output projection.
        self.head.weight = self.token_embedding.weight

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, tokens, targets=None, *, return_hidden=False):
        if tokens.ndim != 2 or not 1 <= tokens.shape[1] <= self.config.block_size:
            raise ValueError(f"Expected a nonempty [batch, length] up to {self.config.block_size}")
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        x = self.dropout(self.token_embedding(tokens) + self.position_embedding(positions))
        hidden = self.final_norm(self.blocks(x))
        logits = self.head(hidden)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return (logits, loss, hidden) if return_hidden else (logits, loss)

    @torch.inference_mode()
    def generate(self, tokens, max_new_tokens=300, temperature=0.8, top_k=20, stop_sequences=None,
                 no_repeat_ngram_size=0, allowed_tokens_fn=None, num_beams=1):
        if max_new_tokens < 0 or not math.isfinite(temperature) or temperature <= 0 or top_k < 0:
            raise ValueError("Use nonnegative length/top_k and a finite, positive temperature")
        if stop_sequences and (tokens.shape[0] != 1 or any(not sequence for sequence in stop_sequences)):
            raise ValueError("Stop sequences require one prompt and nonempty sequences")
        if no_repeat_ngram_size < 0 or (no_repeat_ngram_size and tokens.shape[0] != 1):
            raise ValueError("Repeated-pattern blocking requires one prompt and a nonnegative size")
        if allowed_tokens_fn is not None and tokens.shape[0] != 1:
            raise ValueError("Token constraints require one prompt")
        if not isinstance(num_beams, int) or isinstance(num_beams, bool) or not 1 <= num_beams <= 8:
            raise ValueError("Use between one and eight beams")
        if num_beams > 1:
            if tokens.shape[0] != 1:
                raise ValueError("Beam search requires one prompt")
            return self._beam_generate(tokens, max_new_tokens, temperature, top_k,
                                       stop_sequences, no_repeat_ngram_size, allowed_tokens_fn, num_beams)
        prompt_length = tokens.shape[1]
        was_training = self.training
        self.eval()
        try:
            for index in range(max_new_tokens):
                logits, _ = self(tokens[:, -self.config.block_size:])
                logits = logits[:, -1] / temperature
                if allowed_tokens_fn is not None:
                    allowed = allowed_tokens_fn(tokens[0, prompt_length:].tolist())
                    mask = torch.ones_like(logits, dtype=torch.bool)
                    mask[:, allowed] = False
                    logits = logits.masked_fill(mask, float("-inf"))
                if no_repeat_ngram_size:
                    generated = tokens[0,prompt_length:].tolist()
                    size = no_repeat_ngram_size
                    if len(generated) >= size - 1:
                        prefix = generated[-(size-1):] if size>1 else []
                        banned = {generated[start+size-1] for start in range(len(generated)-size+1)
                                  if generated[start:start+size-1] == prefix}
                        if banned:
                            logits[:,list(banned)] = float("-inf")
                if not torch.isfinite(logits).any():
                    raise ValueError("No valid next token under the generation constraints")
                if top_k:
                    cutoff = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
                    logits = logits.masked_fill(logits < cutoff, float("-inf"))
                token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
                tokens = torch.cat((tokens, token), dim=1)
                for sequence in stop_sequences or ():
                    if index + 1 >= len(sequence) and tokens[0, -len(sequence):].tolist() == list(sequence):
                        return tokens[:, :-len(sequence)]
            return tokens
        finally:
            self.train(was_training)

    @torch.inference_mode()
    def _beam_generate(self, prompt, limit, temperature, top_k, stops, repeat, allowed_fn, width):
        """Explore multiple neural continuations; no response database or reranker."""
        was_training = self.training
        self.eval()
        active = [(prompt, 0.0)]
        finished = []
        prefix_length = prompt.shape[1]
        try:
            for _ in range(limit):
                candidates = []
                sequences = torch.cat([sequence for sequence, _ in active])
                logits, _ = self(sequences[:, -self.config.block_size:])
                for row, (sequence, score) in enumerate(active):
                    generated = sequence[0, prefix_length:].tolist()
                    values = logits[row, -1] / temperature
                    if allowed_fn is not None:
                        mask = torch.ones_like(values, dtype=torch.bool)
                        mask[allowed_fn(generated)] = False
                        values = values.masked_fill(mask, float("-inf"))
                    if repeat and len(generated) >= repeat - 1:
                        suffix = generated[-(repeat - 1):] if repeat > 1 else []
                        banned = {generated[start + repeat - 1] for start in range(len(generated) - repeat + 1)
                                  if generated[start:start + repeat - 1] == suffix}
                        if banned:
                            values[list(banned)] = float("-inf")
                    if top_k:
                        cutoff = torch.topk(values, min(top_k, len(values))).values[-1]
                        values = values.masked_fill(values < cutoff, float("-inf"))
                    if not torch.isfinite(values).any():
                        continue
                    log_probs = F.log_softmax(values, dim=-1)
                    scores, ids = torch.topk(log_probs, min(width, int(torch.isfinite(log_probs).sum())))
                    for probability, token in zip(scores.tolist(), ids.tolist()):
                        extended = torch.cat((sequence, torch.tensor([[token]], device=sequence.device)), dim=1)
                        candidate_score = score + probability
                        stop = next((s for s in stops or () if len(generated) + 1 >= len(s)
                                     and extended[0, -len(s):].tolist() == list(s)), None)
                        if stop is not None:
                            # Count the end marker in scoring but remove it from text.
                            if len(generated) + 1 > len(stop):
                                normalized = candidate_score / ((len(generated) + 1) ** 0.7)
                                finished.append((normalized, extended[:, :-len(stop)]))
                        else:
                            candidates.append((extended, candidate_score))
                if not candidates:
                    active = []
                    break
                active = sorted(candidates, key=lambda item: item[1], reverse=True)[:width]
                finished = sorted(finished, key=lambda item: item[0], reverse=True)[:width]
            if finished:
                return max(finished, key=lambda item: item[0])[1]
            if active:
                return max(active, key=lambda item: item[1])[0]
            raise ValueError("No valid continuation under the generation constraints")
        finally:
            self.train(was_training)
