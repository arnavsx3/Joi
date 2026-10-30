"""Project-authored training conversations; never a runtime reply database."""

import json
from pathlib import Path

from joi.dialogue import build_prompt
from joi.tune_recall import supervised_example

SITUATIONS = Path(__file__).resolve().parent.parent / "data/curated/companion_situations.json"
MEMORY = (
    ("color", ["blue", "green", "red", "white", "black", "yellow", "orange", "pink", "brown"], ["purple", "gold"]),
    ("food", ["rice", "pizza", "pasta", "soup", "bread", "chicken", "fish", "salad"], ["noodles", "cake"]),
    ("drink", ["tea", "coffee", "water", "milk", "juice"], ["lemonade", "chocolate"]),
)


def records(split, path=SITUATIONS):
    """Hold out user phrasings, but share topics and reference responses.

    This is a development split, not evidence of generalization to new topics.
    All variants of an opener stay in the same split.
    """
    if split not in ("train", "validation"):
        raise ValueError("Use train or validation")
    result = []
    for situation in json.loads(path.read_text()):
        for opener in situation[split]:
            history = [{"role": "user", "content": opener}]
            for reply in situation["replies"]:
                # Familiar and neutral previous turns both occur during training.
                for prelude in ([], [{"role": "user", "content": "Hello!"},
                                     {"role": "assistant", "content": "Hi! What's on your mind?"}]):
                    result.append((situation["id"], prelude + history, reply))
                for followup in situation["followups"]:
                    messages = history + [{"role": "assistant", "content": reply},
                                          {"role": "user", "content": followup[split]}]
                    for continuation in followup["replies"]:
                        result.append((situation["id"], messages, continuation))
    for category, training, validation in MEMORY:
        for value in training if split == "train" else validation:
            # Values, rather than random expanded rows, define this task's split.
            for introduction in (f"My favorite {category} is {value}.", f"I like {value} best. It's my favorite {category}."):
                opener = [{"role": "user", "content": introduction}]
                greeting = f"Your favorite {category} is {value}. I'll keep that in mind while we chat."
                result.append((f"memory-{category}", opener, greeting))
                for prior in (greeting, "That's nice to know. Tell me more.", "Thanks for telling me."):
                    for question in (f"What is my favorite {category}?", f"Do you remember which {category} I like best?"):
                        history = opener + [{"role": "assistant", "content": prior},
                                            {"role": "user", "content": question}]
                        result.append((f"memory-{category}-{value}", history, f"Your favorite {category} is {value}."))
    return result


def prepare_examples(tokenizer, context, items):
    examples = []
    for topic, messages, target in items:
        prefix, _ = build_prompt(messages, context, tokenizer)
        sample = supervised_example(tokenizer, prefix, target)
        # Do not silently train a truncated target or a broken role prefix.
        if len(sample[0]) > context:
            raise ValueError(f"Example {topic} does not fit the context")
        examples.append((topic, sample))
    return examples
