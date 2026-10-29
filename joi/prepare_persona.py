"""Build a warm-dialogue and synthetic name-copying curriculum, using our vocabulary."""

import argparse
import json
from pathlib import Path
import random
import re

from joi.dialogue import format_conversation
from joi.prepare import digest, distinct_stories
from joi.prepare_dialogue import curated_conversations
from joi.tokenizer import tokenizer_from_state

RESERVED_NAMES = {"arnav", "maya", "sam", "neel"}


def make_names(count, *, excluded=(), seed=314):
    rng = random.Random(seed)
    names, seen = [], set(RESERVED_NAMES) | {name.lower() for name in excluded}
    while len(names) < count:
        pattern = rng.choice(("cvcv","cvvc","cvcvc","cvccv","cvccvc","cvcvcv"))
        name = "".join(rng.choice("aeiou" if letter=="v" else "bcdfghjklmnprstvz")
                       for letter in pattern).capitalize()
        if name.lower() not in seen:
            seen.add(name.lower())
            names.append(name)
    return names


def recall_conversations(names, seed):
    rng = random.Random(seed)
    result = []
    for name in names:
        opener = rng.choice((f"My name is {name}.",f"You can call me {name}.",f"I'm {name}, by the way."))
        greeting = rng.choice(("Hey, you! I'm Joi. How has your day been?",
                               "Hi! I'm glad you're here. What's on your mind?",
                               f"Hi, {name}! I'm Joi. How are you feeling today?"))
        question = rng.choice(("What is my name?","Do you remember my name?","What did I say my name was?"))
        answer = rng.choice((f"You're {name}. What would you like to talk about?",
                              f"Your name is {name}. How has your day been?",
                              f"You told me your name is {name}. What's on your mind?"))
        messages = [{"role":"user" if index%2==0 else "assistant","content":text}
                    for index,text in enumerate((opener,greeting,question,answer))]
        result.append(format_conversation(messages))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=Path("data/processed/dialogue-words"))
    parser.add_argument("--output",type=Path,default=Path("data/processed/persona"))
    parser.add_argument("--curated",type=Path,default=Path("data/curated/joi_dialogue_seeds.json"))
    parser.add_argument("--train-names",type=int,default=1200)
    parser.add_argument("--val-names",type=int,default=200)
    args = parser.parse_args()
    if min(args.train_names,args.val_names) < 1:
        parser.error("Name counts must be positive")
    tokenizer_state = json.loads((args.source/"tokenizer.json").read_text())
    tokenizer = tokenizer_from_state(tokenizer_state)
    train_text = (args.source/"train.txt").read_text()
    val_text = (args.source/"val.txt").read_text()
    source_words = {word.lower() for word in re.findall(r"[A-Za-z]+",train_text+"\n"+val_text)}
    names = make_names(args.train_names+args.val_names)
    # Novel generated names stay outside the learned whole-word vocabulary.
    names = [name for name in names if name not in tokenizer.stoi and name.lower() not in source_words]
    while len(names) < args.train_names+args.val_names:
        names.extend(name for name in make_names(20,excluded=names,seed=314+len(names))
                     if name not in tokenizer.stoi and name.lower() not in source_words)
    train_names = names[:args.train_names]
    val_names = names[args.train_names:args.train_names+args.val_names]
    curated = curated_conversations(args.curated)
    general = train_text.strip().split("\n\n")
    random.Random(2718).shuffle(general)
    general = list(dict.fromkeys(general))[:1000]
    train = [item.strip() for item in general + curated["train"]*20 + recall_conversations(train_names,21)]
    val = [item.strip() for item in val_text.strip().split("\n\n") + recall_conversations(val_names,22)]
    _, keys = distinct_stories(train)
    val, _ = distinct_stories(val,keys)
    manifest = {
        "dataset":"DailyDialog replay + original Joi persona + synthetic name curriculum",
        "format":"joi-dialogue-v1", "revision":"persona-curriculum-v1",
        "parent_manifest_sha256":digest((args.source/"manifest.json").read_bytes()),
        "curated_sha256":digest(args.curated.read_bytes()),
        "license":"DailyDialog replay: CC BY-NC-SA 4.0; other examples: project-authored",
        "tokenizer":{"type":tokenizer.kind,"fit":"unchanged from parent training vocabulary"},
        "vocab_size":len(tokenizer.chars),"splits":{},
        "synthetic_names":{"training":len(train_names),"validation":len(val_names),
                           "reserved_development_names":sorted(RESERVED_NAMES),
                           "note":"Artificial letter combinations, not personal data. Train/validation name sets are disjoint. Reserved probe names are excluded from generated curriculum; some already occur in curated source examples."},
        "selection":"1000 unique replay conversations, curated training dialogues repeated 20 times, generated four-turn name conversations. Original validation plus disjoint generated validation names. No test split used.",
    }
    args.output.mkdir(parents=True,exist_ok=True)
    for split,conversations in (("train",train),("val",val)):
        text = "\n\n".join(conversation.strip() for conversation in conversations)+"\n"
        tokens=tokenizer.encode(text)
        raw=text.encode()
        (args.output/f"{split}.txt").write_bytes(raw)
        manifest["splits"][split]={"sha256":digest(raw),"characters":len(text),"tokens":len(tokens),
                                  "unique_conversations":len(set(conversations)),
                                  "weighted_conversations":len(conversations)}
    (args.output/"tokenizer.json").write_text(json.dumps(tokenizer_state)+"\n")
    (args.output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(manifest,indent=2))


if __name__ == "__main__":
    main()
