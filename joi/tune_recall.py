"""Assistant-only tuning with extra loss weight on copying a supplied name."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import random
import re
import time

import torch
from torch.nn import functional as F

from joi.checkpoint import load_checkpoint, save_checkpoint
from joi.dialogue import format_conversation
from joi.prepare_dialogue import curated_conversations
from joi.train import load_data


def supervised_example(tokenizer, prefix, target, name=None):
    before=tokenizer.encode(prefix)
    after=tokenizer.encode(target)+tokenizer.encode("\n")
    ids=before+after
    labels=[-100]*len(before)+after
    weights=[0.0]*len(before)+[1.0]*len(after)
    if name:
        start=len(before)+len(tokenizer.encode(target[:target.index(name)]))
        for index in range(start,start+len(tokenizer.encode(name))):
            weights[index]=8.0
    return ids[:-1],labels[1:],weights[1:]


def make_examples(tokenizer, conversations, names_only=False):
    examples=[]
    for conversation in conversations:
        messages=[]
        lines=conversation.strip().splitlines()
        name_match=re.search(r"(?:You're |Your name is |You told me your name is )([A-Za-z]+)\.",lines[-1])
        if names_only:
            if not name_match:
                raise ValueError("Synthetic recall target is missing its name")
            name=name_match.group(1)
            users=[line[6:] for line in lines if line.startswith("User: ")]
            messages=[{"role":"user","content":users[0]},
                      {"role":"assistant","content":"Hey, you! I'm Joi. How has your day been?"},
                      {"role":"user","content":users[1]}]
            prefix=format_conversation(messages)+"Joi: "
            examples.append(supervised_example(tokenizer,prefix,f"Your name is {name}. What would you like to talk about?",name))
            continue
        for line in lines[1:]:
            if line.startswith("User: "):
                messages.append({"role":"user","content":line[6:]})
            elif line.startswith("Joi: "):
                target=line[5:]
                examples.append(supervised_example(tokenizer,format_conversation(messages)+"Joi: ",target))
                messages.append({"role":"assistant","content":target})
    return examples


def batch(examples, size, rng, context):
    selected=[rng.choice(examples) for _ in range(size)]
    length=max(len(item[0]) for item in selected)
    if length>context:
        raise ValueError("A supervised example exceeds the model's context")
    x=torch.zeros((size,length),dtype=torch.long)
    y=torch.full((size,length),-100,dtype=torch.long)
    w=torch.zeros((size,length))
    for index,(tokens,labels,weights) in enumerate(selected):
        x[index,:len(tokens)]=torch.tensor(tokens)
        y[index,:len(labels)]=torch.tensor(labels)
        w[index,:len(weights)]=torch.tensor(weights)
    return x,y,w


def weighted_loss(model, sample):
    x,y,weights=sample
    logits,_=model(x)
    losses=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),y.reshape(-1),reduction="none").view_as(weights)
    return (losses*weights).sum()/weights.sum()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-from",type=Path,default=Path("checkpoints/dialogue-words/best.pt"))
    parser.add_argument("--data",type=Path,default=Path("data/processed/persona"))
    parser.add_argument("--out",type=Path,default=Path("checkpoints/companion-recall"))
    parser.add_argument("--steps",type=int,default=1200)
    args=parser.parse_args()
    if args.steps<1:
        parser.error("--steps must be positive")
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("Use an empty output directory")
    torch.set_num_threads(2)
    torch.manual_seed(2026)
    raw=args.init_from.read_bytes()
    model,tokenizer,parent=load_checkpoint(io.BytesIO(raw))
    data_tokenizer,_,hashes,manifest=load_data(args.data)
    if tokenizer.kind!="word" or tokenizer.chars!=data_tokenizer.chars:
        parser.error("Use the original word-model checkpoint and matching vocabulary")
    counts=manifest["synthetic_names"]
    train_text=(args.data/"train.txt").read_text().strip().split("\n\n")
    val_text=(args.data/"val.txt").read_text().strip().split("\n\n")
    train=make_examples(tokenizer,train_text[-counts["training"]:],names_only=True)
    val=make_examples(tokenizer,val_text[-counts["validation"]:],names_only=True)
    curated=curated_conversations(Path("data/curated/joi_dialogue_seeds.json"))
    warm=make_examples(tokenizer,curated["train"])
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.0003,weight_decay=0.01)
    rng=random.Random(2026)
    args.out.mkdir(parents=True,exist_ok=True)
    rows=[]
    best=float("inf")
    started=time.monotonic()
    initialization={"mode":"own-checkpoint","source":str(args.init_from),
                    "source_sha256":hashlib.sha256(raw).hexdigest(),"source_step":parent["step"],
                    "source_initialization":parent.get("initialization")}
    def record(step):
        nonlocal best
        model.eval()
        with torch.inference_mode():
            loss=weighted_loss(model,batch(val,64,random.Random(77),model.config.block_size)).item()
        rows.append({"step":step,"weighted_validation_loss":loss,"elapsed_seconds":round(time.monotonic()-started,3)})
        state={"step":step,"optimizer_state":optimizer.state_dict(),"data_hashes":hashes,
               "dataset_manifest":manifest,"initialization":initialization,
               "objective":"Assistant-only target loss; name tokens weight 8; neutral reference greeting; 75% recall batches and 25% warm dialogue batches. This is not whole-corpus next-token loss.",
               "metrics":rows,"torch_version":str(torch.__version__)}
        save_checkpoint(args.out/"last.pt",model,tokenizer,**state)
        if loss<best:
            best=loss
            save_checkpoint(args.out/"best.pt",model,tokenizer,**state)
        if step==0:
            save_checkpoint(args.out/"initial.pt",model,tokenizer,**state)
        (args.out/"metrics.json").write_text(json.dumps(rows,indent=2)+"\n")
        print(f"step {step}: weighted assistant validation loss {loss:.4f} | {rows[-1]['elapsed_seconds']:.1f}s",flush=True)
        model.train()
    record(0)
    for step in range(1,args.steps+1):
        chosen=warm if rng.random()<0.25 else train
        optimizer.zero_grad(set_to_none=True)
        loss=weighted_loss(model,batch(chosen,16,rng,model.config.block_size))
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite weighted loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
        optimizer.step()
        if step%200==0 or step==args.steps:
            record(step)
    print(f"Saved supervised recall experiment in {args.out}",flush=True)


if __name__ == "__main__":
    main()
