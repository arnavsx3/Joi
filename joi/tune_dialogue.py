"""Tune our word model on assistant replies in general and warm dialogues."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import random
import time

import torch

from joi.checkpoint import load_checkpoint, save_checkpoint
from joi.prepare_dialogue import curated_conversations
from joi.train import load_data
from joi.tune_recall import make_examples, batch, weighted_loss


def shorten(examples, context):
    # Retain the current assistant target and its nearest context. Long targets
    # are excluded so this tiny model learns short replies with context present.
    return [(x[-context:],y[-context:],w[-context:]) for x,y,w in examples
            if sum(label!=-100 for label in y)<=96]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-from",type=Path,default=Path("checkpoints/dialogue-words/best.pt"))
    parser.add_argument("--data",type=Path,default=Path("data/processed/dialogue-words"))
    parser.add_argument("--out",type=Path,default=Path("checkpoints/joi-chat"))
    parser.add_argument("--steps",type=int,default=1600)
    args=parser.parse_args()
    if args.steps<1 or (args.out.exists() and any(args.out.iterdir())):
        parser.error("Use positive steps and a new empty output directory")
    torch.set_num_threads(2)
    torch.manual_seed(2027)
    raw=args.init_from.read_bytes()
    model,tokenizer,parent=load_checkpoint(io.BytesIO(raw))
    data_tokenizer,_,hashes,manifest=load_data(args.data)
    if tokenizer.kind!="word" or tokenizer.chars!=data_tokenizer.chars:
        parser.error("Use our word-model checkpoint and its vocabulary")
    train_conversations=list(dict.fromkeys((args.data/"train.txt").read_text().strip().split("\n\n")))[:8000]
    validation=(args.data/"val.txt").read_text().strip().split("\n\n")
    train=shorten(make_examples(tokenizer,train_conversations),model.config.block_size)
    val=shorten(make_examples(tokenizer,validation),model.config.block_size)
    curated=curated_conversations(Path("data/curated/joi_dialogue_seeds.json"))
    warm=shorten(make_examples(tokenizer,curated["train"]),model.config.block_size)
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.0002,weight_decay=0.01)
    rng=random.Random(2027)
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
            loss=sum(weighted_loss(model,batch(val,16,random.Random(77+index),model.config.block_size)).item()
                     for index in range(4))/4
        rows.append({"step":step,"assistant_validation_loss":loss,
                     "elapsed_seconds":round(time.monotonic()-started,3)})
        state={"step":step,"optimizer_state":optimizer.state_dict(),"data_hashes":hashes,
               "dataset_manifest":manifest,"initialization":initialization,"metrics":rows,
               "objective":"Assistant reply tokens only; 75% general-dialogue batches and 25% curated warm-dialogue batches. Targets at most 96 tokens; retain nearest 256 input tokens.",
               "torch_version":str(torch.__version__),
               "training_examples":{"general":len(train),"warm":len(warm),"validation":len(val)}}
        save_checkpoint(args.out/"last.pt",model,tokenizer,**state)
        if loss<best:
            best=loss
            save_checkpoint(args.out/"best.pt",model,tokenizer,**state)
        if step==0:
            save_checkpoint(args.out/"initial.pt",model,tokenizer,**state)
        (args.out/"metrics.json").write_text(json.dumps(rows,indent=2)+"\n")
        print(f"step {step}: assistant validation loss {loss:.4f} | {rows[-1]['elapsed_seconds']:.1f}s",flush=True)
        model.train()
    record(0)
    for step in range(1,args.steps+1):
        examples=warm if rng.random()<0.25 else train
        optimizer.zero_grad(set_to_none=True)
        loss=weighted_loss(model,batch(examples,16,rng,model.config.block_size))
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite assistant loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
        optimizer.step()
        if step%200==0 or step==args.steps:
            record(step)
    print(f"Saved general assistant tuning to {args.out}",flush=True)


if __name__ == "__main__":
    main()
