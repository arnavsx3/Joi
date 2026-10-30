"""Test generated replies on names held out from the synthetic curriculum."""

import argparse
import hashlib
import json
from pathlib import Path
import re

from joi.chat import ChatEngine, CompanionEngine
from joi.companion_data import SITUATIONS, records as companion_records


def recall_trial(engine, conversation, seed):
    lines=conversation.strip().splitlines()
    users=[line[6:] for line in lines if line.startswith("User: ")]
    if len(users)!=2:
        raise ValueError("A recall example needs two user turns")
    target=lines[-1]
    found=re.search(r"(?:You're |Your name is |You told me your name is )([A-Za-z]+)\.",target)
    if not found:
        raise ValueError("Recall target did not contain a labeled name")
    name=found.group(1)
    history=[]
    replies=[]
    for index,text in enumerate(users):
        history.append({"role":"user","content":text})
        try:
            result=engine.reply(history,seed=seed+index)
        except ValueError as exc:
            return {"name":name,"users":users,"replies":replies,"error":str(exc),"correct":False}
        replies.append(result["reply"])
        history.append({"role":"assistant","content":result["reply"]})
    correct=bool(re.search(r"\b"+re.escape(name)+r"\b",replies[-1],flags=re.I))
    return {"name":name,"users":users,"replies":replies,"correct":correct}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",type=Path,default=Path("checkpoints/companion/best.pt"))
    parser.add_argument("--data",type=Path,default=Path("data/processed/persona"))
    parser.add_argument("--out",type=Path,default=Path("runs/heldout-recall.json"))
    parser.add_argument("--limit",type=int,default=20)
    parser.add_argument("--with-recall-specialist",action="store_true")
    args=parser.parse_args()
    if args.limit<1:
        parser.error("--limit must be positive")
    engine=(CompanionEngine(args.checkpoint,tokens=48) if args.with_recall_specialist
            else ChatEngine(args.checkpoint,tokens=48))
    manifest=json.loads((args.data/"manifest.json").read_text())
    count=manifest["synthetic_names"]["validation"]
    records=(args.data/"val.txt").read_text().strip().split("\n\n")[-count:][:args.limit]
    train=(args.data/"train.txt").read_text()
    general=engine.general if args.with_recall_specialist else engine
    if general.checkpoint.get("curriculum_sha256"):
        if general.checkpoint["curriculum_sha256"] != hashlib.sha256(SITUATIONS.read_bytes()).hexdigest():
            parser.error("Restore the checkpoint's exact authored curriculum before checking held-out names")
        train += "\n" + "\n".join(" ".join(message["content"] for message in history) + " " + target
                                   for _, history, target in companion_records("train"))
    trials=[recall_trial(engine,conversation,1000+index*10) for index,conversation in enumerate(records)]
    for trial in trials:
        if re.search(r"\b"+re.escape(trial["name"])+r"\b",train,flags=re.I):
            raise ValueError("A purported held-out name occurs in training text")
    correct=sum(trial["correct"] for trial in trials)
    report={"checkpoint":str(args.checkpoint),"checkpoint_sha256":engine.checkpoint_sha256,
            "runtime":engine.status(),
            "phase_step":engine.checkpoint["step"],"correct":correct,"total":len(trials),
            "literal_recall_fraction":correct/len(trials),
            "protocol":{"tokens_per_reply":general.tokens,"temperature":general.temperature,"top_k":general.top_k,
                        "beams":general.beams,
                        "no_repeat_ngram_size":engine.no_repeat_ngram_size,
                        "seed":"1000 + example_index*10 + turn_index"},
            "note":"Held-out synthetic names with familiar question templates. Both replies are generated; reference assistant turns are never given to the model. Exact name occurrence does not establish unrestricted conversational memory.",
            "trials":trials}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2)+"\n")
    print(f"Held-out literal name recall: {correct}/{len(trials)}")
    print(f"Saved generated trials to {args.out}")


if __name__ == "__main__":
    main()
