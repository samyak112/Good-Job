"""
Export a trained GoodJob checkpoint to the Hugging Face Hub.

Edit the CONFIG block below, then run:  python export_to_hf.py

CHECKPOINT can be:
    - a directory saved with model.save_pretrained(...)
    - an old .pt / .pth file saved with torch.save(model.state_dict(), ...)

Run `hf auth login` (with a write token) once before uploading.
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import HfApi
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from model import MODEL, GoodJobConfig, GoodJobForSequenceClassification

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
CHECKPOINT = "runs/run/best.pt"       # save_pretrained directory, or an old .pt file
REPO = "spiderhomie/goodjob"       # Hub repo id
OUT_DIR = "hf_export"                # local folder for the exported files
PUBLIC = True                       # False = private repo
PUSH = True                          # False = export and verify only, no upload
# ----------------------------------------------------------------------------

def export_onnx(model, tok, out: Path):
    onnx_dir = out / "onnx"
    onnx_dir.mkdir(exist_ok=True)
    max_len = model.config.max_seq_length

    class Wrap(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask, token_type_ids):
            return self.m(input_ids=input_ids, attention_mask=attention_mask,
                          token_type_ids=token_type_ids).logits

    ex = tok(["hello world"], return_tensors="pt", padding="max_length",
             max_length=16, truncation=True)
    wrapper = Wrap(model).eval()

    torch.onnx.export(
        wrapper,
        (ex["input_ids"], ex["attention_mask"], ex["token_type_ids"]),
        str(onnx_dir / "model.onnx"),
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "token_type_ids": {0: "batch", 1: "seq"},
            "logits": {0: "batch"},
        },
        opset_version=17,
    )

    model.eval()

    cfg = model.config
    with open(onnx_dir / "labels.json", "w") as f:
        json.dump({"labels": [cfg.id2label[i] for i in range(cfg.num_labels)],
                   "max_len": max_len}, f)

    # parity check: compare logits (stricter than probabilities)
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_dir / "model.onnx"),
                                providers=["CPUExecutionProvider"])
    samples = [
        "We're hiring an ML engineer, DM me",
        "Looking for a job, open to work",
        "come join us!",
        "hi",
        "We are hiring a senior machine learning engineer to build ranking models "
        "for our search team. You will work with PyTorch and ship to production. " * 3,
    ]
    worst = 0.0
    for s in samples:
        for pad in (False, True):   # True = padded to 16 tokens, same shape as the export example
            if pad:
                e = tok([s], return_tensors="pt", truncation=True,
                        padding="max_length", max_length=16)
            else:
                e = tok([s], return_tensors="pt", truncation=True, max_length=max_len)
            with torch.inference_mode():
                a = model(**e).logits.numpy()
            b = sess.run(None, {k: e[k].numpy() for k in
                                ("input_ids", "attention_mask", "token_type_ids")})[0]
            d = float(np.abs(a - b).max())
            worst = max(worst, d)
            print(f"len={e['input_ids'].shape[1]:3d} pad={pad!s:5} "
                  f"maxdiff={d:.2e} | torch {np.round(a[0], 3)} | onnx {np.round(b[0], 3)}")
    print("worst logit diff:", worst)

def load_checkpoint(ckpt: Path) -> GoodJobForSequenceClassification:
    if ckpt.is_dir():
        return GoodJobForSequenceClassification.from_pretrained(ckpt)

    # legacy state_dict from the old GoodJob(nn.Module)
    sd = torch.load(ckpt, map_location="cpu")
    for key in ("model_state_dict", "state_dict", "model"):
        if isinstance(sd, dict) and key in sd:
            sd = sd[key]
            break

    # drop the removed board head's weights
    sd = {k: v for k, v in sd.items() if not k.startswith("board_head.")}

    model = GoodJobForSequenceClassification.from_base()
    missing, unexpected = model.load_state_dict(sd, strict=False)
    unexpected = [k for k in unexpected if not k.endswith("position_ids")]
    assert not missing and not unexpected, (
        f"state_dict key mismatch\n  missing: {missing}\n  unexpected: {unexpected}"
    )
    return model


def main():
    ckpt = Path(CHECKPOINT)
    out = Path(OUT_DIR)

    # 1. load trained weights into the Hub-compatible model
    model = load_checkpoint(ckpt)
    model.eval()

    tok_src = ckpt if (ckpt.is_dir() and (ckpt / "tokenizer_config.json").exists()) else MODEL
    tok = AutoTokenizer.from_pretrained(tok_src)

    # 2. save in Hub format (also copies modeling_goodjob.py next to the weights)
    GoodJobConfig.register_for_auto_class()
    GoodJobForSequenceClassification.register_for_auto_class("AutoModelForSequenceClassification")
    model.save_pretrained(out)
    tok.save_pretrained(out)
    print(f"saved to {out}/")

    # 3. reload exactly the way the other project will, and compare outputs
    reloaded = AutoModelForSequenceClassification.from_pretrained(
        out, trust_remote_code=True
    ).eval()

    samples = ["We're hiring an ML engineer, DM me", "Looking for a job, open to work", "come join us!"]
    x = tok(samples, return_tensors="pt", padding=True, truncation=True,
            max_length=model.config.max_seq_length)
    with torch.inference_mode():
        a, b = model(**x), reloaded(**x)
    assert torch.allclose(a.logits, b.logits, atol=1e-5), "type head mismatch after reload"
    print("reload parity OK")
    print(reloaded.predict(samples[0], tokenizer=tok))
    export_onnx(model, tok, out)

    # 4. upload the exact folder that was just tested
    if not PUSH:
        print("PUSH=False: nothing uploaded")
        return

    api = HfApi()
    api.create_repo(REPO, private=not PUBLIC, exist_ok=True)
    api.upload_folder(folder_path=str(out), repo_id=REPO, commit_message="Export GoodJob")
    print(f"pushed to https://huggingface.co/{REPO}")


if __name__ == "__main__":
    main()