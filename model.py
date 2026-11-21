"""
GoodJob: one BERT encoder with two heads on the [CLS] token.

  type_head  -> `logits`        (primary head: ML Job / Job / Other / Not Job)
  board_head -> `board_logits`  (Regular account / Job board)

Fully self-contained: depends only on torch + transformers. This one file is
used for training, local inference, and is uploaded as-is to the Hugging Face Hub.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoTokenizer, BertConfig, BertModel, BertPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

# ---- hardcoded settings (change here only) --------------------------------
MODEL = "bert-base-uncased"
TYPE_LABELS = ["ML Job", "Job", "Other", "Not Job"]
BOARD_LABELS = ["Regular account", "Job board"]
DROPOUT = 0.1
MAX_LEN = 256  # must match the max_length used in training
# ---------------------------------------------------------------------------


class GoodJobConfig(BertConfig):
    model_type = "goodjob"

    def __init__(self, board_labels=None, max_seq_length=MAX_LEN, **kwargs):
        kwargs.setdefault("id2label", dict(enumerate(TYPE_LABELS)))
        kwargs.setdefault("label2id", {label: i for i, label in enumerate(TYPE_LABELS)})
        super().__init__(**kwargs)
        self.board_labels = list(board_labels) if board_labels else list(BOARD_LABELS)
        self.max_seq_length = max_seq_length


@dataclass
class GoodJobOutput(SequenceClassifierOutput):
    """`logits` is the type head, so anything expecting a standard classifier works."""
    board_logits: Optional[torch.Tensor] = None


class GoodJobForSequenceClassification(BertPreTrainedModel):
    config_class = GoodJobConfig
    base_model_prefix = "encoder"

    def __init__(self, config):
        super().__init__(config)
        # Same attribute names as the original GoodJob, so old state_dict
        # checkpoints load without remapping.
        self.encoder = BertModel(config)
        h = config.hidden_size
        self.drop = nn.Dropout(DROPOUT)
        self.type_head = nn.Linear(h, config.num_labels)
        self.board_head = nn.Linear(h, len(config.board_labels))
        self._tokenizer = None
        self.post_init()

    # ------------------------------------------------------------------
    # Construction for training
    # ------------------------------------------------------------------
    @classmethod
    def from_base(cls):
        """Fresh model: pretrained bert-base-uncased encoder, new random heads."""
        d = BertConfig.from_pretrained(MODEL).to_dict()
        for k in ("model_type", "id2label", "label2id", "num_labels",
                  "architectures", "_name_or_path", "transformers_version"):
            d.pop(k, None)
        model = cls(GoodJobConfig(**d))
        model.encoder.load_state_dict(BertModel.from_pretrained(MODEL).state_dict())
        return model

    # ------------------------------------------------------------------
    # Forward (same logic as the original GoodJob)
    # ------------------------------------------------------------------
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, **kwargs):
        kw = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kw["token_type_ids"] = token_type_ids
        cls = self.encoder(**kw).last_hidden_state[:, 0]  # [CLS]
        z = self.drop(cls)
        return GoodJobOutput(logits=self.type_head(z), board_logits=self.board_head(z))

    # ------------------------------------------------------------------
    # Inference: single or batch, CPU or GPU
    # ------------------------------------------------------------------
    def _get_tokenizer(self, tokenizer=None):
        if tokenizer is not None:
            return tokenizer
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.config._name_or_path)
        return self._tokenizer

    @torch.inference_mode()
    def predict(self, texts, tokenizer=None, device=None, batch_size=32,
                max_length=None, fp16=False):
        """
        texts:      str -> returns one dict; list[str] -> returns list[dict] (input order kept)
        tokenizer:  optional; by default it is loaded from the same repo/folder as the model
        device:     None (keep model where it is), "cpu", "cuda", "cuda:1", or "auto"
        batch_size: texts per forward pass
        max_length: defaults to MAX_LEN
        fp16:       half-precision autocast, CUDA only (ignored on CPU)
        """
        single = isinstance(texts, str)
        if single:
            texts = [texts]
        if not texts:
            return []

        tok = self._get_tokenizer(tokenizer)
        max_length = max_length or self.config.max_seq_length

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device is not None:
            self.to(device)
        self.eval()
        dev = self.device

        not_job = self.config.label2id["Not Job"]
        id2label = self.config.id2label
        use_amp = fp16 and dev.type == "cuda"

        # Sort by length to minimise padding, then restore the original order.
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        results = [None] * len(texts)

        for s in range(0, len(order), batch_size):
            idx = order[s:s + batch_size]
            enc = tok([texts[i] for i in idx], return_tensors="pt",
                      truncation=True, padding=True, max_length=max_length)
            enc = {k: v.to(dev) for k, v in enc.items()}

            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
                out = self(**enc)

            type_p = out.logits.float().softmax(-1).cpu()
            board_p = out.board_logits.float().softmax(-1).cpu()

            for i, tp, bp in zip(idx, type_p, board_p):
                t = int(tp.argmax())
                is_job = t != not_job
                results[i] = {
                    "type": id2label[t],
                    "type_probs": {id2label[j]: float(p) for j, p in enumerate(tp)},
                    "is_job": is_job,
                    "p_job": float(1 - tp[not_job]),
                    # the board head is only trained on job rows, so only report it for jobs
                    "is_job_board": bool(bp.argmax()) if is_job else None,
                    "board_p": float(bp[1]) if is_job else None,
                }

        return results[0] if single else results