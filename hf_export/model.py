"""
GoodJob: one BERT encoder with a job-type head on the [CLS] token.

  classifier -> `logits`  (ML Job / Job / Other / Not Job)

Fully self-contained: depends only on torch + transformers.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, BertConfig, BertModel, BertPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

# ---- hardcoded settings ----------------------------------------------------
MODEL = "bert-base-uncased"
TYPE_LABELS = ["ML JOB", "JOB", "OTHER", "NOT JOB"]
DROPOUT = 0.1
MAX_LEN = 256
# ---------------------------------------------------------------------------


class GoodJobConfig(BertConfig):
    model_type = "goodjob"

    def __init__(self, max_seq_length=MAX_LEN, **kwargs):
        kwargs.setdefault("id2label", dict(enumerate(TYPE_LABELS)))
        kwargs.setdefault("label2id", {label: i for i, label in enumerate(TYPE_LABELS)})
        kwargs.pop("board_labels", None)  # old configs may still carry this
        super().__init__(**kwargs)
        self.max_seq_length = max_seq_length


class GoodJobForSequenceClassification(BertPreTrainedModel):
    config_class = GoodJobConfig
    base_model_prefix = "encoder"
    # old checkpoints contain board_head.*; don't warn about them
    _keys_to_ignore_on_load_unexpected = [r"board_head\..*"]

    def __init__(self, config):
        super().__init__(config)
        self.encoder = BertModel(config)
        self.drop = nn.Dropout(DROPOUT)
        self.type_head = nn.Linear(config.hidden_size, config.num_labels)
        self._tokenizer = None
        self.post_init()

    @classmethod
    def from_base(cls):
        """Fresh model: pretrained bert-base-uncased encoder, new random head."""
        d = BertConfig.from_pretrained(MODEL).to_dict()
        for k in ("model_type", "id2label", "label2id", "num_labels",
                  "architectures", "_name_or_path", "transformers_version"):
            d.pop(k, None)
        model = cls(GoodJobConfig(**d))
        model.encoder.load_state_dict(BertModel.from_pretrained(MODEL).state_dict())
        return model

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, **kwargs):
        kw = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kw["token_type_ids"] = token_type_ids
        cls = self.encoder(**kw).last_hidden_state[:, 0]  # [CLS]
        return SequenceClassifierOutput(logits=self.type_head(self.drop(cls)))

    def _get_tokenizer(self, tokenizer=None):
        if tokenizer is not None:
            return tokenizer
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.config._name_or_path)
        return self._tokenizer

    @torch.inference_mode()
    def predict(self, texts, tokenizer=None, device=None, batch_size=32,
                max_length=None, fp16=False):
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

        not_job = self.config.label2id["NOT JOB"]
        id2label = self.config.id2label
        use_amp = fp16 and dev.type == "cuda"

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

            for i, tp in zip(idx, type_p):
                t = int(tp.argmax())
                results[i] = {
                    "type": id2label[t],
                    "type_probs": {id2label[j]: float(p) for j, p in enumerate(tp)},
                    "is_job": t != not_job,
                    "p_job": float(1 - tp[not_job]),
                }

        return results[0] if single else results