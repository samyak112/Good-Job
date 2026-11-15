import torch.nn as nn
from transformers import AutoModel

from config import BOARD_LABELS, TYPE_LABELS


class GoodJob(nn.Module):
    def __init__(self, name, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(name)
        h = self.encoder.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.type_head = nn.Linear(h, len(TYPE_LABELS))
        self.board_head = nn.Linear(h, len(BOARD_LABELS))

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kw = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kw["token_type_ids"] = token_type_ids
        cls = self.encoder(**kw).last_hidden_state[:, 0]  # [CLS]
        z = self.drop(cls)
        return self.type_head(z), self.board_head(z)