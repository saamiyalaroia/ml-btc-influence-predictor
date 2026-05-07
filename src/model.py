"""
BERT + two-layer MLP classifier.

Architecture (as specified in the proposal):
  BERT (bert-base-uncased)
    -> [CLS] token embedding  (768-dim)
    -> Linear(768, 256) + ReLU + Dropout
    -> Linear(256, 3)
    -> (Softmax applied externally for probabilities)

Labels: 0 = down  |  1 = flat  |  2 = up
"""

import torch
import torch.nn as nn
from transformers import BertModel

from config import BERT_MODEL


class BertMLP(nn.Module):
    """
    BERT encoder followed by a small two-layer MLP head.

    Parameters
    ----------
    num_classes : int   Number of output classes (3: down / flat / up).
    dropout     : float Dropout probability between the two MLP layers.
    """

    def __init__(self, num_classes: int = 3, dropout: float = 0.3):
        super().__init__()

        self.bert = BertModel.from_pretrained(BERT_MODEL)
        hidden_size = self.bert.config.hidden_size  # 768 for bert-base

        # Two-layer fully-connected classification head.
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Returns
        -------
        logits : (batch, num_classes) — raw scores before softmax.
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        # Use the [CLS] token representation as the sentence embedding.
        cls_emb = outputs.last_hidden_state[:, 0, :]  # (batch, 768)
        logits  = self.mlp(cls_emb)                   # (batch, num_classes)
        return logits

    def predict_proba(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Return softmax probabilities.

        Returns
        -------
        probs : (batch, num_classes)  values sum to 1 along dim=1.
        """
        logits = self.forward(input_ids, attention_mask, token_type_ids)
        return torch.softmax(logits, dim=-1)
