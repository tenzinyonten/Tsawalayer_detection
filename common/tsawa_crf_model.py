"""mmBERT + linear-chain CRF for tsawa token classification."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput


class TsawaCRFConfig(PretrainedConfig):
    model_type = "mmbert_tsawa_crf"

    def __init__(
        self,
        base_model_name: str = "jhu-clsp/mmBERT-base",
        num_labels: int = 3,
        hidden_size: int = 768,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_model_name = base_model_name
        self.num_labels = num_labels
        self.hidden_size = hidden_size


class LinearChainCRF(nn.Module):
    def __init__(self, num_tags: int):
        super().__init__()
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags) * 0.1)
        self.start_transitions = nn.Parameter(torch.randn(num_tags) * 0.1)
        self.end_transitions = nn.Parameter(torch.randn(num_tags) * 0.1)

    def _gold_score(self, emissions, tags, mask):
        # emissions: [B, T, C], tags: [B, T], mask: [B, T] bool
        # Valid tokens are one contiguous run (content between CLS/SEP/pad).
        B, T, C = emissions.shape
        idx = torch.arange(B, device=emissions.device)
        emit = emissions.gather(-1, tags.unsqueeze(-1)).squeeze(-1)
        score = (emit * mask).sum(-1)
        first = mask.long().argmax(-1)
        last = T - 1 - mask.flip(-1).long().argmax(-1)
        has = mask.any(-1)
        score = score + torch.where(
            has, self.start_transitions[tags[idx, first]], emit.new_zeros(B)
        )
        score = score + torch.where(
            has, self.end_transitions[tags[idx, last]], emit.new_zeros(B)
        )
        if T > 1:
            pair = mask[:, 1:] & mask[:, :-1]
            score = score + (self.transitions[tags[:, :-1], tags[:, 1:]] * pair).sum(-1)
        return score

    def _log_partition(self, emissions, mask):
        B, T, C = emissions.shape
        neg = emissions.new_full((B, C), -1.0e9)
        alpha = neg
        started = mask.new_zeros((B,), dtype=torch.bool)
        for t in range(T):
            start_t = self.start_transitions + emissions[:, t]
            cont_t = torch.logsumexp(
                alpha.unsqueeze(2) + self.transitions + emissions[:, t].unsqueeze(1),
                dim=1,
            )
            nxt = torch.where(started.unsqueeze(-1), cont_t, start_t)
            alpha = torch.where(mask[:, t].unsqueeze(-1), nxt, alpha)
            started = started | mask[:, t]
        return torch.logsumexp(alpha + self.end_transitions, dim=-1)

    def nll(self, emissions, tags, mask):
        gold = self._gold_score(emissions, tags, mask)
        z = self._log_partition(emissions, mask)
        return (z - gold).mean()


class TsawaCRFModel(PreTrainedModel):
    config_class = TsawaCRFConfig

    def __init__(self, config: TsawaCRFConfig):
        super().__init__(config)
        enc_cfg = AutoConfig.from_pretrained(config.base_model_name)
        self.encoder = AutoModel.from_config(enc_cfg)
        self.classifier = nn.Linear(enc_cfg.hidden_size, config.num_labels)
        self.crf = LinearChainCRF(config.num_labels)
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        hidden = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        logits = self.classifier(hidden)
        loss = None
        if labels is not None:
            mask = attention_mask.bool() if attention_mask is not None else torch.ones_like(labels, dtype=torch.bool)
            mask = mask & (labels != -100)
            tags = labels.clamp(min=0)
            loss = self.crf.nll(logits, tags, mask)
        return TokenClassifierOutput(loss=loss, logits=logits)


def build_tsawa_crf_model(base_model: str, num_labels: int, id2label: dict, label2id: dict):
    enc_cfg = AutoConfig.from_pretrained(base_model)
    cfg = TsawaCRFConfig(
        base_model_name=base_model,
        num_labels=num_labels,
        hidden_size=enc_cfg.hidden_size,
        id2label=id2label,
        label2id=label2id,
    )
    model = TsawaCRFModel(cfg)
    model.encoder = AutoModel.from_pretrained(base_model)
    return model
