"""mmBERT + 5-d clause features for tsawa token classification."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput


class TsawaFeatConfig(PretrainedConfig):
    model_type = "mmbert_tsawa_feat"

    def __init__(
        self,
        base_model_name: str = "jhu-clsp/mmBERT-base",
        num_labels: int = 3,
        hidden_size: int = 768,
        feat_dim: int = 5,
        feat_hidden: int = 32,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_model_name = base_model_name
        self.num_labels = num_labels
        self.hidden_size = hidden_size
        self.feat_dim = feat_dim
        self.feat_hidden = feat_hidden


class TsawaFeatModel(PreTrainedModel):
    config_class = TsawaFeatConfig

    def __init__(self, config: TsawaFeatConfig):
        super().__init__(config)
        enc_cfg = AutoConfig.from_pretrained(config.base_model_name)
        self.encoder = AutoModel.from_config(enc_cfg)
        self.feat_proj = nn.Sequential(
            nn.Linear(config.feat_dim, config.feat_hidden),
            nn.GELU(),
        )
        self.classifier = nn.Linear(enc_cfg.hidden_size + config.feat_hidden, config.num_labels)
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, features=None, labels=None, **kwargs):
        hidden = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        if features is None:
            features = hidden.new_zeros(hidden.size(0), hidden.size(1), self.config.feat_dim)
        features = features.to(device=hidden.device, dtype=hidden.dtype)
        fused = torch.cat([hidden, self.feat_proj(features)], dim=-1)
        logits = self.classifier(fused)
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss(ignore_index=-100)(
                logits.view(-1, self.config.num_labels), labels.view(-1)
            )
        return TokenClassifierOutput(loss=loss, logits=logits)


def build_tsawa_feat_model(base_model: str, num_labels: int, id2label: dict, label2id: dict):
    enc_cfg = AutoConfig.from_pretrained(base_model)
    cfg = TsawaFeatConfig(
        base_model_name=base_model,
        num_labels=num_labels,
        hidden_size=enc_cfg.hidden_size,
        id2label=id2label,
        label2id=label2id,
    )
    model = TsawaFeatModel(cfg)
    model.encoder = AutoModel.from_pretrained(base_model)
    return model
