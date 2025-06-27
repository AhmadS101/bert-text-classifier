import os
import torch
import torch.nn as nn
from transformers import AutoModel
import constants


class BERTClassifier(nn.Module):
    def __init__(self, model_name=None, num_classes=None, dropout_rate=None):
        super(BERTClassifier, self).__init__()
        model_name = constants.MODEL_NAME
        dropout_rate = constants.DROPOUT_RATE

        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

        if constants.FREEZE_LAYERS:
            self.freeze_layers()

    def freeze_layers(self):
        if constants.FREEZE_EMBEDDINGS:
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False

        if constants.FREEZE_EARLY_LAYERS > 0:
            for layer in self.bert.encoder.layer[: constants.FREEZE_EARLY_LAYERS]:
                for param in layer.parameters():
                    param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

    def get_model_info(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "frozen_params": total_params - trainable_params,
        }
