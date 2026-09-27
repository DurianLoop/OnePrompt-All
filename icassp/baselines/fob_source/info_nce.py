"""Minimal compatible InfoNCE implementation for the FoB release.

FoB constructs this loss object but its released training forward does not call
it.  The implementation is supplied to make the published source importable
without installing an unpinned third-party package; it follows the usual
cosine-similarity InfoNCE objective for future use.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class InfoNCE(nn.Module):
    def __init__(self, temperature: float = 0.1, negative_mode: str = "unpaired") -> None:
        super().__init__()
        if negative_mode not in {"unpaired", "paired"}:
            raise ValueError(f"unsupported negative_mode={negative_mode}")
        self.temperature = temperature
        self.negative_mode = negative_mode

    def forward(
        self,
        query: torch.Tensor,
        positive_key: torch.Tensor,
        negative_keys: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query = F.normalize(query, dim=-1)
        positive_key = F.normalize(positive_key, dim=-1)
        positive = (query * positive_key).sum(dim=-1, keepdim=True)
        if negative_keys is None:
            logits = query @ positive_key.transpose(-1, -2)
            labels = torch.arange(query.shape[0], device=query.device)
        else:
            negative_keys = F.normalize(negative_keys, dim=-1)
            if self.negative_mode == "unpaired":
                negative = query @ negative_keys.transpose(-1, -2)
            else:
                negative = torch.einsum("bd,bkd->bk", query, negative_keys)
            logits = torch.cat((positive, negative), dim=1)
            labels = torch.zeros(query.shape[0], dtype=torch.long, device=query.device)
        return F.cross_entropy(logits / self.temperature, labels)
