# MIT License — Copyright (c) Facebook, Inc. and its affiliates.
# Vendored unchanged from https://github.com/nervjack2/MelHuBERT

import math
import torch
import torch.nn as nn


def gelu_accurate(x):
    if not hasattr(gelu_accurate, "_a"):
        gelu_accurate._a = math.sqrt(2 / math.pi)
    return (
        0.5 * x * (1 + torch.tanh(gelu_accurate._a * (x + 0.044715 * torch.pow(x, 3))))
    )


def gelu(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.gelu(x.float()).type_as(x)
