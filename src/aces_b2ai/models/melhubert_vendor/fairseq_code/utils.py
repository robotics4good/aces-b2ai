# MIT License — Copyright (c) Facebook, Inc. and its affiliates.
# Vendored unchanged from https://github.com/nervjack2/MelHuBERT

import torch
import torch.nn.functional as F
from typing import Callable


def get_activation_fn(activation: str) -> Callable:
    from .gelu import gelu, gelu_accurate

    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return gelu
    elif activation == "gelu_accurate":
        return gelu_accurate
    elif activation == "tanh":
        return torch.tanh
    elif activation == "linear":
        return lambda x: x
    else:
        raise RuntimeError("--activation-fn {} not supported".format(activation))
