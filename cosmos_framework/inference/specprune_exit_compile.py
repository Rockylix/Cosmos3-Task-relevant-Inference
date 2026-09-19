"""Opt-in decoder compile; online observation scoring remains the frozen eager algorithm."""

import torch


def compile_exit_layers(model, *, cuda_graphs):
    if model.net.pad_for_cuda_graphs:
        raise ValueError("SpecPrune requires actual, unpadded token sequences")
    return [
        torch.compile(layer, fullgraph=True, dynamic=True, mode="reduce-overhead" if cuda_graphs else "default")
        for layer in model.net.language_model.model.layers
    ]
