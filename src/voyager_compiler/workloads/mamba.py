"""Canonical unbatched selective SSM, retaining the recurrence as one op."""
import torch

from voyager_compiler.targets.trainium.scan import selective_scan


class SelectiveScan(torch.nn.Module):
    def forward(self, delta, u, a, b, c):
        return selective_scan(delta, u, a, b, c)
