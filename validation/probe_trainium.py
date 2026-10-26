"""Compile/replay a small GEMM and generate NKI without a device runtime."""

import sys
from pathlib import Path

import torch

from voyager_compiler.extensions import WorkloadSpec
from voyager_compiler.pipeline import compile_workload
from voyager_compiler.runtime import run
from voyager_compiler.targets.trainium.config import LAYOUT
from voyager_compiler.targets.trainium.target import TARGET

torch.set_num_threads(4)
torch.manual_seed(0)


class Model(torch.nn.Module):
    def forward(self, a, b):
        return torch.matmul(a, b)


args = (
    torch.randn(128, 128, dtype=torch.bfloat16),
    torch.randn(128, 128, dtype=torch.bfloat16),
)
spec = WorkloadSpec("gemm-smoke", lambda _: Model(), lambda m, o: (args, {}))
result = compile_workload(
    spec,
    TARGET,
    Path(sys.argv[1]),
    normalize_options=LAYOUT,
    dump_tensors=False,
)
outputs, evidence = run(result, args, reference=result.reference)
print(evidence)
print("NKI generated", (Path(sys.argv[1]) / "nki/kernels.py").is_file())
