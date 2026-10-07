"""Isolate the pinned SDK's native-transpose reuse failure on saved collaterals.

The input directory must contain the original failing explicit-transpose source
and reference.npz. This is a correctness probe only; production conversion uses
the verified transpose realization. Run using the NKI environment.
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import numpy as np
import neuronxcc.nki as nki


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    args = parser.parse_args()
    root = args.case.resolve()
    os.environ["PATH"] = (
        "/opt/aws_neuronx_venv_pytorch_2_9/bin:/opt/aws/neuron/bin:"
        + os.environ["PATH"]
    )
    source = (root / "nki/program.py").read_text()
    if "nisa.nc_transpose" not in source:
        raise SystemExit(
            "Requires the retained failing native-transpose program"
        )
    variants = {
        "generic_matmul": re.sub(
            r"nisa.nc_matmul\((t\d+), (t\d+)\)",
            r"nl.matmul(\1, \2, transpose_x=True)",
            source,
        ),
        "generic_copy": re.sub(
            r"nisa.tensor_copy\((t\d+), engine=nisa.vector_engine\)",
            r"nl.copy(\1)",
            source,
        ),
        "generic_transpose": re.sub(
            r"nisa.nc_transpose\((t\d+), engine=nisa.tensor_engine\)",
            r"nl.transpose(\1)",
            source,
        ),
    }
    rows = {}
    with np.load(root / "reference.npz") as data:
        for name, code in variants.items():
            path = root / (name + ".py")
            path.write_text(code)
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            output = nki.baremetal(
                module.kernel, additional_compile_opt="--target=trn2 --lnc=1"
            )(data["a0"], data["a1"])
            if isinstance(output, (tuple, list)):
                output = output[0]
            error = float(np.max(np.abs(output - data["expected"])))
            rows[name] = dict(
                source_sha256=hashlib.sha256(code.encode()).hexdigest(),
                max_abs_error=error,
                correct=bool(
                    np.allclose(output, data["expected"], atol=5e-4, rtol=5e-4)
                ),
            )
            print(name, rows[name], flush=True)
    (root / "isa-isolation.json").write_text(
        json.dumps(
            dict(
                scope="Fresh device correctness probes, identical inputs and software schedule; no timing claims",
                variants=rows,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
