"""Run a pinned frontend with optional checksum-verified local ImageNet inputs."""

import argparse
import hashlib
import json
import runpy
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--imagenet-manifest", type=Path)
    args, compiler_args = parser.parse_known_args()
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "test"))
    if args.imagenet_manifest:
        from PIL import Image
        from utils.dataset import imagenet

        samples = json.loads(args.imagenet_manifest.read_text())["samples"]
        for sample in samples:
            path = Path(sample["path"])
            if (
                hashlib.sha256(path.read_bytes()).hexdigest()
                != sample["sha256"]
            ):
                raise ValueError(f"ImageNet fixture checksum mismatch: {path}")
        classes = sorted(
            p.name
            for p in Path(samples[0]["path"]).parents[1].iterdir()
            if p.is_dir()
        )

        def retrieve_dataset(num_samples, model_type):
            if num_samples > len(samples):
                raise ValueError(
                    "requested more ImageNet images than the pinned fixture"
                )
            transform = imagenet.get_transforms(model_type)
            result = []
            for sample in samples[:num_samples]:
                with Image.open(sample["path"]) as image:
                    result.append(
                        dict(
                            image=transform(image).unsqueeze(0),
                            label=classes.index(sample["label"]),
                        )
                    )
            print(
                f"Local ImageNet fixture: {num_samples} checksum-verified images in pinned order"
            )
            return result

        imagenet.retrieve_dataset = retrieve_dataset
    script = repo / "test/test_codegen.py"
    sys.argv = [str(script), *compiler_args]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
