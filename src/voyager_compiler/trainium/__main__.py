"""Convert an existing bufferized model.txt directory without re-exporting."""

import argparse
from pathlib import Path

from .converter import convert


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("collaterals", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--target",
        choices=("trainium-v2", "trainium-v3"),
        default="trainium-v2",
    )
    args = parser.parse_args()
    result = convert(args.collaterals, args.output, args.target)
    print(
        f"Converted {result['stats']['expanded_operations']} scheduled operations"
    )


if __name__ == "__main__":
    main()
