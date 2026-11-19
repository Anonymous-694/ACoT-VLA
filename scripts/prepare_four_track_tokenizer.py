"""Download the public tokenizer once before training/export."""

import argparse
from pathlib import Path
import shutil

from openpi.shared import download

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.resolve() != source.resolve():
        shutil.copyfile(source, target)
    print(target)
