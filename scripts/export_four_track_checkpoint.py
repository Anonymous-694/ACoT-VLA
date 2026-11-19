"""Export parameters and portable assets without training datasets or optimizer state."""

import argparse

from openpi.four_tracks import checkpoint_bundle
from openpi.four_tracks.training.four_tracks import config_from_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", help="required for older checkpoints without portable assets")
    parser.add_argument(
        "--license-file", action="append", default=[], help="UTF-8 license/notice text to include; repeatable"
    )
    args = parser.parse_args()
    config = config_from_manifest(args.manifest, validate_datasets=False) if args.manifest else None
    dc = config.data.create(config.assets_dirs, config.model) if config else None
    print(
        checkpoint_bundle.export(
            args.source, args.output, config=config, data_config=dc, license_files=args.license_file
        )
    )


if __name__ == "__main__":
    main()
