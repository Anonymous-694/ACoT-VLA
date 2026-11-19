"""Serve a complete four-track simulation checkpoint without training datasets."""

import argparse
import json
import logging
from pathlib import Path

from openpi.four_tracks import checkpoint_bundle
from openpi.four_tracks.policies import policy_config
from openpi.four_tracks.serving import task_routing
from openpi.four_tracks.serving.websocket_policy_server import WebsocketPolicyServer
from openpi.four_tracks.shared import robot_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--validate-only", action="store_true", help="check assets without loading model parameters")
    args = parser.parse_args()
    spec = checkpoint_bundle.inspect(args.checkpoint)
    if args.validate_only:
        print(json.dumps({"valid": True, "groups": sorted(spec["norm_stats"]), "layout": spec["layout"]}))
        return
    assets = Path(args.checkpoint).resolve() / "assets"
    robot_config.load_registry(assets / spec["robot_config"])
    routes = task_routing.load(assets / spec["task_routes"])
    policy = policy_config.create_trained_policy(None, args.checkpoint)
    WebsocketPolicyServer(
        policy,
        host=args.host,
        port=args.port,
        prepend_robot_type=spec["data"]["prepend_robot_type"],
        arm_canonical_per_arm_dim=7,
        scenario_routing=True,
        unified_layout=True,
        prefix_from_norm_group=spec["data"]["prepend_robot_type_from_norm_group"],
        task_routes=routes,
        waist_groups=tuple(spec["waist_groups"]),
    ).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
