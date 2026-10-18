"""Inspect or clean an inactive managed routing deployment without touching providers."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from reef.core.errors import DeployConfigError
from reef.service.deploy.config_utils import load_config
from reef.service.deploy.orchestrator import resolve_deployment_config
from reef.service.deploy.routing_containers import ContainerCommandError, DockerComposeRuntime
from reef.service.deploy.routing_controller import RoutingDeploymentController
from reef.service.deploy.routing_registry import DeploymentRegistry, RoutingOwnershipError
from reef.service.deploy.service_config import service_config_from_mapping


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("status", "cleanup", "validate"))
    parser.add_argument("--deployment-id")
    parser.add_argument("--config", help="Candidate Reef configuration for read-only validate.")
    parser.add_argument("--timeout-seconds", type=float, default=10)
    parser.add_argument(
        "--acknowledge-unknown",
        metavar="OPERATION_ID",
        help="Only after independently verifying that prior Docker mutations have ceased.",
    )
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0 or args.timeout_seconds > 300:
        parser.error("timeout-seconds must be in (0, 300]")
    if args.acknowledge_unknown and args.action != "cleanup":
        parser.error("acknowledge-unknown is only available with cleanup")
    if args.action == "validate" and not args.config:
        parser.error("validate requires --config")
    if args.action != "validate" and (not args.deployment_id or args.config):
        parser.error("status and cleanup require --deployment-id, without --config")
    runtime = DockerComposeRuntime()
    try:
        if args.action == "validate":
            # The same resolution as ``reef serve``: a public configuration file, its environment and defaults.
            path = Path(args.config).expanduser().resolve()
            resolved, _ = resolve_deployment_config(load_config(path, interpolate_env=False), None, path)
            controller = RoutingDeploymentController(service_config_from_mapping(resolved), runtime)
            version = controller.validate_candidate()
            print(json.dumps({"valid": True, "snapshot_version": version, "docker_mutations": 0}))
            return
        registry = DeploymentRegistry(args.deployment_id)
        try:
            if args.action == "cleanup":
                registry.acquire()
            ownership = registry.read_owner()
            if ownership is None:
                print(json.dumps({"deployment_id": args.deployment_id, "state": "absent"}))
                return
            runtime.ensure_available()
            if args.action == "cleanup":
                # An unknown previous operation is retained even after observed
                # resources disappear. Only an explicit operation ID can clear it.
                try:
                    runtime.remove(ownership, args.timeout_seconds)
                except ContainerCommandError as exc:
                    if not args.acknowledge_unknown or not exc.outcome_unknown:
                        raise
                if args.acknowledge_unknown:
                    runtime.acknowledge_unknown(ownership, args.acknowledge_unknown)
            inventory = runtime.inspect(ownership)
            print(
                json.dumps(
                    {
                        "deployment_id": ownership.deployment_id,
                        "generation": ownership.generation,
                        "configuration_directory": str(ownership.configuration_directory),
                        "containers": [asdict(item) for item in inventory.containers],
                        "network_ids": inventory.network_ids,
                        "pending_operation_id": runtime.pending_operation_id(ownership),
                    },
                    sort_keys=True,
                )
            )
        finally:
            registry.release()
    except (RoutingOwnershipError, ContainerCommandError, DeployConfigError, OSError, ValueError) as exc:
        parser.exit(1, f"routing {args.action} failed: {exc}\n")


if __name__ == "__main__":
    main()
