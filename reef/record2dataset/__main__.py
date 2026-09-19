"""Run the generator service from the deployment ``reef serve`` resolved, or from a deployment file."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from aiohttp import web

from reef.record2dataset.designer import ReefDesigner
from reef.record2dataset.harbor import HarborRuns
from reef.record2dataset.service import GeneratorService, HarborChecks, JobRunner, ReefTaskPlays, readiness_probes
from reef.service.deploy.config_utils import DeployConfigError, load_config
from reef.service.deploy.generator import GeneratorSettings, generator_settings
from reef.service.deploy.inference import local_service_url
from reef.service.deploy.service_config import ServiceConfig, service_config_from_mapping


def generator_service(settings: ServiceConfig, generator: GeneratorSettings) -> GeneratorService:
    """The service the deployment describes: the designer through Reef, Harbor's checks, the task player."""
    reef_url = local_service_url(settings)
    token = settings.tokens[0] if settings.tokens else None
    tasks_root = Path(generator.tasks_root).expanduser()
    work_dir = Path(generator.work_dir).expanduser() if generator.work_dir else tasks_root / ".play"
    designer = ReefDesigner(
        reef_url=generator.designer_url or reef_url,
        token=generator.designer_token if generator.designer_url else token,
        request_options=generator.designer_options,
        timeout_s=generator.designer_timeout_s,
    )
    plays = ReefTaskPlays(
        reef_url=reef_url,
        work_dir=work_dir,
        token=token,
        agent=generator.agent,
        agent_host=generator.agent_host,
        concurrency=generator.concurrency,
    )
    runs = HarborRuns()
    return GeneratorService(
        tasks_root=tasks_root,
        designer=designer,
        checks=HarborChecks(harbor=generator.harbor, runs=runs),
        plays=plays,
        default_model=settings.upstream_model or settings.model_path,
        designer_model=generator.designer_model,
        probes=readiness_probes(harbor=generator.harbor),
        jobs=JobRunner(runs),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m reef.record2dataset",
        description="The generator service: writes, checks and plays Harbor tasks for a task generating processor.",
    )
    parser.add_argument(
        "-c",
        "--config",
        default=os.environ.get("REEF_CONFIG"),
        help="the deployment reef serve resolved (REEF_CONFIG under reef serve) or a deployment file",
    )
    arguments = parser.parse_args(argv)
    if not arguments.config:
        parser.error("a deployment is required: REEF_CONFIG under reef serve, or -c <file>")
    try:
        config = load_config(arguments.config)
        if config.get("schema-version") == 2:
            from reef.service.deploy.orchestrator import resolve_deployment_config

            config, _ = resolve_deployment_config(config, None, arguments.config)
        settings = service_config_from_mapping(config)
        if settings.generator_settings is None:
            raise DeployConfigError("the deployment has no generator section")
        generator = generator_settings(settings.generator_settings)
    except (DeployConfigError, ValueError) as exc:
        parser.error(str(exc))
    service = generator_service(settings, generator)
    web.run_app(service.app(), host=generator.host, port=generator.port, print=None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
