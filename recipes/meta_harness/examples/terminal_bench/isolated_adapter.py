"""Executable Terminus candidate surface, admitted only with the E2B executor."""

import ast
import dataclasses
import re

from reef.harness.adapters.terminus.quirks import finalize_render as declarative_render
from reef.harness.render import RenderError

PREFIX = "terminus/context/"


def finalize_render(files):
    code = {path: value for path, value in files.items() if path.startswith(PREFIX)}
    rest = declarative_render({path: value for path, value in files.items() if path not in code})
    if len(code) > 1:
        raise RenderError("the executable harness must be one self-contained Python module exporting Agent")
    for path, source in code.items():
        if not re.fullmatch(r"terminus/context/[A-Za-z_][A-Za-z_0-9]*\.py", path):
            raise RenderError("executable harness module name must be a Python identifier")
        try:
            parsed = ast.parse(source, filename=path)
            compile(parsed, path, "exec")  # Compile only: never import a candidate on the host.
        except (SyntaxError, ValueError) as exc:
            raise RenderError("executable harness must compile") from exc
        if not any(isinstance(node, ast.ClassDef) and node.name == "Agent" for node in parsed.body):
            raise RenderError("executable harness must define class Agent, a Harbor BaseAgent subclass")
    return {**rest, **code}


def isolated_descriptor(descriptor, executor):
    from .e2b_executor import E2BEpisodeExecutor

    if not isinstance(executor, E2BEpisodeExecutor):
        raise ValueError("executable Terminus candidates require E2BEpisodeExecutor")
    executor.preflight()
    return dataclasses.replace(
        descriptor,
        binary="reef-terminus-e2b",
        finalize_render=finalize_render,
        writable_paths=("terminus/sessions", "terminus/trials"),
    )
