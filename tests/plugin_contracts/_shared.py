from __future__ import annotations

import contextlib
import os
import sys
import types
from argparse import ArgumentParser
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest

ENV_PREFIX = "SLIME_CONTRACT_"


def install_paths() -> None:
    current = Path(__file__).resolve()
    repo_root = current.parents[2]
    sys.path.insert(0, str(current.parent))
    sys.path.insert(0, str(current.parents[1]))
    sys.path.insert(0, str(repo_root))


@contextlib.contextmanager
def stubbed_modules(
    *,
    with_sglang_router: bool = False,
    with_torch: bool = False,
    with_transformers: bool = False,
) -> Iterator[None]:
    """Provide import stubs for optional heavy dependencies, scoped to the block.

    Two rules keep the stubs from leaking into unrelated tests in the same
    pytest session:

    * Prefer the real library when installed: a stub shadowing a real
      dependency would poison real-dependency tests collected later.
    * Every stub this context installs is removed from ``sys.modules`` on
      exit, so a stub only exists while the block imports what it needs.
      (Modules imported inside the block may keep references to a stub;
      that is fine — they only need the attributes stubbed here.)
    """
    installed: list[str] = []

    def stub(name: str, module: types.ModuleType) -> None:
        sys.modules[name] = module
        installed.append(name)

    if with_torch and "torch" not in sys.modules:
        # Prefer the real library when installed (importing it registers it in
        # sys.modules, which is exactly what the block needs).
        try:
            import torch  # noqa: F401
        except ImportError:
            mod = types.ModuleType("torch")
            mod.Tensor = type("Tensor", (), {})
            stub("torch", mod)

    if with_sglang_router and "sglang_router" not in sys.modules:
        try:
            import sglang_router  # noqa: F401
        except ImportError:
            mod = types.ModuleType("sglang_router")
            mod.__version__ = "0.2.3"
            stub("sglang_router", mod)

    if with_transformers and "transformers" not in sys.modules:
        # Prefer the real library when installed: a stub left in sys.modules
        # would poison real-transformers tests later in the same session.
        try:
            import transformers  # noqa: F401
        except ImportError:
            mod = types.ModuleType("transformers")
            mod.AutoTokenizer = type(
                "AutoTokenizer", (), {"from_pretrained": staticmethod(lambda *args, **kwargs: object())}
            )
            mod.AutoProcessor = type(
                "AutoProcessor",
                (),
                {"from_pretrained": staticmethod(lambda *args, **kwargs: (_ for _ in ()).throw(OSError()))},
            )
            mod.PreTrainedTokenizerBase = type("PreTrainedTokenizerBase", (), {})
            mod.ProcessorMixin = type("ProcessorMixin", (), {})
            stub("transformers", mod)

    try:
        yield
    finally:
        for name in installed:
            sys.modules.pop(name, None)


def contract_env_name(key: str) -> str:
    return f"{ENV_PREFIX}{key}"


def get_contract_path(key: str, default: str | None = None) -> str | None:
    return os.environ.get(contract_env_name(key), default)


def run_contract_test_for_file(
    file: str,
    path_args: Sequence[str] = (),
    extra_args: Sequence[tuple[str, dict]] = (),
    extra_setup=None,
) -> None:
    """Parse ``--xxx-path`` CLI arguments, store as ``SLIME_CONTRACT_XXX_PATH``
    env vars, then call pytest on *file*.

    Args:
        file:        ``__file__`` of the calling contract test module.
        path_args:   Argument names **without** the leading ``--``, e.g.
                     ``["rollout-data-postprocess-path"]``. Each is exposed as a
                     positional env var ``FOO_BAR_PATH`` automatically.
        extra_args:  Additional ``(flag, kwargs)`` pairs forwarded verbatim to
                     ``parser.add_argument``.  Useful for boolean flags like
                     ``("--group-rm", {"action": "store_true", "default": False})``.
        extra_setup: Optional callable ``fn(parsed_args)`` invoked after env
                     vars are set, for any file-specific side-effects.
    """
    parser = ArgumentParser()
    for arg in path_args:
        parser.add_argument(f"--{arg}", default=None)
    for flag, kwargs in extra_args:
        parser.add_argument(flag, **kwargs)
    parsed, remaining = parser.parse_known_args()
    for key, value in vars(parsed).items():
        if value and isinstance(value, str):
            os.environ[contract_env_name(key.upper())] = value
    if extra_setup is not None:
        extra_setup(parsed)
    raise SystemExit(pytest.main([file, *remaining]))
