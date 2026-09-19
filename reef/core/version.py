"""Installed Reef distribution version, without application imports."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("reef-infra")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"
