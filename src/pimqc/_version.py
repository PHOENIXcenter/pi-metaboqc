"""Resolve the installed pi-metaboqc distribution version.

Read package metadata without importing processing engines; an uninstalled
source checkout receives an explicit unknown-version marker.
"""

from importlib.metadata import PackageNotFoundError, version


try:
    __version__ = version("pi-metaboqc")
except PackageNotFoundError:
    __version__ = "0+unknown"


__all__ = ["__version__"]
