"""
Distutils compatibility shim.

Python 3.12 removed the stdlib ``distutils`` package. Some third-party
dependencies still ``import distutils``; when setuptools is installed we expose
its bundled ``setuptools._distutils`` under the name ``distutils``.

Importing this module must never fail: if setuptools is unavailable the shim is
simply unavailable, and callers can continue without it.
"""
import sys
from types import ModuleType
from typing import Optional


def _load_bundled_distutils() -> Optional[ModuleType]:
    try:
        from setuptools import _distutils  # type: ignore
        return _distutils
    except Exception:
        return None


def install_distutils_compat() -> None:
    """Expose setuptools' bundled distutils as ``distutils`` when the stdlib
    version is missing (Python >= 3.12).

    Safe no-op if distutils is already importable or setuptools is not installed.
    """
    try:
        import distutils  # noqa: F401
        return  # stdlib distutils is available; nothing to do
    except ImportError:
        pass

    bundled = _load_bundled_distutils()
    if bundled is None:
        return  # setuptools not installed; cannot provide a shim

    class _DistutilsModule(ModuleType):
        def __getattr__(self, name: str):
            if hasattr(bundled, name):
                return getattr(bundled, name)
            try:
                return __import__(f"setuptools._distutils.{name}", fromlist=[""])
            except ImportError:
                raise AttributeError(f"module distutils has no attribute {name}") from None

    sys.modules.setdefault("distutils", _DistutilsModule("distutils"))
