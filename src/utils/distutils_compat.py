import sys
from types import ModuleType

from setuptools import _distutils


class _DistutilsModule(ModuleType):
    def __getattr__(self, name: str):
        if hasattr(_distutils, name):
            return getattr(_distutils, name)
        try:
            return __import__(f"setuptools._distutils.{name}", fromlist=[""])
        except ImportError:
            raise AttributeError(f"module distutils has no attribute {name}") from None


def install_distutils_compat() -> None:
    sys.modules["distutils"] = _DistutilsModule("distutils")
