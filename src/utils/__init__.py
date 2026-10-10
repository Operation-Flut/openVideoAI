"""
Utilities package for openVideoAI
"""
from .distutils_compat import install_distutils_compat
from .system_resources import maxram, maxvram, cpu_threads, get_hardware_gpus, get_system_hardware_info

__all__ = [
    'install_distutils_compat',
    'maxram',
    'maxvram',
    'cpu_threads',
    'get_hardware_gpus',
    'get_system_hardware_info',
]
