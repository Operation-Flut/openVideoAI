"""
System resource detection and hardware inspection utilities
"""
import os
import json
import platform
import subprocess
import psutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

try:
    import torch
except ImportError:
    torch = None

# Calculate RAM boundaries
_total_ram_gb = round(psutil.virtual_memory().total / (1024**3))
maxram = min(512, max(2, _total_ram_gb))

# Detected logical threads
cpu_threads = min(128, psutil.cpu_count(logical=True) or 8)
total_cpu_threads = psutil.cpu_count(logical=True) or 8

# Estimate default maximum VRAM
_detected_vram_gb = 8
if torch is not None and torch.cuda.is_available():
    try:
        _props = torch.cuda.get_device_properties(0)
        _detected_vram_gb = round(_props.total_memory / (1024**3))
    except Exception:
        pass
maxvram = min(128, max(2, _detected_vram_gb))

# Which compute stack is PyTorch built with: NVIDIA CUDA or AMD HIP/ROCm.
# has_cuda is intentionally gated to CUDA builds so ROCm builds do not misreport.
_torch_cuda_build = torch is not None and bool(getattr(torch.version, "cuda", None))
_torch_hip_build = torch is not None and bool(getattr(torch.version, "hip", None))

# Platform capabilities
has_cuda = bool(_torch_cuda_build and torch.cuda.is_available())
has_hip = bool(_torch_hip_build and torch.cuda.is_available())
has_metal = torch is not None and hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

# DirectML (AMD / Intel GPUs on Windows) is detected lazily so the heavy
# torch_directml import (which registers the 'dml' device type with PyTorch)
# only happens once a backend is actually requested.
_DML_STATUS: Optional[Tuple[bool, int]] = None
_OS_ADAPTERS_CACHE: Optional[List[Dict[str, Any]]] = None
_OS_ADAPTERS_CACHE_INITIALIZED: bool = False


def _get_os_adapters_wmi() -> List[Dict[str, Any]]:
    """Query OS video controllers via WMI (name + AdapterRAM estimate).

    Win32_VideoController.AdapterRAM is a 32-bit value that caps at 4 GB, so
    it underreports modern cards. Only used as a fallback when the registry
    query finds nothing.
    """
    result: List[Dict[str, Any]] = []
    if platform.system() == "Windows":
        try:
            cmd = [
                "powershell", "-NoProfile", "-Command",
                "Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name + '|' + [math]::Round(($_.AdapterRAM / 1GB), 1) }",
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=4)
            for line in res.stdout.strip().splitlines():
                line = line.strip()
                if not line or "|" not in line:
                    continue
                name, _, vram = line.rpartition("|")
                try:
                    vram_gb = float(vram)
                except ValueError:
                    vram_gb = 0.0
                result.append({"name": name.strip(), "vram": round(vram_gb)})
        except Exception:
            pass
    return result


def _get_os_adapters_registry() -> List[Dict[str, Any]]:
    """Read GPU adapter names + true VRAM from the Windows driver registry.

    The 'HardwareInformation.qwMemorySize' value holds the real adapter
    memory in bytes (a QWORD), which reports e.g. 8 GB correctly where WMI's
    AdapterRAM caps at 4 GB.
    """
    result: List[Dict[str, Any]] = []
    if platform.system() != "Windows":
        return result
    try:
        import winreg
    except ImportError:
        return result
    clsid = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, clsid) as base:
            index = 0
            while True:
                try:
                    subkey_name = winreg.EnumKey(base, index)
                except OSError:
                    break
                index += 1
                try:
                    with winreg.OpenKey(base, subkey_name) as subkey:
                        try:
                            desc = winreg.QueryValueEx(subkey, "DriverDesc")[0] or ""
                        except OSError:
                            desc = ""
                        try:
                            qw_memory = int(winreg.QueryValueEx(subkey, "HardwareInformation.qwMemorySize")[0])
                        except (OSError, TypeError, ValueError):
                            qw_memory = 0
                        if desc and qw_memory > 0:
                            result.append({
                                "name": desc.strip(),
                                "vram": round(qw_memory / (1024 ** 3)),
                            })
                except OSError:
                    continue
    except OSError:
        return []
    return result


def _get_os_adapters() -> List[Dict[str, Any]]:
    """Query OS video controllers (name + VRAM in GB), cached."""
    global _OS_ADAPTERS_CACHE, _OS_ADAPTERS_CACHE_INITIALIZED
    if _OS_ADAPTERS_CACHE_INITIALIZED:
        return _OS_ADAPTERS_CACHE
    # Registry first: accurate VRAM on Windows. WMI AdapterRAM caps at 4 GB.
    result = _get_os_adapters_registry()
    if not result:
        result = _get_os_adapters_wmi()
    _OS_ADAPTERS_CACHE = result
    _OS_ADAPTERS_CACHE_INITIALIZED = True
    return result


def has_dml_backend() -> bool:
    """Whether a DirectML accelerator (AMD/Intel GPU) is usable from PyTorch."""
    return _get_dml_status()[0]


def _get_dml_status() -> Tuple[bool, int]:
    """Return (has_directml, device_count). Importing torch_directml registers
    the 'privateuseone' backend with PyTorch; that name is what must be passed
    to torch.device() (str(torch_directml.device(i))) for the backend to work.
    The logical backend is labelled 'dml' throughout the app."""
    global _DML_STATUS
    if _DML_STATUS is None:
        try:
            import torch_directml
            count = int(torch_directml.device_count())
            _DML_STATUS = (count > 0, count)
        except Exception:
            _DML_STATUS = (False, 0)
    return _DML_STATUS


def _get_os_adapters_legacy() -> List[Dict[str, Any]]:
    """Legacy wrapper for backward compatibility."""
    return _get_os_adapters()


def is_amd_hardware_detected() -> bool:
    """Whether any OS-level video controller looks like an AMD GPU."""
    for adapter in _get_os_adapters():
        name = adapter["name"].lower()
        if any(keyword in name for keyword in ("amd", "radeon", "advanced micro devices")):
            return True
    return False


total_ram_gb = _total_ram_gb
total_vram_gb = _detected_vram_gb if (has_cuda or has_hip) else 0
if total_vram_gb == 0 and has_dml_backend():
    _os_adapters = _get_os_adapters()
    if _os_adapters:
        total_vram_gb = max(a["vram"] for a in _os_adapters)

# Cache for the raw models.json catalog
_MODELS_JSON_CACHE: Optional[Dict[str, Any]] = None


def get_platform_capabilities() -> Dict[str, Any]:
    """Get platform capabilities for model and backend compatibility"""
    has_dml = has_dml_backend()
    return {
        "has_cuda": has_cuda,
        "has_hip": has_hip,
        "has_dml": has_dml,
        "has_amd": has_hip or has_dml,
        "amd_hardware_detected": is_amd_hardware_detected(),
        "has_metal": has_metal,
        "total_ram_gb": total_ram_gb,
        "total_vram_gb": total_vram_gb,
        "total_cpu_threads": total_cpu_threads,
        "best_backend": (
            "cuda" if has_cuda else
            ("hip" if has_hip else
             ("dml" if has_dml else
              ("metal" if has_metal else "cpu")))
        ),
    }


# Order of preference when resolving the default compute device
SUPPORTED_DEVICE_TYPES = ("cuda", "hip", "dml", "privateuseone", "mps", "cpu")
_DEVICE_PREFERENCE = ("cuda", "hip", "dml", "mps")


def get_available_devices() -> List[str]:
    """Torch device types actually usable on this machine, best first."""
    devices: List[str] = []
    if has_cuda:
        devices.append("cuda")
    if has_hip:
        devices.append("hip")
    if has_dml_backend():
        devices.append("dml")
    if has_metal:
        devices.append("mps")
    devices.append("cpu")
    return devices


def recommended_tile_size(vram_gb: int) -> int:
    """
    Pick a Real-ESRGAN tile size that keeps GPU utilization high while still
    fitting comfortably in the detected VRAM. Larger tiles mean far fewer
    model dispatches per frame (each dispatch has fixed overhead, especially
    on DirectML), so this can cut processing time several-fold.
    """
    if vram_gb >= 12:
        return 1024
    if vram_gb >= 8:
        return 768
    if vram_gb >= 4:
        return 512
    return 400


_GPU_DEVICE_PREFIXES = ("cuda", "hip", "dml", "privateuseone")


def resolve_tile_size(requested: int, device: str) -> int:
    """
    Resolve the tile size for a GPU upscale job. The legacy shipped default
    (400) is treated as "not tuned" and auto-upgraded to a VRAM-aware tile;
    any other explicitly chosen value is respected as-is.
    """
    requested = int(requested or 400)
    if requested == 400 and device.startswith(_GPU_DEVICE_PREFIXES):
        return recommended_tile_size(total_vram_gb or 8)
    return requested


def _dml_device_string(index: int = 0) -> str:
    """Canonical torch device string for a DirectML adapter."""
    count = _get_dml_status()[1]
    if count <= 1:
        return "privateuseone"
    return f"privateuseone:{index}"


def resolve_default_device(device: str = "auto") -> str:
    """
    Resolve a user-supplied device string to a device guaranteed to exist on
    this machine. Unknown or unavailable strings (e.g. legacy "gpu:0" entries)
    fall back to the best available accelerator (CUDA > HIP/ROCm > DirectML >
    Metal) and finally CPU, so torch.device() never receives an invalid type.
    """
    available = set(get_available_devices())
    raw = (device or "auto").strip()

    if raw and raw != "auto":
        base = raw.split(":", 1)[0]
        if base in SUPPORTED_DEVICE_TYPES and (base in available or base == "privateuseone"):
            # DirectML: canonicalize to the privateuseone-device string and clamp the index
            if base in ("dml", "privateuseone"):
                if "dml" in available:
                    try:
                        index = int(raw.split(":", 1)[1]) if ":" in raw else 0
                    except (ValueError, TypeError):
                        index = 0
                    return _dml_device_string(index)
                # DirectML not present on this machine -> fall through to auto
            # Clamp CUDA/HIP indices that are out of range on this machine
            if base in ("cuda", "hip") and torch is not None:
                try:
                    index = int(raw.split(":", 1)[1]) if ":" in raw else 0
                    if index < 0 or index >= torch.cuda.device_count():
                        return base
                except (ValueError, TypeError):
                    return base
            return raw

    for candidate in _DEVICE_PREFERENCE:
        if candidate in available:
            return _dml_device_string(0) if candidate == "dml" else candidate
    return "cpu"


def get_model_platform_requirements(model_name: str) -> Dict[str, bool]:
    """
    Get platform requirements for a specific model.

    Requirements are read from models.json so the catalog stays the single
    source of truth for which backend each model needs.
    """
    for model in _load_models_json().get("models", []):
        if model.get("name") == model_name:
            return {
                "requires_cuda": bool(model.get("requiresCuda", False)),
                "requires_metal": bool(model.get("requiresMetal", False)),
                "can_run_on_cpu": bool(model.get("canRunOnCpu", True)),
                "is_ai": bool(model.get("isAi", True)),
            }

    return {
        "requires_cuda": False,
        "requires_metal": False,
        "can_run_on_cpu": True,
        "is_ai": True,
    }


def check_model_compatibility(
    requires_cuda: bool,
    requires_metal: bool,
    can_run_on_cpu: bool = True,
    capabilities: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Determine whether a model can be downloaded/run on the current platform.

    Model requirements are declared in models.json (requiresCuda / requiresMetal /
    canRunOnCpu) and are enforced here against the detected hardware capabilities.
    PyTorch models tagged requiresCuda run on any available GPU compute backend
    (NVIDIA CUDA, AMD ROCm/HIP, or AMD/Intel DirectML), so the requirement is
    considered satisfied whenever any such backend is present.
    """
    caps = capabilities if capabilities is not None else get_platform_capabilities()

    has_gpu_backend = (
        caps.get("has_cuda", False)
        or caps.get("has_hip", False)
        or caps.get("has_dml", False)
    )

    compatible = True
    reason = ""
    if requires_metal and not caps.get("has_metal", False):
        compatible = False
        reason = "Requires an Apple Silicon (Metal) device"
    elif requires_cuda and not has_gpu_backend:
        compatible = False
        reason = "Requires a GPU accelerator (NVIDIA CUDA, AMD DirectML, or ROCm) — none detected"
    elif not can_run_on_cpu and not has_gpu_backend:
        compatible = False
        reason = "Requires a GPU accelerator — none detected"

    return {
        "compatible": compatible,
        "reason": reason,
    }


def _load_models_json() -> Dict[str, Any]:
    """Load the raw models.json catalog from the project root (cached)."""
    global _MODELS_JSON_CACHE
    if _MODELS_JSON_CACHE is not None:
        return _MODELS_JSON_CACHE

    try:
        from src.core.paths import models_json as _models_json
        catalog_path = _models_json()
    except Exception:
        catalog_path = Path(__file__).resolve().parents[2] / "models.json"

    try:
        _MODELS_JSON_CACHE = json.loads(catalog_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        _MODELS_JSON_CACHE = {"models": []}
    return _MODELS_JSON_CACHE


def get_hardware_gpus() -> List[Dict[str, Any]]:
    """
    Detect usable GPU accelerators (CUDA / HIP-ROCm / Metal / DirectML) and,
    on Windows, list OS-level video adapters for information. Only entries with
    a real PyTorch device string (entry.selectable) may be chosen for processing.
    """
    gpus: List[Dict[str, Any]] = []
    os_adapters = _get_os_adapters()

    def _adapter_name(index: int, fallback: str) -> str:
        if 0 <= index < len(os_adapters) and os_adapters[index]["name"]:
            return os_adapters[index]["name"]
        return fallback

    # 1. NVIDIA CUDA devices
    if has_cuda:
        for i in range(torch.cuda.device_count()):
            try:
                props = torch.cuda.get_device_properties(i)
                vram_gb = round(props.total_memory / (1024**3))
                gpus.append({
                    "index": i,
                    "device": f"cuda:{i}",
                    "backend": "cuda",
                    "name": props.name,
                    "vram": vram_gb,
                    "hardware_accelerated": True,
                    "selectable": True,
                })
            except Exception:
                pass

    # 2. AMD HIP / ROCm devices (ROCm PyTorch builds)
    if has_hip:
        for i in range(torch.cuda.device_count()):
            try:
                props = torch.cuda.get_device_properties(i)
                vram_gb = round(props.total_memory / (1024**3))
                gpus.append({
                    "index": i,
                    "device": f"hip:{i}",
                    "backend": "hip",
                    "name": props.name or _adapter_name(i, "AMD GPU (ROCm/HIP)"),
                    "vram": vram_gb,
                    "hardware_accelerated": True,
                    "selectable": True,
                    "note": "AMD GPU acceleration via ROCm/HIP",
                })
            except Exception:
                pass

    # 3. Apple Metal (MPS)
    if torch is not None and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        gpus.append({
            "index": 0,
            "device": "mps",
            "backend": "metal",
            "name": "Apple GPU (Metal)",
            "vram": 0,
            "hardware_accelerated": True,
            "selectable": True,
        })

    # 4. DirectML (AMD / Intel GPUs on Windows)
    dml_ok, dml_count = _get_dml_status()
    if dml_ok:
        for i in range(dml_count):
            vram = os_adapters[i]["vram"] if i < len(os_adapters) else 0
            gpus.append({
                "index": i,
                "device": _dml_device_string(i),
                "backend": "dml",
                "name": _adapter_name(i, "DirectML GPU"),
                "vram": vram,
                "hardware_accelerated": True,
                "selectable": True,
                "note": "AMD GPU acceleration via DirectML",
            })

    # 5. OS-level adapters not backed by an active PyTorch accelerator.
    #    These are informational only and carry no usable device string.
    if platform.system() == "Windows":
        accelerator_names = [
            g.get("name", "").lower()
            for g in gpus
            if g.get("backend") in ("cuda", "hip", "dml")
        ]
        for idx, adapter in enumerate(os_adapters):
            name = adapter["name"]
            if not name:
                continue
            covered = any(
                name.lower() in gname or gname in name.lower()
                for gname in accelerator_names
            )
            if covered:
                continue
            gpus.append({
                "index": 100 + idx,
                "device": "",
                "backend": "hardware",
                "name": name,
                "vram": adapter["vram"],
                "hardware_accelerated": False,
                "selectable": False,
                "note": "Detected on host system; no PyTorch accelerator backend is active for this adapter",
            })

    # Fallback CPU entry
    gpus.append({
        "index": -1,
        "device": "cpu",
        "backend": "cpu",
        "name": "CPU (Software Acceleration)",
        "vram": 0,
        "hardware_accelerated": False,
        "selectable": True,
    })

    return gpus


def get_system_hardware_info() -> Dict[str, Any]:
    """Get summarized system hardware specifications"""
    cpu_cores = psutil.cpu_count(logical=True) or os.cpu_count() or 4
    mem = psutil.virtual_memory()
    total_ram_gb = mem.total / (1024**3)
    available_ram_gb = mem.available / (1024**3)

    gpu_info = "CPU (software)"
    vram_info = "N/A"

    if torch is not None:
        if has_cuda:
            gpu_info = torch.cuda.get_device_name(0)
            vram_info = f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB"
        elif has_hip:
            gpu_info = torch.cuda.get_device_name(0)
            vram_info = f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB"
        elif has_dml_backend():
            adapters = _get_os_adapters()
            gpu_info = adapters[0]["name"] if adapters else "DirectML GPU (AMD/Intel)"
            vram_info = f"{total_vram_gb} GB" if total_vram_gb else "Shared system memory"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            gpu_info = "Apple GPU (Metal)"
            vram_info = "Shared system memory"

    # If PyTorch doesn't see an accelerator, note the OS-level adapter name
    if gpu_info == "CPU (software)":
        hardware_list = get_hardware_gpus()
        for g in hardware_list:
            if g.get("backend") == "hardware":
                gpu_info = f"{g['name']} (CPU PyTorch Mode)"
                break

    return {
        "cpu": f"{cpu_cores} threads",
        "ram": f"{total_ram_gb:.1f} GB ({available_ram_gb:.1f} GB free)",
        "gpu": gpu_info,
        "vram": vram_info,
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version()
    }
