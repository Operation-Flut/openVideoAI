from types import SimpleNamespace

from src.utils import system_resources


class _FakeXPU:
    @staticmethod
    def device_count():
        return 1

    @staticmethod
    def get_device_properties(index):
        assert index == 0
        return SimpleNamespace(name="Intel Test GPU", total_memory=8 * 1024**3)


def test_xpu_is_available_and_selected_automatically(monkeypatch):
    monkeypatch.setattr(system_resources, "has_cuda", False)
    monkeypatch.setattr(system_resources, "has_hip", False)
    monkeypatch.setattr(system_resources, "has_xpu", True)
    monkeypatch.setattr(system_resources, "has_metal", False)
    monkeypatch.setattr(system_resources, "has_dml_backend", lambda: False)
    monkeypatch.setattr(system_resources, "is_amd_hardware_detected", lambda: False)

    assert system_resources.get_available_devices() == ["xpu", "cpu"]
    assert system_resources.resolve_default_device() == "xpu"
    assert system_resources.get_platform_capabilities()["best_backend"] == "xpu"


def test_xpu_is_reported_as_a_selectable_gpu(monkeypatch):
    monkeypatch.setattr(system_resources, "has_xpu", True)
    monkeypatch.setattr(
        system_resources,
        "torch",
        SimpleNamespace(
            xpu=_FakeXPU,
            backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
        ),
    )
    monkeypatch.setattr(system_resources, "_get_os_adapters", lambda: [])

    gpu = system_resources.get_hardware_gpus()[0]

    assert gpu["device"] == "xpu:0"
    assert gpu["backend"] == "xpu"
    assert gpu["name"] == "Intel Test GPU"
    assert gpu["vram"] == 8
    assert gpu["selectable"] is True


def test_xpu_satisfies_gpu_model_requirements():
    result = system_resources.check_model_compatibility(
        requires_cuda=True,
        requires_metal=False,
        capabilities={"has_xpu": True},
    )

    assert result == {"compatible": True, "reason": ""}
