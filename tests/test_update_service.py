import runpy
from pathlib import Path

import pytest
import requests

from src.services import update


def test_check_for_update_reports_latest_release(monkeypatch):
    def get_release(url: str) -> dict[str, str]:
        assert url.endswith("/releases/latest")
        return {"tag_name": "v2.0.0"}

    monkeypatch.setattr(update, "_get_json", get_release)

    result = update.check_for_update()

    assert result == {
        "current_version": update.version,
        "latest_version": "v2.0.0",
        "update_available": True,
    }


def test_check_for_update_reports_request_errors(monkeypatch):
    def fail_request(url: str):
        assert url.endswith("/releases/latest")
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(update, "_get_json", fail_request)

    result = update.check_for_update()

    assert result["error"] == "offline"
    assert result["current_version"] == update.version
    assert result["latest_version"] is None
    assert result["update_available"] is False


def test_check_for_update_handles_no_published_stable_release(monkeypatch):
    response = requests.Response()
    response.status_code = 404

    def no_stable_release(url: str):
        assert url.endswith("/releases/latest")
        raise requests.HTTPError("not found", response=response)

    monkeypatch.setattr(update, "_get_json", no_stable_release)

    result = update.check_for_update()

    assert result["latest_version"] is None
    assert result["update_available"] is False
    assert result["message"] == "No stable release is currently published."


def test_latest_pre_release_uses_first_prerelease(monkeypatch):
    def get_releases(url: str) -> list[dict[str, object]]:
        assert url.endswith("/releases?per_page=100")
        return [
            {"prerelease": False, "tag_name": "v1.0.0"},
            {
                "prerelease": True,
                "tag_name": "v1.1.0-beta",
                "published_at": "2026-10-01T12:00:00Z",
            },
            {
                "prerelease": True,
                "tag_name": "v1.0.1-beta",
                "published_at": "2026-09-01T12:00:00Z",
            },
        ]

    monkeypatch.setattr(update, "_get_json", get_releases)

    result = update.latest_pre_release()

    assert result["latest_pre_release_version"] == "v1.1.0-beta"
    assert result["latest_pre_release_date"] == "2026-10-01T12:00:00Z"
    assert result["pre_release_available"] is True


def test_install_update_downloads_windows_asset(monkeypatch, tmp_path):
    monkeypatch.setattr(update, "get_os", lambda: "Windows")

    def get_release(url: str) -> dict[str, list[dict[str, str]]]:
        assert url.endswith("/releases/tags/v1.0.0")
        return {
            "assets": [
                {
                    "name": "openVideoAI-windows-x64.msi",
                    "browser_download_url": "https://example.test/app.msi",
                }
            ]
        }

    monkeypatch.setattr(update, "_get_json", get_release)
    monkeypatch.setattr(update.tempfile, "gettempdir", lambda: str(tmp_path))

    def write_asset(url: str, destination: Path) -> None:
        assert url == "https://example.test/app.msi"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"installer")

    monkeypatch.setattr(update, "_download_asset", write_asset)

    downloaded_path = update.install_update("v1.0.0")

    assert downloaded_path.name == "openVideoAI-windows-x64.msi"
    assert downloaded_path.read_bytes() == b"installer"


def test_install_update_joins_linux_release_parts(monkeypatch, tmp_path):
    monkeypatch.setattr(update, "get_os", lambda: "Linux")

    def get_release(url: str) -> dict[str, list[dict[str, str]]]:
        assert url.endswith("/releases/tags/v1.0.0")
        return {
            "assets": [
                {
                    "name": "openVideoAI-linux-x64.tar.gz.part-aa",
                    "browser_download_url": "https://example.test/part-aa",
                },
                {
                    "name": "openVideoAI-linux-x64.tar.gz.part-ab",
                    "browser_download_url": "https://example.test/part-ab",
                },
            ]
        }

    monkeypatch.setattr(update, "_get_json", get_release)
    monkeypatch.setattr(update.tempfile, "gettempdir", lambda: str(tmp_path))

    def write_asset(url: str, destination: Path) -> None:
        assert url in {
            "https://example.test/part-aa",
            "https://example.test/part-ab",
        }
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(url[-2:].encode())

    monkeypatch.setattr(update, "_download_asset", write_asset)

    downloaded_path = update.install_update("v1.0.0")

    assert downloaded_path.name == "openVideoAI-linux-x64.tar.gz"
    assert downloaded_path.read_bytes() == b"aaab"


def test_direct_script_loading_does_not_make_network_requests():
    script_path = Path(update.__file__)

    loaded_module = runpy.run_path(str(script_path))

    assert callable(loaded_module["check_for_update"])


@pytest.mark.parametrize("release_tag", ["", "../v1.0.0", "v1/0/0", ".."])
def test_install_update_rejects_invalid_release_tag(release_tag):
    with pytest.raises(ValueError, match="release_tag"):
        update.install_update(release_tag)
