"""Check for openVideoAI releases and download their platform-specific assets."""

from __future__ import annotations

import json
import platform
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.core.config import __version__ as version  # noqa: E402

repo = "Operation-Flut/openVideoAI"
_API_URL = f"https://api.github.com/repos/{repo}"
_REQUEST_TIMEOUT = 15
_DOWNLOAD_TIMEOUT = 60


def _get_json(url: str) -> Any:
    response = requests.get(url, timeout=_REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


def check_for_update() -> dict[str, object]:
    """Return the current and latest stable release versions."""
    try:
        release = _get_json(f"{_API_URL}/releases/latest")
    except requests.RequestException as error:
        if isinstance(error, requests.HTTPError) and error.response is not None:
            if error.response.status_code == 404:
                return {
                    "current_version": version,
                    "latest_version": None,
                    "update_available": False,
                    "message": "No stable release is currently published.",
                }
        return {
            "error": str(error),
            "current_version": version,
            "latest_version": None,
            "update_available": False,
        }

    latest_version = release["tag_name"]
    return {
        "current_version": version,
        "latest_version": latest_version,
        "update_available": latest_version != version,
    }


def latest_pre_release() -> dict[str, object]:
    """Return the latest pre-release version, if one is available."""
    try:
        releases = _get_json(f"{_API_URL}/releases?per_page=100")
    except requests.RequestException as error:
        return {
            "error": str(error),
            "current_version": version,
            "latest_pre_release_version": None,
            "pre_release_available": False,
        }

    pre_releases = [release for release in releases if release["prerelease"]]
    if not pre_releases:
        return {
            "current_version": version,
            "latest_pre_release_version": None,
            "pre_release_available": False,
        }

    latest = pre_releases[0]
    latest_version = latest["tag_name"]
    return {
        "current_version": version,
        "latest_pre_release_version": latest_version,
        "latest_pre_release_date": latest["published_at"],
        "pre_release_available": latest_version != version,
    }


def get_os() -> str:
    """Return the platform name reported by Python."""
    return platform.system()


def _asset_names(system: str) -> tuple[str, ...]:
    if system == "Windows":
        return ("openVideoAI-windows-x64.msi",)
    if system == "Darwin":
        return ("openVideoAI-macos.dmg",)
    if system == "Linux":
        return ()
    raise OSError(f"Update downloads are not available for {system}.")


def _download_asset(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with requests.get(url, stream=True, timeout=_DOWNLOAD_TIMEOUT) as response:
            response.raise_for_status()
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".download",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        temporary_file.write(chunk)
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def install_update(release_tag: str) -> Path:
    """Download the release asset for this platform and return its local path.

    The downloaded installer or archive is not opened or executed.
    """
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", release_tag):
        raise ValueError("release_tag must contain only letters, numbers, '.', '_' or '-'.")

    release = _get_json(f"{_API_URL}/releases/tags/{quote(release_tag, safe='')}")
    assets = release["assets"]
    system = get_os()
    asset_names = _asset_names(system)

    if system == "Linux":
        matching_assets = sorted(
            (
                asset
                for asset in assets
                if asset["name"].startswith("openVideoAI-linux-x64.tar.gz.part-")
            ),
            key=lambda asset: asset["name"],
        )
        if not matching_assets:
            raise FileNotFoundError(f"No Linux release assets found for {release_tag}.")
        output_name = "openVideoAI-linux-x64.tar.gz"
    else:
        matching_assets = [asset for asset in assets if asset["name"] in asset_names]
        if not matching_assets:
            raise FileNotFoundError(f"No {system} release asset found for {release_tag}.")
        matching_assets = matching_assets[:1]
        output_name = matching_assets[0]["name"]

    download_dir = Path(tempfile.gettempdir()) / "openVideoAI-updates" / release_tag
    if system != "Linux":
        destination = download_dir / output_name
        _download_asset(matching_assets[0]["browser_download_url"], destination)
        return destination

    destination = download_dir / output_name
    download_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=download_dir.parent, prefix=".linux-update-") as staging:
        staging_dir = Path(staging)
        archive_path = staging_dir / output_name
        with archive_path.open("wb") as archive:
            for asset in matching_assets:
                part_path = staging_dir / asset["name"]
                _download_asset(asset["browser_download_url"], part_path)
                with part_path.open("rb") as part:
                    shutil.copyfileobj(part, archive)
        destination.parent.mkdir(parents=True, exist_ok=True)
        archive_path.replace(destination)
    return destination


def main() -> int:
    """Run the stable-release check and print its result as JSON."""
    result = check_for_update()
    print(json.dumps(result, indent=2))
    return 1 if "error" in result else 0


if __name__ == "__main__":
    raise SystemExit(main())
