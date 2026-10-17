from .. import __version__ as version
import os
repo = "Operation-Flut/openVideoAI"

def check_for_update():
    """
    Check for updates to the application.

    Returns:
        dict: A dictionary containing the current version and the latest version available.
    """
    import requests

    try:
        response = requests.get(f"https://api.github.com/repos/{repo}/releases/latest")
        response.raise_for_status()
        latest_release = response.json()
        latest_version = latest_release["tag_name"]

        return {
            "current_version": version,
            "latest_version": latest_version,
            "update_available": latest_version != version
        }
    except requests.RequestException as e:
        return {
            "error": str(e),
            "current_version": version,
            "latest_version": None,
            "update_available": False
        }

def latest_pre_release():
    """
    Check for the latest pre-release version of the application.

    Returns:
        dict: A dictionary containing the current version and the latest pre-release version available.
    """
    import requests

    try:
        response = requests.get(f"https://api.github.com/repos/{repo}/releases")
        response.raise_for_status()
        releases = response.json()

        # Filter for pre-releases
        pre_releases = [release for release in releases if release["prerelease"]]
        if not pre_releases:
            return {
                "current_version": version,
                "latest_pre_release_version": None,
                "pre_release_available": False
            }

        latest_pre_release = pre_releases[0]  # Assuming the first one is the latest
        latest_pre_release_version = latest_pre_release["tag_name"]
        latest_pre_release_date = latest_pre_release["published_at"]

        return {
            "current_version": version,
            "latest_pre_release_version": latest_pre_release_version,
            "latest_pre_release_date": latest_pre_release_date,
            "pre_release_available": latest_pre_release_version != version
        }
    except requests.RequestException as e:
        return {
            "error": str(e),
            "current_version": version,
            "latest_pre_release_version": None,
            "pre_release_available": False
        }
def get_os():
    """
    Get the current operating system.

    Returns:
        str: The name of the operating system.
    """
    import platform
    return platform.system()

def install_update(release_tag):
    """
    Installs the Valid Installer File for the given release tag from the GitHub repository.
    """
    def get_installer_url(release_tag):
        if get_os() == "Windows":
            installer = True
            installer_url = f"https://github.com/{repo}/releases/download/{release_tag}/openVideoAI-Setup.exe"
        elif get_os() == "MacOS":
            installer = True
            installer_url = f"https://github.com/{repo}/releases/download/{release_tag}/openVideoAI-Setup.dmg"
        elif get_os() == "Linux":
            installer = True
            installer_url = f"https://github.com/{repo}/releases/download/{release_tag}/openVideoAI-Setup.AppImage"
        else:
            installer = False
            installer_url = f"https://github.com/{repo}/releases/download/{release_tag}/openVideoAI-Setup.zip"
        return installer, installer_url
    