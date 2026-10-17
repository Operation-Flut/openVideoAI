"""
Native file and folder pickers.

The UI cannot read a filesystem path out of a browser file input, so "Browse"
in the Export dialog asks the machine - which is the same machine the backend
runs on - to show its own picker. Windows uses the WinForms dialog through a
short PowerShell script, macOS uses ``osascript`` and Linux falls back to
``zenity``/``kdialog``.

Everything here is best effort: when no picker is available the caller gets
``ok: False`` plus a reason, and the UI falls back to typing the path.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

# Long enough that a user can browse a slow network drive, short enough that a
# request cannot hang forever.
_PICKER_TIMEOUT_SECONDS = 300

_WINDOWS_SCRIPT = r"""
param([string]$Kind = 'file', [string]$Filter = '', [string]$Initial = '')
Add-Type -AssemblyName System.Windows.Forms
[System.Windows.Forms.Application]::EnableVisualStyles()
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$selected = @()
if ($Kind -eq 'folder') {
    $dlg = New-Object System.Windows.Forms.FolderBrowserDialog
    $dlg.Description = 'Choose a folder'
    $dlg.ShowNewFolderButton = $true
    if ($Initial -and (Test-Path $Initial)) { $dlg.SelectedPath = $Initial }
    if ($dlg.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) { $selected = @($dlg.SelectedPath) }
} else {
    $dlg = New-Object System.Windows.Forms.OpenFileDialog
    $dlg.Multiselect = $false
    $dlg.RestoreDirectory = $true
    if ($Filter) { $dlg.Filter = $Filter } else { $dlg.Filter = 'All files (*.*)|*.*' }
    if ($Initial -and (Test-Path $Initial)) { $dlg.InitialDirectory = $Initial }
    if ($dlg.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) { $selected = @($dlg.FileName) }
}
foreach ($item in $selected) { Write-Output $item }
"""


def native_dialogs_available() -> bool:
    """Whether this machine can show an OS picker for the backend process."""
    if _powershell_path():
        return True
    if sys.platform == "darwin":
        return shutil.which("osascript") is not None
    return any(shutil.which(tool) for tool in ("zenity", "kdialog"))


def _powershell_path() -> Optional[str]:
    if sys.platform != "win32":
        return None
    return shutil.which("powershell") or shutil.which("pwsh")


def _windows_filter(suffixes: Sequence[str]) -> str:
    """WinForms filter string for a list of extensions."""
    cleaned = [suffix if suffix.startswith(".") else f".{suffix}" for suffix in suffixes if suffix]
    if not cleaned:
        return "All files (*.*)|*.*"
    pattern = ";".join(f"*{suffix}" for suffix in cleaned)
    label = "Supported files"
    return f"{label} ({pattern})|{pattern}|All files (*.*)|*.*"


def _run(command: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=_PICKER_TIMEOUT_SECONDS,
    )


def _pick_windows(kind: str, suffixes: Sequence[str], start_dir: Optional[str]) -> List[str]:
    powershell = _powershell_path()
    if not powershell:
        return []
    script_path = Path(tempfile.gettempdir()) / "openvideoai_pick.ps1"
    try:
        script_path.write_text(_WINDOWS_SCRIPT, encoding="utf-8")
        result = _run([
            powershell, "-NoProfile", "-ExecutionPolicy", "Bypass",
            "-File", str(script_path),
            "-Kind", "folder" if kind == "folder" else "file",
            "-Filter", _windows_filter(suffixes),
            "-Initial", start_dir or "",
        ])
    finally:
        script_path.unlink(missing_ok=True)
    return [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]


def _pick_macos(kind: str) -> List[str]:
    if not shutil.which("osascript"):
        return []
    if kind == "folder":
        script = 'POSIX path of (choose folder with prompt "Choose a folder")'
    else:
        script = 'POSIX path of (choose file with prompt "Choose a file")'
    result = _run(["osascript", "-e", script])
    return [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]


def _pick_linux(kind: str, suffixes: Sequence[str], start_dir: Optional[str]) -> List[str]:
    zenity = shutil.which("zenity")
    if zenity:
        command = [zenity, "--file-selection", "--title=Choose a file" if kind != "folder" else "--title=Choose a folder"]
        if kind == "folder":
            command.append("--directory")
        if suffixes:
            pattern = " ".join(f"*{suffix if suffix.startswith('.') else '.' + suffix}" for suffix in suffixes)
            command.append(f"--file-filter=Supported files | {pattern}")
        if start_dir and Path(start_dir).is_dir():
            command.append(f"--filename={start_dir}{'/' if kind == 'folder' else ''}")
        result = _run(command)
        return [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]

    kdialog = shutil.which("kdialog")
    if kdialog:
        command = [kdialog, "--getexistingdirectory" if kind == "folder" else "--getopenfilename"]
        if start_dir:
            command.append(start_dir)
        if kind != "folder" and suffixes:
            command += ["", " ".join(f"*{s}" for s in suffixes)]
        result = _run(command)
        return [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]
    return []


def pick_path(
    kind: str = "file",
    *,
    suffixes: Optional[Sequence[str]] = None,
    start_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Show the OS picker and return the chosen path.

    ``{"ok": True, "path": "..."}`` on a selection, ``{"ok": True, "path": None}``
    when the user cancelled, and ``{"ok": False, "reason": "..."}`` when this
    machine has no picker to show.
    """
    if not native_dialogs_available():
        return {
            "ok": False,
            "reason": "This machine has no native file picker; type the path instead.",
        }

    suffixes = tuple(suffixes or ())
    try:
        if _powershell_path():
            selected = _pick_windows(kind, suffixes, start_dir)
        elif sys.platform == "darwin":
            selected = _pick_macos(kind)
        else:
            selected = _pick_linux(kind, suffixes, start_dir)
    except subprocess.TimeoutExpired:
        return {"ok": False, "reason": "The file picker timed out."}
    except Exception as error:  # pragma: no cover - a picker failure must not 500
        logger.warning("Native file picker failed: %s", error)
        return {"ok": False, "reason": f"Could not open the file picker: {error}"}

    if not selected:
        return {"ok": True, "path": None, "cancelled": True}
    return {"ok": True, "path": selected[0]}
