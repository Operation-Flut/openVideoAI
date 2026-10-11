#!/usr/bin/env python3
"""Generate categorized GitHub release notes for openVideoAI.

Provider order:

1. OpenCode CLI (``opencode run``) - the primary provider. It walks a waterfall
   of OpenCode Zen free models (``exo-free`` -> ``big-pickle`` -> ``nemotron``
   -> ``space-bunny`` -> ``muse-spark``), using the first one that returns
   usable notes. Authenticate with ``OPENCODE_API_KEY``. Override the chain with
   ``OPENCODE_MODELS`` (comma-separated) or ``OPENCODE_MODEL`` (single).
2. Mistral chat completions - the fallback provider. Configure it with
   ``MISTRAL_API_KEY`` (and optionally ``MISTRAL_MODEL``).
3. Local commit summarization - a deterministic last resort so a release is
   never blocked just because no AI provider is reachable.

The script only uses the standard library so it can run in a release job without
installing the project's heavy AI dependencies.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

# Canonical section order for the rendered notes. Only sections with at least
# one entry are emitted, but this order is always preserved.
SECTION_ORDER: List[str] = [
    "Features",
    "New",
    "Breaking",
    "Improved",
    "Fixes",
    "UI/UX",
    "Docs",
    "Technical",
    "Other",
]

# Lower-cased heading alias -> canonical section name.
SECTION_ALIASES: Dict[str, str] = {
    "feature": "Features",
    "features": "Features",
    "new": "New",
    "added": "New",
    "addition": "New",
    "additions": "New",
    "breaking": "Breaking",
    "breaking change": "Breaking",
    "breaking changes": "Breaking",
    "improved": "Improved",
    "improvement": "Improved",
    "improvements": "Improved",
    "enhancement": "Improved",
    "enhancements": "Improved",
    "fix": "Fixes",
    "fixes": "Fixes",
    "bug fix": "Fixes",
    "bug fixes": "Fixes",
    "bugfix": "Fixes",
    "bugfixes": "Fixes",
    "ui/ux": "UI/UX",
    "ui": "UI/UX",
    "ux": "UI/UX",
    "design": "UI/UX",
    "doc": "Docs",
    "docs": "Docs",
    "documentation": "Docs",
    "technical": "Technical",
    "tech": "Technical",
    "internal": "Technical",
    "chore": "Technical",
    "chores": "Technical",
    "build": "Technical",
    "ci": "Technical",
    "dependencies": "Technical",
    "other": "Other",
    "misc": "Other",
    "miscellaneous": "Other",
}

# Conventional-commit type -> canonical section (used by the offline fallback).
COMMIT_TYPE_SECTIONS: Dict[str, str] = {
    "feat": "Features",
    "feature": "Features",
    "new": "New",
    "fix": "Fixes",
    "bugfix": "Fixes",
    "docs": "Docs",
    "doc": "Docs",
    "perf": "Improved",
    "refactor": "Technical",
    "chore": "Technical",
    "build": "Technical",
    "ci": "Technical",
    "test": "Technical",
    "tests": "Technical",
    "style": "UI/UX",
    "revert": "Other",
}

_HEADING_RE = re.compile(r"^#{1,6}\s*(.+?)\s*$")
_BULLET_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)(.*\S)\s*$")
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
_COMMIT_RE = re.compile(
    r"^(?P<type>[a-zA-Z]+)(?:\((?P<scope>[^)]*)\))?(?P<breaking>!)?:\s*(?P<desc>.+)$"
)


def canonical_section(heading: str) -> Optional[str]:
    """Map a free-form heading to a canonical section name."""
    key = heading.strip().strip("#:*").lower()
    if not key:
        return None
    if key in SECTION_ALIASES:
        return SECTION_ALIASES[key]
    # Loose match for decorated headings such as "Features :sparkles:".
    for alias, section in SECTION_ALIASES.items():
        if alias in key:
            return section
    return None


def parse_sections(text: str) -> Dict[str, List[str]]:
    """Extract bullet entries grouped by canonical section from Markdown."""
    sections: Dict[str, List[str]] = {}
    current: Optional[str] = None
    cleaned = _strip_code_fences(_strip_ansi(text))
    for raw in cleaned.splitlines():
        line = raw.rstrip()
        heading = _HEADING_RE.match(line)
        if heading:
            current = canonical_section(heading.group(1))
            if current is not None:
                sections.setdefault(current, [])
            continue
        bullet = _BULLET_RE.match(line)
        if bullet:
            if current is not None:
                item = bullet.group(1).strip()
                if item:
                    sections.setdefault(current, []).append(item)
            continue
        if current is not None and line.strip():
            # Fold stray prose into the current section as a bullet.
            sections.setdefault(current, []).append(line.strip())
    return sections


def render_sections(sections: Dict[str, List[str]]) -> str:
    """Render sections as Markdown in the canonical order, skipping empties."""
    blocks: List[str] = []
    for name in SECTION_ORDER:
        items = _dedupe(sections.get(name, []))
        if not items:
            continue
        body = "\n".join(f"- {item}" for item in items)
        blocks.append(f"## {name}\n{body}")
    return "\n\n".join(blocks) + "\n" if blocks else ""


def section_for_subject(subject: str) -> str:
    """Best-effort section for a commit subject (offline fallback)."""
    head = subject.strip()
    if "BREAKING CHANGE" in subject.upper():
        return "Breaking"
    match = _COMMIT_RE.match(head)
    if match:
        if match.group("breaking"):
            return "Breaking"
        return COMMIT_TYPE_SECTIONS.get(match.group("type").lower(), "Other")
    return "Other"


def summarize_locally(subjects: Sequence[str]) -> Dict[str, List[str]]:
    """Categorize commit subjects without any AI provider."""
    sections: Dict[str, List[str]] = {}
    for subject in subjects:
        subject = subject.strip()
        if not subject:
            continue
        match = _COMMIT_RE.match(subject)
        text = match.group("desc").strip() if match else subject
        if not text:
            continue
        sections.setdefault(section_for_subject(subject), []).append(
            text[0].upper() + text[1:]
        )
    return sections


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dedupe(items: Sequence[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for item in items:
        key = item.strip().lower()
        if key and key not in seen:
            seen.add(key)
            result.append(item.strip())
    return result


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def _strip_code_fences(text: str) -> str:
    lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
    return "\n".join(lines)


def _log(message: str) -> None:
    print(f"[release-notes] {message}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Repository changes
# ---------------------------------------------------------------------------


def _run_git(args: Sequence[str], cwd: Optional[str] = None) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, ["git", *args], result.stdout, result.stderr
        )
    return result.stdout


def previous_tag(to_ref: str = "HEAD", cwd: Optional[str] = None) -> Optional[str]:
    """Return the most recent reachable version tag before ``to_ref``."""
    try:
        tag = _run_git(
            ["describe", "--tags", "--abbrev=0", "--match", "v[0-9]*", f"{to_ref}^"],
            cwd,
        ).strip()
    except subprocess.CalledProcessError:
        return None
    return tag or None


def collect_repository_changes(
    from_ref: Optional[str],
    to_ref: str,
    cwd: Optional[str] = None,
    max_commits: int = 300,
) -> Tuple[List[str], str]:
    """Return (commit subjects, prompt-ready change summary)."""
    commit_range = f"{from_ref}..{to_ref}" if from_ref else to_ref

    log = _run_git(
        [
            "log",
            "--no-merges",
            f"--max-count={max_commits}",
            "--pretty=format:%s",
            commit_range,
        ],
        cwd,
    )
    subjects = [line.strip() for line in log.splitlines() if line.strip()]

    try:
        diff = _run_git(["diff", "--name-only", commit_range], cwd)
    except subprocess.CalledProcessError:
        diff = ""
    changed_files = [line.strip() for line in diff.splitlines() if line.strip()]

    lines = ["COMMITS:"]
    lines.extend(f"- {subject}" for subject in subjects)
    if not subjects:
        lines.append("- (no commits found)")
    if changed_files:
        lines.append("")
        lines.append("CHANGED FILES:")
        lines.extend(f"- {path}" for path in changed_files[:200])
    return subjects, "\n".join(lines)


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = """You write GitHub release notes for the openVideoAI project.

Produce concise Markdown release notes for version {version} from the changes
listed below.

Use exactly these section headings, in this order, and include ONLY the
sections that contain at least one entry:

## Features
## New
## Breaking
## Improved
## Fixes
## UI/UX
## Docs
## Technical
## Other

Guidelines:
- One short bullet per change (max ~15 words); never include commit hashes.
- Features: new user-facing functionality.
- New: newly added models, integrations, or content.
- Breaking: changes that require user action or break compatibility.
- Improved: performance and refinements of existing behaviour.
- Fixes: bug fixes.
- UI/UX: interface, workflow, or visual changes.
- Docs: documentation changes.
- Technical: internal, build, CI, or dependency changes.
- Other: anything that does not fit the categories above.
- Output Markdown only: no preamble, no code fences, no extra headings.

Changes:
{changes}
"""


# OpenCode Zen free models, tried in order until one produces usable notes.
# Override with the OPENCODE_MODELS env var (comma-separated) or a single
# OPENCODE_MODEL.
DEFAULT_OPENCODE_MODELS: List[str] = [
    "opencode/exo-free",
    "opencode/big-pickle",
    "opencode/nemotron-3-ultra-free",
    "opencode/space-bunny-free",
    "opencode/muse-spark-1.3-contributor-free",
]


def opencode_model_chain() -> List[str]:
    """Return the ordered list of OpenCode models to try."""
    raw = os.environ.get("OPENCODE_MODELS", "").strip()
    if raw:
        models = [item.strip() for item in re.split(r"[,\n]", raw) if item.strip()]
        if models:
            return models
    single = os.environ.get("OPENCODE_MODEL", "").strip()
    if single:
        return [single]
    return list(DEFAULT_OPENCODE_MODELS)


def generate_with_opencode(
    prompt: str,
    *,
    model: Optional[str] = None,
    timeout: int = 180,
) -> str:
    """Generate notes with the OpenCode CLI (``opencode run``)."""
    executable = shutil.which("opencode")
    if not executable:
        raise RuntimeError("opencode CLI not found on PATH")

    command = [executable, "run", "--standalone"]
    if model:
        command += ["--model", model]
    extra = os.environ.get("OPENCODE_RUN_ARGS", "").strip()
    if extra:
        command += shlex.split(extra)
    command.append(prompt)

    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip().splitlines()
        raise RuntimeError(
            f"opencode run exited {result.returncode}: {detail[-1] if detail else 'no output'}"
        )
    text = _strip_ansi(result.stdout).strip()
    if not text:
        raise RuntimeError("opencode run produced no output")
    return text


def generate_with_mistral(
    prompt: str,
    *,
    api_key: Optional[str],
    model: str = "mistral-large-latest",
    timeout: int = 120,
) -> str:
    """Generate notes with the Mistral chat completions API."""
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY is not set")

    payload = json.dumps(
        {
            "model": model,
            "temperature": 0.2,
            "messages": [{"role": "user", "content": prompt}],
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        "https://api.mistral.ai/v1/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"unexpected Mistral response: {exc}") from exc


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def build_notes(
    *,
    version: str,
    from_ref: Optional[str],
    to_ref: str,
    cwd: Optional[str] = None,
    max_commits: int = 300,
) -> Tuple[str, str]:
    """Return ``(markdown, provider)`` for the given revision range."""
    subjects, changes = collect_repository_changes(from_ref, to_ref, cwd, max_commits)
    prompt = PROMPT_TEMPLATE.format(version=version or "(unreleased)", changes=changes)
    errors: List[str] = []

    # Primary: the OpenCode CLI, walking the free-model waterfall in order.
    opencode_timeout = int(os.environ.get("OPENCODE_TIMEOUT", "180"))
    for model in opencode_model_chain():
        try:
            text = generate_with_opencode(prompt, model=model, timeout=opencode_timeout)
            sections = parse_sections(text)
            if sections:
                return render_sections(sections), f"opencode ({model})"
            errors.append(f"opencode [{model}]: output contained no recognized sections")
        except Exception as exc:  # noqa: BLE001 - try the next model in the chain
            errors.append(f"opencode [{model}]: {exc}")

    # Fallback: Mistral.
    try:
        text = generate_with_mistral(
            prompt,
            api_key=os.environ.get("MISTRAL_API_KEY"),
            model=os.environ.get("MISTRAL_MODEL") or "mistral-large-latest",
            timeout=int(os.environ.get("MISTRAL_TIMEOUT", "120")),
        )
        sections = parse_sections(text)
        if sections:
            return render_sections(sections), "mistral"
        errors.append("mistral: output contained no recognized sections")
    except Exception as exc:  # noqa: BLE001 - fall through to the local summary
        errors.append(f"mistral: {exc}")

    for error in errors:
        _log(error)
    _log("falling back to local commit summarization")
    return render_sections(summarize_locally(subjects)), "local"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version",
        default=os.environ.get("RELEASE_VERSION", ""),
        help="Version label shown in the notes (default: $RELEASE_VERSION).",
    )
    parser.add_argument(
        "--from",
        dest="from_ref",
        default=None,
        help="Start revision (tag/sha). Defaults to the previous version tag.",
    )
    parser.add_argument(
        "--to",
        dest="to_ref",
        default="HEAD",
        help="End revision (default: HEAD).",
    )
    parser.add_argument(
        "--output",
        default="RELEASE_NOTES.md",
        help="Markdown file to write (default: RELEASE_NOTES.md).",
    )
    parser.add_argument("--repo", default=None, help="Repository directory.")
    parser.add_argument(
        "--max-commits",
        type=int,
        default=300,
        help="Maximum commits to summarize (default: 300).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if no AI provider could generate the notes.",
    )
    args = parser.parse_args(argv)

    from_ref = args.from_ref
    if from_ref is None:
        from_ref = previous_tag(args.to_ref, args.repo)

    notes, provider = build_notes(
        version=args.version,
        from_ref=from_ref,
        to_ref=args.to_ref,
        cwd=args.repo,
        max_commits=args.max_commits,
    )

    output = Path(args.output)
    output.write_text(notes, encoding="utf-8")
    _log(
        f"wrote {output} using '{provider}' provider "
        f"(range {from_ref or 'start'}..{args.to_ref})"
    )

    if args.strict and provider == "local":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
