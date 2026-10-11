import importlib.util
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

_MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "generate_release_notes.py"
)
_spec = importlib.util.spec_from_file_location("generate_release_notes", _MODULE_PATH)
release_notes = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = release_notes
_spec.loader.exec_module(release_notes)


class CanonicalSectionTests(unittest.TestCase):
    def test_known_aliases_map_to_canonical_sections(self):
        cases = {
            "Features": "Features",
            "feature": "Features",
            "Bug Fixes": "Fixes",
            "fixed": "Fixes",
            "UI/UX": "UI/UX",
            "Documentation": "Docs",
            "Chores": "Technical",
            "Breaking Changes": "Breaking",
            "Misc": "Other",
        }
        for heading, expected in cases.items():
            with self.subTest(heading=heading):
                self.assertEqual(release_notes.canonical_section(heading), expected)

    def test_decorated_headings_still_match(self):
        self.assertEqual(release_notes.canonical_section("Features :sparkles:"), "Features")
        self.assertEqual(release_notes.canonical_section("## Improved"), "Improved")

    def test_unknown_heading_returns_none(self):
        self.assertIsNone(release_notes.canonical_section("Random Thoughts"))


class ParseAndRenderTests(unittest.TestCase):
    def test_render_uses_canonical_order_and_skips_empty_sections(self):
        text = (
            "## Fixes\n"
            "- Fix a crash on start\n"
            "\n"
            "### Features\n"
            "* Add tiling controls\n"
            "\n"
            "## Docs\n"
            "- Update README\n"
        )
        rendered = release_notes.render_sections(release_notes.parse_sections(text))
        self.assertIn("## Features", rendered)
        self.assertIn("## Fixes", rendered)
        self.assertNotIn("## New", rendered)
        self.assertLess(rendered.index("## Features"), rendered.index("## Fixes"))
        self.assertLess(rendered.index("## Fixes"), rendered.index("## Docs"))

    def test_code_fences_and_ansi_are_ignored(self):
        text = "```\n## Features\n```\n## Fixes\n\x1b[31m- Fix thing\x1b[0m\n"
        sections = release_notes.parse_sections(text)
        self.assertEqual(sections.get("Features"), [])
        self.assertEqual(sections.get("Fixes"), ["Fix thing"])


class LocalFallbackTests(unittest.TestCase):
    def test_conventional_commit_types_map_to_sections(self):
        subjects = [
            "feat: add GPU picker",
            "fix: stop crash on empty model list",
            "docs: describe settings",
            "perf: faster tiling",
            "chore: bump dependencies",
            "unstructured change",
        ]
        sections = release_notes.summarize_locally(subjects)
        self.assertEqual(sections["Features"], ["Add GPU picker"])
        self.assertEqual(sections["Fixes"], ["Stop crash on empty model list"])
        self.assertEqual(sections["Docs"], ["Describe settings"])
        self.assertEqual(sections["Improved"], ["Faster tiling"])
        self.assertEqual(sections["Technical"], ["Bump dependencies"])
        self.assertEqual(sections["Other"], ["Unstructured change"])

    def test_breaking_change_detection(self):
        self.assertEqual(
            release_notes.section_for_subject("feat!: drop legacy API"), "Breaking"
        )
        self.assertEqual(
            release_notes.section_for_subject("fix: x\n\nBREAKING CHANGE: y"),
            "Breaking",
        )


class OpenCodeModelChainTests(unittest.TestCase):
    def test_default_chain_starts_with_exo_free_then_big_pickle(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            chain = release_notes.opencode_model_chain()
        self.assertEqual(chain[:2], ["opencode/exo-free", "opencode/big-pickle"])
        self.assertEqual(chain, release_notes.DEFAULT_OPENCODE_MODELS)

    def test_single_model_override(self):
        with mock.patch.dict(os.environ, {"OPENCODE_MODEL": "opencode/custom"}, clear=True):
            self.assertEqual(release_notes.opencode_model_chain(), ["opencode/custom"])

    def test_comma_separated_models_override(self):
        env = {"OPENCODE_MODELS": "opencode/a, opencode/b\nopencode/c"}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(
                release_notes.opencode_model_chain(),
                ["opencode/a", "opencode/b", "opencode/c"],
            )


if __name__ == "__main__":
    unittest.main()
