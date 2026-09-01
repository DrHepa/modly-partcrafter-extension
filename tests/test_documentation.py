from __future__ import annotations

import unittest

from tests._support import ROOT


class DocumentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.readme = (ROOT / "README.md").read_text(encoding="utf-8")
        cls.notices = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")

    def test_readme_is_extension_focused_and_concise(self) -> None:
        self.assertLessEqual(len(self.readme.splitlines()), 180)
        for phrase in (
            "Install from GitHub",
            "does not download model weights",
            "PartCrafter Object",
            "PartCrafter Scene",
            "PartCrafter RMBG Preprocess",
            "Models Directory",
            "Repair",
            "CPython 3.11",
            "3.12",
            "Linux ARM64",
            "SBSA (SM90+)",
            "Windows x64",
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, self.readme)

    def test_readme_states_runtime_boundaries_truthfully(self) -> None:
        for phrase in (
            "Full-checkpoint generation was not run",
            "RMBG",
            "Turntable rendering",
            "GEMINI_API_KEY",
            "local-only",
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, self.readme)

    def test_upstream_attribution_and_pins_are_recorded(self) -> None:
        for text in (self.readme, self.notices):
            with self.subTest(document="README" if text is self.readme else "notices"):
                self.assertIn("3d773bf02fad51c7ab31a5615573fec93b287b30", text)
                self.assertIn("wgsxm/PartCrafter", text)
                self.assertIn("MIT", text)
        self.assertIn("d7d1cf92c8d642af134f225ab447ff63b3b4784f1516d0c133c41e7cd0e2ccb6", self.notices)
        self.assertIn("69a0ffc1dad5e48e7e5ed91c0609f2b1276eb31f", self.notices)
        self.assertIn("0454bb8e595a2765e8cb1f17ffacad9ba159777a", self.notices)

    def test_file_level_license_limits_are_not_hidden_by_root_mit(self) -> None:
        for phrase in (
            "Tencent Hunyuan Community License",
            "European Union",
            "Acceptable Use Policy",
            "NOTICE",
            "BSD-3-Clause",
            "non-commercial",
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, self.readme + self.notices)
        self.assertIn("2ceba5a5efaec153162aedea169f76caf9b46cf8", self.notices)
        self.assertIn(
            "46ef7fe46f2ae284d8f1aaa24bfa5fca5ef25a34e2c7caa890a0029eb100e87f",
            self.notices,
        )
        self.assertIn("non-commercial", self.readme)
        self.assertIn("not covered", self.readme)


if __name__ == "__main__":
    unittest.main()
