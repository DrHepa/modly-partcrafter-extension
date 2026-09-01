from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path

from tests._support import ROOT


TEXT_SUFFIXES = {".py", ".json", ".md", ".txt", ".yml", ".yaml", ".toml"}


class RepositoryHygieneTests(unittest.TestCase):
    def test_required_extension_files_exist(self) -> None:
        for relative in (
            "manifest.json",
            "setup.py",
            "generator.py",
            "README.md",
            "LICENSE",
            "THIRD_PARTY_NOTICES.md",
            "NOTICE",
            "requirements-runtime.txt",
            "LICENSES/PARTCRAFTER-MIT.txt",
            "LICENSES/SMOOTHING-BSD-3-CLAUSE.txt",
        ):
            with self.subTest(path=relative):
                self.assertTrue((ROOT / relative).is_file(), relative)

    def test_python_files_parse(self) -> None:
        for path in sorted(ROOT.rglob("*.py")):
            if any(part in {"venv", ".venv", ".modly"} for part in path.parts):
                continue
            with self.subTest(path=path.relative_to(ROOT)):
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_no_repository_binaries_caches_or_weights(self) -> None:
        cache_names = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
        forbidden_names = {"venv", ".venv", ".modly"}
        forbidden_suffixes = {".pyc", ".pyo", ".safetensors", ".ckpt", ".pth", ".pt"}
        violations: list[str] = []
        for path in ROOT.rglob("*"):
            relative = path.relative_to(ROOT)
            if any(part in cache_names for part in relative.parts):
                # Running unittest/compileall creates ignored caches. This test
                # guards shipped sources, not ephemeral ignored test output.
                continue
            if any(part in forbidden_names for part in relative.parts):
                violations.append(str(relative))
            elif path.is_file() and path.suffix.lower() in forbidden_suffixes:
                violations.append(str(relative))
        self.assertEqual(violations, [])

    def test_no_placeholders_or_unresolved_work_markers(self) -> None:
        marker = re.compile(
            r"\b(?:TO" r"DO|FIXME|HACK|PLACEHOLDER|NOT[ _-]IMPLEMENTED|planned_identity)\b",
            re.IGNORECASE,
        )
        violations: list[str] = []
        for path in ROOT.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            if path.name == "test_repository.py":
                continue
            match = marker.search(path.read_text(encoding="utf-8", errors="replace"))
            if match:
                violations.append(f"{path.relative_to(ROOT)}:{match.group(0)}")
        self.assertEqual(violations, [])

    def test_no_hardcoded_user_home(self) -> None:
        patterns = (
            re.compile(r"/home/[A-Za-z0-9_.-]+/"),
            re.compile(r"[A-Za-z]:\\\\Users\\\\[A-Za-z0-9_.-]+", re.IGNORECASE),
        )
        violations: list[str] = []
        for path in ROOT.rglob("*"):
            # README examples may show an explicitly labelled path requested by
            # the user. Runtime code/configuration must remain relocatable.
            if not path.is_file() or path.suffix.lower() not in {
                ".py",
                ".json",
                ".yml",
                ".yaml",
                ".toml",
            }:
                continue
            if path.name == "test_repository.py":
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            if any(pattern.search(text) for pattern in patterns):
                violations.append(str(path.relative_to(ROOT)))
        self.assertEqual(violations, [])

    def test_runtime_has_no_weight_download_or_cache_fallback(self) -> None:
        text = (ROOT / "generator.py").read_text(encoding="utf-8")
        forbidden = (
            "snapshot_download",
            "hf_hub_download",
            "HUGGINGFACE_HUB_CACHE",
            "HF_HOME",
            "TRANSFORMERS_CACHE",
        )
        for token in forbidden:
            with self.subTest(token=token):
                self.assertNotIn(token, text)
        self.assertIn("local_files_only=True", text.replace(" ", ""))

    def test_setup_does_not_download_model_weights(self) -> None:
        text = (ROOT / "setup.py").read_text(encoding="utf-8")
        forbidden = (
            "snapshot_download",
            "hf_hub_download",
            "from_pretrained",
            "git clone",
            "huggingface.co/wgsxm",
            "wgsxm/PartCrafter-Scene",
        )
        for token in forbidden:
            with self.subTest(token=token):
                self.assertNotIn(token, text)

    def test_gitignore_excludes_generated_state(self) -> None:
        patterns = {
            line.strip()
            for line in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        for expected in ("venv", ".modly", "__pycache__", "*.py[cod]"):
            with self.subTest(pattern=expected):
                self.assertTrue(
                    any(line.rstrip("/") == expected.rstrip("/") for line in patterns),
                    expected,
                )


if __name__ == "__main__":
    unittest.main()
