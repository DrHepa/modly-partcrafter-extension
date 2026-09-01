from __future__ import annotations

import re
import unittest

from tests._support import ROOT


class DependencyContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        lines = (ROOT / "requirements-runtime.txt").read_text(encoding="utf-8").splitlines()
        cls.active_lines = [
            line
            for raw in lines
            if (line := raw.strip()) and not line.startswith("#")
        ]
        cls.requirements = {
            line.split("==", 1)[0].lower(): line.split("==", 1)[1]
            for line in cls.active_lines
            if "==" in line
        }

    def test_runtime_dependencies_are_exactly_pinned(self) -> None:
        exact_requirement = re.compile(
            r"^[A-Za-z0-9][A-Za-z0-9._-]*==[A-Za-z0-9][A-Za-z0-9.+_-]*$"
        )
        for line in self.active_lines:
            with self.subTest(requirement=line):
                self.assertRegex(line, exact_requirement)
        self.assertEqual(
            len(self.active_lines),
            len(self.requirements),
            "requirements must not contain duplicate package entries",
        )
        expected = {
            "diffusers": "0.34.0",
            "transformers": "4.53.0",
            "accelerate": "1.8.1",
            "huggingface-hub": "0.33.2",
            "hf-xet": "1.1.5",
            "safetensors": "0.5.3",
            "peft": "0.16.0",
            "einops": "0.8.1",
            "numpy": "1.26.4",
            "scipy": "1.15.3",
            "scikit-image": "0.25.2",
            "opencv-python-headless": "4.11.0.86",
            "trimesh": "4.6.13",
            "pyrender": "0.1.45",
            "pyopengl": "3.1.0",
            "pyglet": "1.5.31",
            "freetype-py": "2.5.1",
            "imageio": "2.37.0",
            "networkx": "3.4.2",
            "six": "1.17.0",
            "omegaconf": "2.3.0",
            "antlr4-python3-runtime": "4.9.3",
            "pillow": "11.2.1",
            "tqdm": "4.67.1",
            "google-genai": "1.73.1",
        }
        self.assertEqual(self.requirements, expected)
        for package, version in self.requirements.items():
            with self.subTest(package=package):
                self.assertRegex(version, r"^\d+(?:\.\d+)+(?:[A-Za-z0-9.+-]*)$")

    def test_torch_is_selected_by_setup_profile(self) -> None:
        self.assertNotIn("torch", self.requirements)
        self.assertNotIn("torchvision", self.requirements)
        setup_text = (ROOT / "setup.py").read_text(encoding="utf-8")
        self.assertIn("torch", setup_text)
        self.assertIn("torchvision", setup_text)
        self.assertIn("2.5.1", setup_text)
        self.assertIn("2.7.1", setup_text)

    def test_training_stacks_are_not_installed(self) -> None:
        excluded = {
            "deepspeed",
            "wandb",
            "xformers",
            "flash-attn",
            "kaolin",
            "open3d",
            "diso",
        }
        self.assertTrue(excluded.isdisjoint(self.requirements))

    def test_requirements_contains_no_mutable_sources(self) -> None:
        text = (ROOT / "requirements-runtime.txt").read_text(encoding="utf-8")
        for pattern in (r"\bgit\+", r"@[ \t]*(?:main|master)\b", r">=", r"~=", r"\*\s*$"):
            with self.subTest(pattern=pattern):
                self.assertIsNone(re.search(pattern, text, re.IGNORECASE | re.MULTILINE))


if __name__ == "__main__":
    unittest.main()
