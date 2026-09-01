from __future__ import annotations

import unittest

from tests._support import ROOT


class ContinuousIntegrationTests(unittest.TestCase):
    def test_ci_runs_stdlib_suite_on_linux_and_windows_python_311_and_312(self) -> None:
        workflow = (ROOT / ".github" / "workflows" / "tests.yml").read_text(
            encoding="utf-8"
        )
        for required in (
            "ubuntu-latest",
            "windows-latest",
            'python: ["3.11", "3.12"]',
            "python-version: ${{ matrix.python }}",
            "python -m unittest discover",
            "python -m compileall",
            "contents: read",
        ):
            with self.subTest(required=required):
                self.assertIn(required, workflow)
        self.assertNotIn("pip install", workflow)


if __name__ == "__main__":
    unittest.main()
