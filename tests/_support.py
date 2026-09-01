from __future__ import annotations

import importlib.util
import json
import struct
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


ROOT = Path(__file__).resolve().parents[1]


def read_json(relative_path: str) -> dict:
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {relative_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def minimal_glb() -> bytes:
    """Return a tiny, structurally valid GLB 2.0 document."""
    document = b'{"asset":{"version":"2.0"},"scene":0,"scenes":[{}]}'
    document += b" " * ((-len(document)) % 4)
    chunk = struct.pack("<I4s", len(document), b"JSON") + document
    return struct.pack("<4sII", b"glTF", 2, 12 + len(chunk)) + chunk


class FakeGenerationCancelled(Exception):
    pass


class FakeBaseGenerator:
    """The public surface used by the extension from Modly's BaseGenerator."""

    def __init__(self, model_dir: Path, outputs_dir: Path) -> None:
        self.model_dir = Path(model_dir)
        self.outputs_dir = Path(outputs_dir)
        self._model = None
        self.hf_repo = ""
        self.hf_skip_prefixes: list[str] = []
        self.download_check = ""
        self._params_schema: list[dict] = []

    def is_downloaded(self) -> bool:
        if self.download_check:
            return (self.model_dir / self.download_check).exists()
        return self.model_dir.exists() and any(self.model_dir.iterdir())

    def is_loaded(self) -> bool:
        return self._model is not None

    def _check_cancelled(self, cancel_event) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise FakeGenerationCancelled()

    def _report(self, progress_cb, pct: int, step: str) -> None:
        if progress_cb is not None:
            progress_cb(pct, step)


@contextmanager
def modly_api_fixture() -> Iterator[tuple[type, type]]:
    """Install a realistic minimal Modly module tree for generator imports."""
    names = (
        "services",
        "services.generators",
        "services.generators.base",
    )
    previous = {name: sys.modules.get(name) for name in names}
    services = types.ModuleType("services")
    generators = types.ModuleType("services.generators")
    base = types.ModuleType("services.generators.base")
    base.BaseGenerator = FakeBaseGenerator
    base.GenerationCancelled = FakeGenerationCancelled
    base.select_device = lambda: "cuda"
    base.select_dtype = lambda _device: "float16"
    sys.modules.update(
        {
            "services": services,
            "services.generators": generators,
            "services.generators.base": base,
        }
    )
    try:
        yield FakeBaseGenerator, FakeGenerationCancelled
    finally:
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
