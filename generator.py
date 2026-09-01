"""PartCrafter model adapter for Modly.

The Modly runner imports this module inside the extension virtual environment.
Protocol messages use stdout, so every adapter and upstream diagnostic is sent
to stderr. Model weights are always loaded from the exact ``model_dir`` passed
by Modly and are never fetched by this module.
"""

from __future__ import annotations

import contextlib
import gc
import hashlib
import importlib
import io
import json
import math
import os
import random
import re
import shutil
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from services.generators.base import BaseGenerator, GenerationCancelled


# The core pipeline must remain offline. Gemini is an explicit, optional
# upstream feature and uses Google's client, so these flags do not block it.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["DIFFUSERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("DO_NOT_TRACK", "1")


EXTENSION_ID = "partcrafter"
EXTENSION_DIR = Path(__file__).resolve().parent
UPSTREAM_ROOT = EXTENSION_DIR / ".modly" / "upstream" / "partcrafter"
UPSTREAM_SRC_DIR = UPSTREAM_ROOT / "src"
UPSTREAM_CODE_COMMIT = "3d773bf02fad51c7ab31a5615573fec93b287b30"

NODE_LIMITS = {"object": 16, "scene": 8}
NODE_REPOS = {
    "object": "wgsxm/PartCrafter",
    "scene": "wgsxm/PartCrafter-Scene",
    "rmbg": "briaai/RMBG-1.4",
}

RMBG_MODEL_SIZE = 176_381_984
RMBG_AUDITED_REVISION = "2ceba5a5efaec153162aedea169f76caf9b46cf8"
RMBG_MODEL_SHA256 = "46ef7fe46f2ae284d8f1aaa24bfa5fca5ef25a34e2c7caa890a0029eb100e87f"

# Small JSON/configuration files plus every inference weight file in both
# audited Hugging Face repositories. Minimum sizes reject Git-LFS pointer files
# and incomplete UI downloads without hashing almost 4 GiB on every load.
REQUIRED_MODEL_FILES: tuple[tuple[str, int], ...] = (
    ("model_index.json", 100),
    ("feature_extractor_dinov2/preprocessor_config.json", 100),
    ("image_encoder_dinov2/config.json", 100),
    ("image_encoder_dinov2/model.safetensors", 600_000_000),
    ("scheduler/scheduler_config.json", 50),
    ("transformer/config.json", 100),
    ("transformer/diffusion_pytorch_model.safetensors", 2_800_000_000),
    ("vae/config.json", 100),
    ("vae/diffusion_pytorch_model.safetensors", 480_000_000),
)

_MODEL_INDEX_COMPONENTS = {
    "feature_extractor_dinov2",
    "image_encoder_dinov2",
    "scheduler",
    "transformer",
    "vae",
}

def _log(message: str, *, level: str = "INFO") -> None:
    print(
        f"[PartCrafterGenerator] {level.upper()}: {message}",
        file=sys.stderr,
        flush=True,
    )


def _safe_error(exc: BaseException, *, limit: int = 1_200) -> str:
    """Return an actionable error string without leaking configured API keys."""

    text = str(exc).strip() or type(exc).__name__
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        secret = os.environ.get(name, "")
        if len(secret) >= 4:
            text = text.replace(secret, "[REDACTED]")
    text = " ".join(text.split())
    if len(text) > limit:
        text = text[: limit - 1] + "…"
    return text


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _model_validation_errors(model_dir: Path) -> list[str]:
    """Describe missing or incomplete files in one UI-managed model folder."""

    root = Path(model_dir)
    errors: list[str] = []
    if not root.is_dir():
        errors.append(f"model directory does not exist: {root}")

    for relative, minimum_size in REQUIRED_MODEL_FILES:
        path = root / relative
        if not path.is_file():
            errors.append(f"missing {relative}")
            continue
        try:
            size = path.stat().st_size
        except OSError as exc:
            errors.append(f"cannot inspect {relative}: {_safe_error(exc)}")
            continue
        if size < minimum_size:
            errors.append(
                f"incomplete {relative} ({size:,} bytes; expected at least "
                f"{minimum_size:,})"
            )
            continue
        if relative.endswith(".safetensors"):
            try:
                with path.open("rb") as stream:
                    header_size_raw = stream.read(8)
                    if len(header_size_raw) != 8:
                        raise ValueError("missing 8-byte header length")
                    header_size = int.from_bytes(header_size_raw, "little", signed=False)
                    if header_size < 2 or header_size > 100_000_000:
                        raise ValueError(f"invalid header length {header_size:,}")
                    if header_size + 8 >= size:
                        raise ValueError("header extends beyond tensor data")
                    header = json.loads(stream.read(header_size).decode("utf-8"))
                    if not isinstance(header, dict):
                        raise ValueError("header is not a JSON object")
            except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
                errors.append(f"invalid {relative}: {_safe_error(exc)}")

    index_path = root / "model_index.json"
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            if not isinstance(index, dict):
                raise ValueError("root must be a JSON object")
            if index.get("_class_name") != "PartCrafterPipeline":
                errors.append("model_index.json does not declare PartCrafterPipeline")
            missing_components = sorted(_MODEL_INDEX_COMPONENTS.difference(index))
            if missing_components:
                errors.append(
                    "model_index.json is missing components: "
                    + ", ".join(missing_components)
                )
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"invalid model_index.json: {_safe_error(exc)}")

    for relative, _minimum_size in REQUIRED_MODEL_FILES:
        if not relative.endswith(".json") or relative == "model_index.json":
            continue
        path = root / relative
        if not path.is_file():
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                errors.append(f"invalid {relative}: top level is not a JSON object")
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            errors.append(f"invalid {relative}: {_safe_error(exc)}")

    return errors


def _rmbg_validation_errors(model_dir: Path) -> list[str]:
    """Validate the two UI-managed files required by BRIA RMBG-1.4."""

    root = Path(model_dir)
    errors: list[str] = []
    if not root.is_dir():
        errors.append(f"model directory does not exist: {root}")

    config_path = root / "config.json"
    if not config_path.is_file():
        errors.append("missing config.json")
    else:
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            if not isinstance(config, dict):
                raise ValueError("top level is not a JSON object")
            if config.get("in_ch") != 3 or config.get("out_ch") != 1:
                raise ValueError("expected in_ch=3 and out_ch=1")
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"invalid config.json: {_safe_error(exc)}")

    weights_path = root / "model.safetensors"
    if not weights_path.is_file():
        errors.append("missing model.safetensors")
    else:
        try:
            size = weights_path.stat().st_size
            if size != RMBG_MODEL_SIZE:
                raise ValueError(
                    f"{size:,} bytes; expected audited RMBG-1.4 file size "
                    f"{RMBG_MODEL_SIZE:,}"
                )
            with weights_path.open("rb") as stream:
                raw_header_size = stream.read(8)
                if len(raw_header_size) != 8:
                    raise ValueError("missing 8-byte header length")
                header_size = int.from_bytes(raw_header_size, "little", signed=False)
                if header_size < 2 or header_size > 100_000_000:
                    raise ValueError(f"invalid header length {header_size:,}")
                if header_size + 8 >= size:
                    raise ValueError("header extends beyond tensor data")
                header = json.loads(stream.read(header_size).decode("utf-8"))
                if not isinstance(header, dict):
                    raise ValueError("header is not a JSON object")
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"invalid model.safetensors: {_safe_error(exc)}")

    return errors


def _require_upstream_source(source_root: Path = UPSTREAM_ROOT) -> Path:
    """Validate the immutable source prepared by setup.py and return its root."""

    source_root = Path(source_root)
    required = (
        source_root / "src" / "pipelines" / "pipeline_partcrafter.py",
        source_root / "src" / "models" / "transformers" / "partcrafter_transformer.py",
        source_root / "src" / "models" / "autoencoders" / "autoencoder_kl_triposg.py",
        source_root / "src" / "models" / "briarmbg.py",
        source_root / "src" / "utils" / "inference_utils.py",
        source_root / "src" / "utils" / "image_utils.py",
        source_root / "src" / "utils" / "render_utils.py",
        source_root / "src" / "utils" / "vlm_utils.py",
        source_root / "src" / "utils" / "style_transfer_utils.py",
        source_root / "src" / "utils" / "providers" / "gemini_provider.py",
        source_root / "src" / "schedulers" / "scheduling_rectified_flow.py",
    )
    missing = [str(path.relative_to(source_root)) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(
            "PartCrafter source is not prepared (missing "
            + ", ".join(missing)
            + "). Open Modly Models and run Repair for this extension."
        )
    return source_root


def _load_pipeline_class(source_root: Path = UPSTREAM_ROOT):
    """Import PartCrafterPipeline only from setup.py's pinned source tree."""

    source_root = _require_upstream_source(source_root).resolve()
    root_text = str(source_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)

    from src.pipelines.pipeline_partcrafter import PartCrafterPipeline

    module_path = Path(sys.modules[PartCrafterPipeline.__module__].__file__).resolve()
    if not _is_relative_to(module_path, source_root):
        raise RuntimeError(
            "PartCrafter imported from an unexpected location. Run Repair so the "
            "extension can use its pinned upstream source."
        )
    return PartCrafterPipeline


def _load_rmbg_class(source_root: Path = UPSTREAM_ROOT):
    """Import the exact RMBG architecture shipped in pinned PartCrafter source."""

    source_root = _require_upstream_source(source_root).resolve()
    root_text = str(source_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)

    from src.models.briarmbg import BriaRMBG

    module_path = Path(sys.modules[BriaRMBG.__module__].__file__).resolve()
    if not _is_relative_to(module_path, source_root):
        raise RuntimeError(
            "BRIA RMBG imported from an unexpected location. Run Repair so the "
            "extension can use its pinned PartCrafter source."
        )
    return BriaRMBG


def _parse_bool_select(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    raise ValueError(f"{name} must be 'true' or 'false'")


def _parse_int(
    value: Any,
    name: str,
    minimum: int,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{name} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if parsed < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and parsed > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return parsed


def _parse_float(
    value: Any,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and parsed < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and parsed > maximum:
        if minimum is None:
            raise ValueError(f"{name} must be at most {maximum}")
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return parsed


def _parse_model_name(value: Any, name: str, default: str) -> str:
    parsed = str(value if value is not None else default).strip()
    if not parsed:
        parsed = default
    if len(parsed) > 200 or any(char in parsed for char in "\r\n\0"):
        raise ValueError(f"{name} is invalid")
    return parsed


def _parse_params(params: Mapping[str, Any] | None, node_id: str) -> dict[str, Any]:
    """Normalize and range-check every parameter exposed by the manifest."""

    if node_id not in NODE_LIMITS:
        raise ValueError(f"unsupported PartCrafter node: {node_id}")
    params = params if isinstance(params, Mapping) else {}
    if node_id == "object":
        raw_num_parts = params.get("num_parts", 3)
        raw_num_tokens = params.get("num_tokens", 1024)
        default_num_parts = 3
    else:
        # Scene uses distinct manifest IDs because Modly's classic Generate UI
        # reads params_schema.default directly and does not apply param_defaults.
        # Both values are mapped back to PartCrafter's canonical arguments below.
        raw_num_parts = params.get("scene_num_parts", 6)
        raw_num_tokens = params.get("scene_num_tokens", 2048)
        default_num_parts = 6

    part_count_mode = str(params.get("part_count_mode", "manual")).strip().lower()
    if part_count_mode not in {"manual", "gemini"}:
        raise ValueError("part_count_mode must be 'manual' or 'gemini'")

    if part_count_mode == "manual":
        num_parts = _parse_int(
            raw_num_parts,
            "num_parts",
            1,
            NODE_LIMITS[node_id],
        )
        part_model = "gemini-3-flash-preview"
    else:
        # num_parts is hidden in the workflow for this mode. Ignore a stale
        # manual value rather than rejecting a valid Gemini request.
        num_parts = default_num_parts
        part_model = _parse_model_name(
            params.get("part_model", "gemini-3-flash-preview"),
            "part_model",
            "gemini-3-flash-preview",
        )

    style_transfer = _parse_bool_select(
        params.get("style_transfer", "false"), "style_transfer"
    )
    render = _parse_bool_select(params.get("render", "false"), "render")
    style_model = (
        _parse_model_name(
            params.get("style_model", "gemini-3.1-flash-image-preview"),
            "style_model",
            "gemini-3.1-flash-image-preview",
        )
        if style_transfer
        else "gemini-3.1-flash-image-preview"
    )

    output_name = str(params.get("output_name", "") or "").strip()
    if any(char in output_name for char in "\r\n\0"):
        raise ValueError("output_name contains invalid characters")

    return {
        "part_count_mode": part_count_mode,
        "num_parts": num_parts,
        "part_model": part_model,
        "style_transfer": style_transfer,
        "remove_background": (
            _parse_bool_select(
                params.get("remove_background", "false"), "remove_background"
            )
            if node_id == "object"
            else False
        ),
        "render": render,
        "style_model": style_model,
        "num_tokens": _parse_int(
            raw_num_tokens,
            "num_tokens",
            1,
        ),
        "num_inference_steps": _parse_int(
            params.get("num_inference_steps", 50),
            "num_inference_steps",
            1,
        ),
        "guidance_scale": _parse_float(
            params.get("guidance_scale", 7.0),
            "guidance_scale",
        ),
        "max_num_expanded_coords": _parse_int(
            params.get("max_num_expanded_coords", 1_000_000_000),
            "max_num_expanded_coords",
            0,
        ),
        "use_flash_decoder": _parse_bool_select(
            params.get("use_flash_decoder", "false"), "use_flash_decoder"
        ),
        "seed": _parse_int(
            params.get("seed", 0),
            "seed",
            -(2**63),
            2**64 - 1,
        ),
        "output_name": output_name,
    }


def _safe_output_stem(value: str, fallback: str = "partcrafter") -> str:
    value = value.strip() if value else fallback
    if value.lower().endswith(".glb"):
        value = value[:-4]
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("._-")
    value = (value or fallback)[:80]
    windows_reserved = {"CON", "PRN", "AUX", "NUL"}
    windows_reserved.update(f"COM{index}" for index in range(1, 10))
    windows_reserved.update(f"LPT{index}" for index in range(1, 10))
    if value.upper() in windows_reserved:
        value = f"partcrafter-{value}"
    return value


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _atomic_write_bytes(path, payload)


def _require_gemini_key(feature: str) -> None:
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        raise RuntimeError(
            f"Gemini {feature} requires GEMINI_API_KEY or GOOGLE_API_KEY in the "
            "Modly environment."
        )


@contextlib.contextmanager
def _upstream_output_to_stderr():
    """Protect Modly's JSON stdout protocol from upstream print calls."""

    with contextlib.redirect_stdout(sys.stderr):
        yield


def _decode_input_image(image_bytes: bytes):
    if not image_bytes:
        raise ValueError("input image is empty")
    try:
        from PIL import Image, ImageOps

        with Image.open(io.BytesIO(image_bytes)) as source:
            source.load()
            image = ImageOps.exif_transpose(source)
            if image.mode in {"RGBA", "LA"} or (
                image.mode == "P" and "transparency" in image.info
            ):
                rgba = image.convert("RGBA")
                background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
                image = Image.alpha_composite(background, rgba).convert("RGB")
            else:
                image = image.convert("RGB")
            if image.width < 2 or image.height < 2:
                raise ValueError("input image must be at least 2 x 2 pixels")
            return image.copy()
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"input is not a readable image: {_safe_error(exc)}") from exc


def _decode_rmbg_input_image(image_bytes: bytes):
    """Decode an RMBG input without flattening a useful source alpha channel."""

    if not image_bytes:
        raise ValueError("input image is empty")
    try:
        from PIL import Image, ImageOps

        with Image.open(io.BytesIO(image_bytes)) as source:
            source.load()
            image = ImageOps.exif_transpose(source)
            if image.mode in {"RGBA", "LA"} or (
                image.mode == "P" and "transparency" in image.info
            ):
                image = image.convert("RGBA")
            else:
                image = image.convert("RGB")
            if image.width < 2 or image.height < 2:
                raise ValueError("input image must be at least 2 x 2 pixels")
            return image.copy()
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"input is not a readable image: {_safe_error(exc)}") from exc


def _png_bytes(image: Any) -> bytes:
    """Serialize a Pillow image as a complete PNG before atomic publication."""

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    payload = buffer.getvalue()
    if len(payload) < 12 or not payload.startswith(b"\x89PNG\r\n\x1a\n"):
        raise RuntimeError("image preprocessing did not produce a valid PNG")
    return payload


def _open_rgb_image(path: Path):
    """Open an image and composite any transparency over the model's white background."""

    from PIL import Image, ImageOps

    with Image.open(path) as source:
        source.load()
        image = ImageOps.exif_transpose(source)
        if image.mode in {"RGBA", "LA"} or (
            image.mode == "P" and "transparency" in image.info
        ):
            rgba = image.convert("RGBA")
            background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, rgba).convert("RGB")
        else:
            image = image.convert("RGB")
        return image.copy()


def _validate_meshes(meshes: Any, expected_count: int) -> list[Any]:
    """Reject failed, empty, invalid, or degenerate decoder outputs."""

    try:
        import numpy as np
        import trimesh
    except ImportError as exc:
        raise RuntimeError(
            "PartCrafter export dependencies are missing. Run Repair in Modly Models."
        ) from exc

    if not isinstance(meshes, (list, tuple)):
        raise RuntimeError("PartCrafter returned no mesh list")
    if len(meshes) != expected_count:
        raise RuntimeError(
            f"PartCrafter returned {len(meshes)} meshes; expected {expected_count}"
        )

    validated: list[Any] = []
    for index, mesh in enumerate(meshes):
        if mesh is None:
            raise RuntimeError(
                f"decoder failed for part {index:02d}; no dummy geometry was written"
            )
        if not isinstance(mesh, trimesh.Trimesh):
            raise RuntimeError(
                f"decoder returned {type(mesh).__name__} for part {index:02d}, "
                "expected Trimesh"
            )
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.faces)
        if vertices.ndim != 2 or vertices.shape[0] < 3 or vertices.shape[1] != 3:
            raise RuntimeError(f"part {index:02d} has no usable vertices")
        if faces.ndim != 2 or faces.shape[0] < 1 or faces.shape[1] != 3:
            raise RuntimeError(f"part {index:02d} has no triangular faces")
        if not np.isfinite(vertices).all():
            raise RuntimeError(f"part {index:02d} contains non-finite vertices")
        if faces.min() < 0 or faces.max() >= len(vertices):
            raise RuntimeError(f"part {index:02d} contains invalid face indices")
        extents = np.ptp(vertices, axis=0)
        if not np.isfinite(extents).all() or float(extents.max()) <= 1e-8:
            raise RuntimeError(f"part {index:02d} has degenerate bounds")
        area = float(mesh.area)
        if not math.isfinite(area) or area <= 1e-12:
            raise RuntimeError(f"part {index:02d} has degenerate surface area")
        validated.append(mesh)
    return validated


def _glb_bytes(value: Any) -> bytes:
    payload = value.export(file_type="glb")
    if not isinstance(payload, (bytes, bytearray)):
        raise RuntimeError("trimesh did not return a binary GLB payload")
    payload = bytes(payload)
    if len(payload) < 20 or payload[:4] != b"glTF":
        raise RuntimeError("trimesh produced an invalid GLB payload")
    return payload


def _build_colored_scene(meshes: Sequence[Any]):
    import numpy as np
    import trimesh

    scene = trimesh.Scene()
    for index, mesh in enumerate(meshes):
        name = f"part_{index:02d}"
        colored = mesh.copy()
        # Exact upstream get_colored_mesh_composition default: one random RGB
        # triplet per part after the generation seed has initialized NumPy.
        color = (np.random.rand(3) * 256).astype(int)
        colored.visual = trimesh.visual.ColorVisuals(
            mesh=colored,
            vertex_colors=color,
        )
        scene.add_geometry(colored, node_name=name, geom_name=name)
    return scene


@contextlib.contextmanager
def _render_platform_scope():
    """Keep the correct PyOpenGL selector active through import and rendering."""

    is_linux = sys.platform.startswith("linux")
    is_windows = os.name == "nt" or sys.platform.startswith("win")
    previous_platform = os.environ.get("PYOPENGL_PLATFORM")
    if is_linux:
        os.environ["PYOPENGL_PLATFORM"] = "egl"
    elif is_windows:
        os.environ.pop("PYOPENGL_PLATFORM", None)
    try:
        yield
    finally:
        if previous_platform is None:
            os.environ.pop("PYOPENGL_PLATFORM", None)
        else:
            os.environ["PYOPENGL_PLATFORM"] = previous_platform


def _load_render_utils():
    """Import upstream rendering with the platform selected before PyOpenGL.

    PyOpenGL chooses its backend while ``pyrender`` is imported. Linux's
    headless route therefore has to select EGL first. Windows uses pyrender's
    normal desktop backend; upstream's late, unconditional EGL environment
    assignment is restored after import so it cannot affect later imports.
    """

    with _render_platform_scope():
        with _upstream_output_to_stderr():
            module = importlib.import_module("src.utils.render_utils")

    required = (
        "render_views_around_mesh",
        "render_normal_views_around_mesh",
        "make_grid_for_images_or_videos",
        "export_renderings",
    )
    missing = [name for name in required if not callable(getattr(module, name, None))]
    if missing:
        raise RuntimeError(
            "upstream render_utils is missing callables: " + ", ".join(missing)
        )
    return module


def _render_failure_guidance(exc: BaseException) -> RuntimeError:
    detail = _safe_error(exc)
    if sys.platform.startswith("linux"):
        action = (
            "Install the system EGL/OpenGL libraries (on Debian/Ubuntu: "
            "libegl1 and libgl1), restart Modly, and run Repair"
        )
    elif os.name == "nt" or sys.platform.startswith("win"):
        action = (
            "Update the NVIDIA/OpenGL driver, restart Modly, and run Repair"
        )
    else:
        action = "Run Repair and verify that an OpenGL offscreen context is available"
    return RuntimeError(f"PartCrafter turntable rendering failed: {detail}. {action}.")


def _render_sidecars(
    scene: Any,
    processed_image: Any,
    output_dir: Path,
) -> dict[str, str]:
    """Run upstream ``--render`` post-processing with its exact CLI values."""

    try:
        with _render_platform_scope():
            render_utils = _load_render_utils()
            num_views = 36
            radius = 4
            fps = 18
            with _upstream_output_to_stderr():
                rendered_images = render_utils.render_views_around_mesh(
                    scene,
                    num_views=num_views,
                    radius=radius,
                )
                rendered_normals = render_utils.render_normal_views_around_mesh(
                    scene,
                    num_views=num_views,
                    radius=radius,
                )
                if (
                    len(rendered_images) != num_views
                    or len(rendered_normals) != num_views
                ):
                    raise RuntimeError(
                        "upstream renderer did not return the expected 36 color "
                        "and normal views"
                    )
                rendered_grids = render_utils.make_grid_for_images_or_videos(
                    [
                        [processed_image] * num_views,
                        rendered_images,
                        rendered_normals,
                    ],
                    nrow=3,
                )
                if len(rendered_grids) != num_views:
                    raise RuntimeError(
                        "upstream renderer did not return the expected 36 grid views"
                    )
                render_utils.export_renderings(
                    rendered_images,
                    str(output_dir / "rendering.gif"),
                    fps=fps,
                )
                render_utils.export_renderings(
                    rendered_normals,
                    str(output_dir / "rendering_normal.gif"),
                    fps=fps,
                )
                render_utils.export_renderings(
                    rendered_grids,
                    str(output_dir / "rendering_grid.gif"),
                    fps=fps,
                )
                rendered_images[0].save(output_dir / "rendering.png")
                rendered_normals[0].save(output_dir / "rendering_normal.png")
                rendered_grids[0].save(output_dir / "rendering_grid.png")

        names = (
            "rendering.gif",
            "rendering_normal.gif",
            "rendering_grid.gif",
            "rendering.png",
            "rendering_normal.png",
            "rendering_grid.png",
        )
        missing = [
            name
            for name in names
            if not (output_dir / name).is_file()
            or (output_dir / name).stat().st_size == 0
        ]
        if missing:
            raise RuntimeError(
                "upstream renderer did not create valid files: " + ", ".join(missing)
            )
        return {
            "color_gif": "rendering.gif",
            "normal_gif": "rendering_normal.gif",
            "grid_gif": "rendering_grid.gif",
            "color_png": "rendering.png",
            "normal_png": "rendering_normal.png",
            "grid_png": "rendering_grid.png",
        }
    except Exception as exc:
        if isinstance(exc, RuntimeError) and str(exc).startswith(
            "PartCrafter turntable rendering failed:"
        ):
            raise
        raise _render_failure_guidance(exc) from exc


def _validate_composite_glb(payload: bytes, expected_parts: int) -> None:
    import trimesh

    if len(payload) < 20 or payload[:4] != b"glTF":
        raise RuntimeError("composite export is not a GLB 2.0 file")
    try:
        loaded = trimesh.load(
            io.BytesIO(payload), file_type="glb", force="scene", process=False
        )
    except Exception as exc:
        raise RuntimeError(f"composite GLB cannot be reopened: {_safe_error(exc)}") from exc
    if not isinstance(loaded, trimesh.Scene):
        raise RuntimeError("composite GLB did not reopen as a scene")
    names = set(loaded.geometry)
    expected_names = {f"part_{index:02d}" for index in range(expected_parts)}
    if names != expected_names:
        raise RuntimeError(
            "composite GLB did not preserve the expected named part geometries"
        )


class PartCrafterGenerator(BaseGenerator):
    """PartCrafter object, scene, and upstream RMBG adapter for Modly."""

    MODEL_ID = EXTENSION_ID
    DISPLAY_NAME = "PartCrafter"
    VRAM_GB = 8

    def __init__(self, model_dir: Path, outputs_dir: Path) -> None:
        super().__init__(Path(model_dir), Path(outputs_dir))
        self.model_dir = Path(model_dir)
        self.outputs_dir = Path(outputs_dir)
        self._node_id = self.model_dir.name.lower()
        self._torch = None
        self._rmbg_model = None
        self._load_lock = threading.RLock()

    def _node(self) -> str:
        if self._node_id not in NODE_REPOS:
            raise RuntimeError(
                f"PartCrafter model_dir must end in 'object', 'scene', or 'rmbg'; got "
                f"{self.model_dir}"
            )
        return self._node_id

    def is_downloaded(self) -> bool:
        if self._node() == "rmbg":
            return not _rmbg_validation_errors(self.model_dir)
        return not _model_validation_errors(self.model_dir)

    def _auto_download(self) -> None:
        raise RuntimeError(
            "PartCrafter weights are managed by Modly. Open Models, choose the "
            "PartCrafter node, and use Download; generation never downloads weights."
        )

    def load(self) -> None:
        with self._load_lock:
            if self._model is not None:
                return

            node_id = self._node()
            if node_id == "rmbg":
                try:
                    self._model = self._load_rmbg_model(self.model_dir)
                    _log("PartCrafter RMBG model loaded")
                    return
                except Exception as exc:
                    self._model = None
                    self._torch = None
                    message = _safe_error(exc)
                    _log(f"RMBG load failed: {message}", level="ERROR")
                    if isinstance(exc, RuntimeError) and message.startswith("RMBG"):
                        raise
                    raise RuntimeError(
                        f"Failed to load RMBG from {self.model_dir}: {message}. "
                        "Run Repair and inspect .modly/setup/logs/setup.log."
                    ) from None

            errors = _model_validation_errors(self.model_dir)
            if errors:
                detail = "; ".join(errors[:5])
                if len(errors) > 5:
                    detail += f"; and {len(errors) - 5} more"
                message = (
                    f"Weights for {NODE_REPOS[node_id]} are incomplete in the exact "
                    f"Modly model directory {self.model_dir}: {detail}. Open the "
                    "Models UI and use Download, or run Repair if setup is incomplete."
                )
                _log(message, level="ERROR")
                raise RuntimeError(message)

            _log(f"Loading {node_id} weights from Modly model directory: {self.model_dir}")
            try:
                _require_upstream_source(UPSTREAM_ROOT)
                import torch

                if not torch.cuda.is_available():
                    raise RuntimeError(
                        "PartCrafter inference requires an NVIDIA CUDA GPU; CPU and "
                        "Apple MPS are not supported by the upstream decoder."
                    )
                pipeline_class = _load_pipeline_class(UPSTREAM_ROOT)
                with _upstream_output_to_stderr():
                    pipeline = pipeline_class.from_pretrained(
                        str(self.model_dir),
                        torch_dtype=torch.float16,
                        local_files_only=True,
                    )
                    pipeline = pipeline.to(device="cuda", dtype=torch.float16)
                    if hasattr(pipeline, "set_progress_bar_config"):
                        pipeline.set_progress_bar_config(disable=True)

                required_components = (
                    "vae",
                    "transformer",
                    "scheduler",
                    "image_encoder_dinov2",
                    "feature_extractor_dinov2",
                )
                missing = [name for name in required_components if not hasattr(pipeline, name)]
                if missing:
                    raise RuntimeError(
                        "loaded pipeline is missing components: " + ", ".join(missing)
                    )

                try:
                    total_vram = torch.cuda.get_device_properties(0).total_memory / 2**30
                    if total_vram < self.VRAM_GB:
                        _log(
                            f"CUDA device reports {total_vram:.1f} GiB VRAM; upstream "
                            f"recommends at least {self.VRAM_GB} GiB.",
                            level="WARNING",
                        )
                except Exception as exc:
                    _log(
                        f"Could not read CUDA memory capacity: {_safe_error(exc)}",
                        level="WARNING",
                    )

                self._torch = torch
                self._model = pipeline
                _log(f"PartCrafter {node_id} pipeline loaded")
            except Exception as exc:
                self._model = None
                self._torch = None
                message = _safe_error(exc)
                _log(f"Load failed: {message}", level="ERROR")
                if isinstance(exc, RuntimeError) and message.startswith("PartCrafter"):
                    raise
                raise RuntimeError(
                    f"Failed to load PartCrafter {node_id} from {self.model_dir}: "
                    f"{message}. Run Repair and inspect .modly/setup/logs/setup.log."
                ) from None

    def _load_rmbg_model(self, weights_dir: Path):
        """Load BRIA RMBG-1.4 from one exact Modly-managed local directory."""

        weights_dir = Path(weights_dir)
        errors = _rmbg_validation_errors(weights_dir)
        if errors:
            detail = "; ".join(errors[:5])
            message = (
                f"RMBG weights for {NODE_REPOS['rmbg']} are incomplete in the "
                f"exact Modly model directory {weights_dir}: {detail}. Open the "
                "Models UI and Download the PartCrafter RMBG Preprocess node."
            )
            raise RuntimeError(message)

        _require_upstream_source(UPSTREAM_ROOT)
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "RMBG inference requires an NVIDIA CUDA GPU in this PartCrafter "
                "extension; CPU and Apple MPS are unsupported."
            )

        rmbg_class = _load_rmbg_class(UPSTREAM_ROOT)
        with _upstream_output_to_stderr():
            model = rmbg_class.from_pretrained(
                str(weights_dir),
                local_files_only=True,
            )
            model = model.to(device="cuda")
            model.eval()
        self._torch = torch
        return model

    def _get_auxiliary_rmbg_model(self):
        """Lazily load and retain the sibling UI-managed RMBG model for Object."""

        with self._load_lock:
            if self._rmbg_model is not None:
                return self._rmbg_model
            weights_dir = self.model_dir.parent / "rmbg"
            try:
                self._rmbg_model = self._load_rmbg_model(weights_dir)
            except Exception as exc:
                message = _safe_error(exc)
                raise RuntimeError(
                    f"Object background removal requires the sibling RMBG weights "
                    f"at {weights_dir}. Download the PartCrafter RMBG Preprocess "
                    f"node in Modly's Models UI before enabling Remove Background: "
                    f"{message}"
                ) from exc
            _log(f"Auxiliary RMBG model loaded from {weights_dir}")
            return self._rmbg_model

    def _prepare_with_rmbg(self, image_path: Path, model: Any):
        """Call PartCrafter's exact public RMBG preprocessing function."""

        import numpy as np
        from PIL import Image
        from src.utils.image_utils import prepare_image

        torch = self._torch
        if torch is None:
            raise RuntimeError("RMBG torch runtime is not loaded")
        with torch.inference_mode(), _upstream_output_to_stderr():
            prepared = prepare_image(
                str(image_path),
                bg_color=np.array([1.0, 1.0, 1.0]),
                rmbg_net=model,
                padding_ratio=0.1,
                device="cuda",
            )
        if not isinstance(prepared, Image.Image):
            raise RuntimeError(
                f"upstream prepare_image returned {type(prepared).__name__}, "
                "expected a Pillow image"
            )
        if prepared.width < 2 or prepared.height < 2:
            raise RuntimeError("upstream prepare_image returned an unusable image")
        return prepared

    def _generate_rmbg(
        self,
        image_bytes: bytes,
        progress_cb: Optional[Callable[[int, str], None]],
        cancel_event: Optional[threading.Event],
    ) -> Path:
        """Run the standalone workflow-only image-to-image RMBG node."""

        self._report(progress_cb, 1, "Validating RMBG input")
        if self._model is None:
            self._report(progress_cb, 5, "Loading RMBG-1.4")
            self.load()
        self._check_cancelled(cancel_event)

        outputs_root = self.outputs_dir.resolve()
        temporary_input: Path | None = None
        try:
            outputs_root.mkdir(parents=True, exist_ok=True)
            suffix = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
            temporary_input = outputs_root / f".partcrafter-rmbg-{suffix}.input.png"
            output_path = outputs_root / f"partcrafter-rmbg-{suffix}.png"

            source = _decode_rmbg_input_image(image_bytes)
            _atomic_write_bytes(temporary_input, _png_bytes(source))
            self._report(progress_cb, 20, "RMBG input prepared")
            self._check_cancelled(cancel_event)

            self._report(progress_cb, 35, "Removing background with RMBG-1.4")
            prepared = self._prepare_with_rmbg(temporary_input, self._model)
            self._check_cancelled(cancel_event)
            _atomic_write_bytes(output_path, _png_bytes(prepared))
            if not output_path.is_absolute() or not output_path.is_file():
                raise RuntimeError("final RMBG PNG path was not committed")
            self._report(progress_cb, 100, "RMBG preprocessing complete")
            _log(f"Generated RMBG PNG: {output_path}")
            return output_path
        except GenerationCancelled:
            _log("RMBG generation cancelled", level="WARNING")
            raise
        except Exception as exc:
            message = _safe_error(exc)
            _log(f"RMBG generation failed: {message}", level="ERROR")
            raise RuntimeError(f"PartCrafter RMBG generation failed: {message}") from None
        finally:
            if temporary_input is not None:
                temporary_input.unlink(missing_ok=True)

    def _style_input(
        self,
        input_path: Path,
        styled_path: Path,
        model_name: str,
    ) -> tuple[Path, str | None]:
        """Run upstream style transfer, preserving its original-image fallback."""

        try:
            _require_gemini_key("style transfer")
            from src.utils.style_transfer_utils import stylize_for_objaverse

            with _upstream_output_to_stderr():
                stylize_for_objaverse(
                    str(input_path),
                    str(styled_path),
                    provider="gemini",
                    model_name=model_name,
                )
            if not styled_path.is_file() or styled_path.stat().st_size == 0:
                raise RuntimeError("Gemini returned no styled image file")
            _log(f"Gemini style transfer completed with model {model_name}")
            return styled_path, None
        except Exception as exc:
            message = _safe_error(exc)
            _log(
                f"Gemini style transfer failed; using the original image: {message}",
                level="WARNING",
            )
            styled_path.unlink(missing_ok=True)
            return input_path, message

    def _resolve_part_count(
        self,
        image_path: Path,
        settings: Mapping[str, Any],
        node_id: str,
    ) -> int:
        if settings["part_count_mode"] == "manual":
            return int(settings["num_parts"])

        _require_gemini_key("part suggestion")
        try:
            from src.utils.vlm_utils import suggest_num_parts

            with _upstream_output_to_stderr():
                count = suggest_num_parts(
                    str(image_path),
                    NODE_LIMITS[node_id],
                    mode=node_id,
                    provider="gemini",
                    model_name=settings["part_model"],
                )
        except Exception as exc:
            raise RuntimeError(
                f"Gemini part suggestion failed: {_safe_error(exc)}"
            ) from None
        count = _parse_int(count, "Gemini part count", 1, NODE_LIMITS[node_id])
        _log(
            f"Gemini model {settings['part_model']} suggested {count} "
            f"{'parts' if node_id == 'object' else 'objects'}"
        )
        return count

    def _run_pipeline(
        self,
        image: Any,
        settings: Mapping[str, Any],
        num_parts: int,
        progress_cb: Optional[Callable[[int, str], None]],
        cancel_event: Optional[threading.Event],
    ) -> list[Any]:
        torch = self._torch
        pipeline = self._model
        if torch is None or pipeline is None:
            raise RuntimeError("PartCrafter pipeline is not loaded")

        total_steps = int(settings["num_inference_steps"])
        last_progress = 20

        def callback_on_step_end(
            _pipeline: Any,
            step_index: int,
            _timestep: Any,
            callback_kwargs: dict[str, Any],
        ) -> dict[str, Any]:
            nonlocal last_progress
            self._check_cancelled(cancel_event)
            pct = 20 + int(55 * (step_index + 1) / max(total_steps, 1))
            pct = max(last_progress, min(75, pct))
            last_progress = pct
            self._report(
                progress_cb,
                pct,
                f"Denoising {step_index + 1}/{total_steps}",
            )
            return callback_kwargs

        generator = torch.Generator(device="cuda").manual_seed(int(settings["seed"]))
        seed = int(settings["seed"])
        random.seed(seed)
        import numpy as np

        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        if hasattr(torch.cuda, "manual_seed_all"):
            torch.cuda.manual_seed_all(seed)

        with torch.inference_mode(), _upstream_output_to_stderr():
            result = pipeline(
                image=[image] * num_parts,
                attention_kwargs={"num_parts": num_parts},
                num_tokens=int(settings["num_tokens"]),
                generator=generator,
                num_inference_steps=total_steps,
                guidance_scale=float(settings["guidance_scale"]),
                max_num_expanded_coords=int(settings["max_num_expanded_coords"]),
                use_flash_decoder=bool(settings["use_flash_decoder"]),
                callback_on_step_end=callback_on_step_end,
                callback_on_step_end_tensor_inputs=["latents"],
            )

        meshes = getattr(result, "meshes", None)
        return _validate_meshes(meshes, num_parts)

    def generate(
        self,
        image_bytes: bytes,
        params: dict,
        progress_cb: Optional[Callable[[int, str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Path:
        self._check_cancelled(cancel_event)
        try:
            node_id = self._node()
        except Exception as exc:
            message = _safe_error(exc)
            _log(f"Generation parameter validation failed: {message}", level="ERROR")
            raise RuntimeError(
                f"PartCrafter generation parameter validation failed: {message}"
            ) from None
        if node_id == "rmbg":
            return self._generate_rmbg(image_bytes, progress_cb, cancel_event)
        try:
            settings = _parse_params(params, node_id)
        except Exception as exc:
            message = _safe_error(exc)
            _log(f"Generation parameter validation failed: {message}", level="ERROR")
            raise RuntimeError(
                f"PartCrafter generation parameter validation failed: {message}"
            ) from None
        self._report(progress_cb, 1, "Validating PartCrafter input")

        if self._model is None:
            self._report(progress_cb, 3, "Loading PartCrafter")
            self.load()
        self._check_cancelled(cancel_event)

        try:
            outputs_root = self.outputs_dir.resolve()
            outputs_root.mkdir(parents=True, exist_ok=True)
            fallback_stem = f"partcrafter-{node_id}"
            output_stem = _safe_output_stem(settings["output_name"], fallback_stem)
            run_suffix = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
            final_dir = outputs_root / f"{output_stem}-{run_suffix}"
            temporary_dir = outputs_root / f".{output_stem}-{run_suffix}.tmp"
            temporary_dir.mkdir(parents=False, exist_ok=False)
        except Exception as exc:
            message = _safe_error(exc)
            _log(f"Output initialization failed: {message}", level="ERROR")
            raise RuntimeError(
                f"PartCrafter could not initialize its Modly output directory: {message}"
            ) from None

        stage = "input preparation"
        started = time.time()
        try:
            image = (
                _decode_rmbg_input_image(image_bytes)
                if settings["remove_background"]
                else _decode_input_image(image_bytes)
            )
            input_sha256 = hashlib.sha256(image_bytes).hexdigest()
            input_path = temporary_dir / ".input.png"
            image.save(input_path, format="PNG")
            self._report(progress_cb, 8, "Input image prepared")
            self._check_cancelled(cancel_event)

            style_error: str | None = None
            inference_path = input_path
            if settings["style_transfer"]:
                stage = "Gemini style transfer"
                self._report(progress_cb, 10, "Applying Gemini style transfer")
                inference_path, style_error = self._style_input(
                    input_path,
                    temporary_dir / "styled_input.png",
                    settings["style_model"],
                )
                self._check_cancelled(cancel_event)

            stage = "part count selection"
            self._report(progress_cb, 14, "Selecting part count")
            num_parts = self._resolve_part_count(inference_path, settings, node_id)
            self._check_cancelled(cancel_event)

            if settings["remove_background"]:
                stage = "background removal"
                self._report(progress_cb, 17, "Removing input background")
                inference_image = self._prepare_with_rmbg(
                    inference_path,
                    self._get_auxiliary_rmbg_model(),
                )
                self._check_cancelled(cancel_event)
            else:
                inference_image = _open_rgb_image(inference_path)

            stage = "diffusion and mesh decoding"
            self._report(
                progress_cb,
                20,
                f"Generating {num_parts} "
                f"{'parts' if node_id == 'object' else 'objects'}",
            )
            meshes = self._run_pipeline(
                inference_image,
                settings,
                num_parts,
                progress_cb,
                cancel_event,
            )
            self._check_cancelled(cancel_event)
            self._report(progress_cb, 80, "Validating decoded meshes")

            stage = "part export"
            part_records: list[dict[str, Any]] = []
            for index, mesh in enumerate(meshes):
                self._check_cancelled(cancel_event)
                filename = f"part_{index:02d}.glb"
                payload = _glb_bytes(mesh)
                _atomic_write_bytes(temporary_dir / filename, payload)
                part_records.append(
                    {
                        "index": index,
                        "name": f"part_{index:02d}",
                        "file": filename,
                        "vertices": int(len(mesh.vertices)),
                        "faces": int(len(mesh.faces)),
                    }
                )
                pct = 81 + int(10 * (index + 1) / num_parts)
                self._report(progress_cb, pct, f"Exporting part {index + 1}/{num_parts}")

            stage = "composite export"
            scene = _build_colored_scene(meshes)
            composite_payload = _glb_bytes(scene)
            _validate_composite_glb(composite_payload, num_parts)
            composite_name = f"{output_stem}.glb"
            _atomic_write_bytes(temporary_dir / composite_name, composite_payload)
            self._report(progress_cb, 94, "Composite GLB validated")

            render_outputs: dict[str, str] | None = None
            if settings["render"]:
                stage = "turntable rendering"
                self._check_cancelled(cancel_event)
                self._report(progress_cb, 95, "Rendering PartCrafter turntables")
                render_outputs = _render_sidecars(
                    scene,
                    inference_image,
                    temporary_dir,
                )
                self._check_cancelled(cancel_event)
                self._report(progress_cb, 98, "Turntable sidecars validated")

            stage = "metadata export"
            metadata: dict[str, Any] = {
                "adapter": {
                    "extension_id": EXTENSION_ID,
                    "node_id": node_id,
                    "author": "DrHepa",
                    "upstream_code_commit": UPSTREAM_CODE_COMMIT,
                    "weights_repository": NODE_REPOS[node_id],
                },
                "input": {"sha256": input_sha256},
                "generation": {
                    "num_parts": num_parts,
                    "part_count_mode": settings["part_count_mode"],
                    "part_model": (
                        settings["part_model"]
                        if settings["part_count_mode"] == "gemini"
                        else None
                    ),
                    "style_transfer_requested": settings["style_transfer"],
                    "style_transfer_applied": (
                        settings["style_transfer"] and style_error is None
                    ),
                    "style_model": (
                        settings["style_model"] if settings["style_transfer"] else None
                    ),
                    "style_error": style_error,
                    "remove_background_requested": settings["remove_background"],
                    "remove_background_applied": settings["remove_background"],
                    "num_tokens": settings["num_tokens"],
                    "num_inference_steps": settings["num_inference_steps"],
                    "guidance_scale": settings["guidance_scale"],
                    "max_num_expanded_coords": settings["max_num_expanded_coords"],
                    "use_flash_decoder": settings["use_flash_decoder"],
                    "render_requested": settings["render"],
                    "seed": settings["seed"],
                    "duration_seconds": round(time.time() - started, 3),
                },
                "outputs": {
                    "composite": composite_name,
                    "styled_input": (
                        "styled_input.png"
                        if settings["style_transfer"] and style_error is None
                        else None
                    ),
                    "renderings": render_outputs,
                    "parts": part_records,
                },
            }
            _atomic_write_json(temporary_dir / "generation.json", metadata)
            input_path.unlink(missing_ok=True)

            stage = "finalization"
            self._check_cancelled(cancel_event)
            os.replace(temporary_dir, final_dir)
            output_path = (final_dir / composite_name).resolve()
            if not output_path.is_absolute() or not output_path.is_file():
                raise RuntimeError("final GLB path was not committed")
            self._report(progress_cb, 100, "PartCrafter generation complete")
            _log(f"Generated {num_parts} named meshes: {output_path}")
            return output_path
        except GenerationCancelled:
            _log("Generation cancelled", level="WARNING")
            raise
        except Exception as exc:
            message = _safe_error(exc)
            _log(f"Generation failed during {stage}: {message}", level="ERROR")
            raise RuntimeError(
                f"PartCrafter {node_id} generation failed during {stage}: {message}"
            ) from None
        finally:
            if temporary_dir.exists():
                shutil.rmtree(temporary_dir, ignore_errors=True)

    def unload(self) -> None:
        with self._load_lock:
            pipeline = self._model
            rmbg_model = self._rmbg_model
            self._model = None
            self._rmbg_model = None
            self._torch = None
            if pipeline is not None:
                try:
                    if hasattr(pipeline, "maybe_free_model_hooks"):
                        pipeline.maybe_free_model_hooks()
                except Exception as exc:
                    _log(
                        f"Pipeline offload hook failed during unload: {_safe_error(exc)}",
                        level="WARNING",
                    )
                del pipeline
            if rmbg_model is not None:
                del rmbg_model
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    if hasattr(torch.cuda, "ipc_collect"):
                        torch.cuda.ipc_collect()
            except Exception as exc:
                _log(
                    f"CUDA cache cleanup failed during unload: {_safe_error(exc)}",
                    level="WARNING",
                )
            _log("PartCrafter models unloaded")
