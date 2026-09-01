"""Prepare PartCrafter's isolated Modly runtime without downloading weights.

Modly 0.4.2 invokes this file with one JSON argument.  The legacy positional
form remains accepted for Repair of older installations.  Setup owns only the
extension-local virtual environment and a pinned copy of upstream Python
source; model checkpoints remain exclusively managed by Modly's Models UI.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
import platform as platform_module
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback
import urllib.request
import uuid
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / "venv"
MODLY_DIR = ROOT / ".modly"
SETUP_DIR = MODLY_DIR / "setup"
LOG_PATH = SETUP_DIR / "logs" / "setup.log"
STATUS_PATH = SETUP_DIR / "setup-status.json"
DOWNLOADS_DIR = SETUP_DIR / "downloads"
UPSTREAM_PARENT = MODLY_DIR / "upstream"
UPSTREAM_ROOT = UPSTREAM_PARENT / "partcrafter"
SOURCE_STATE_NAME = ".modly-source.json"
REQUIREMENTS_PATH = ROOT / "requirements-runtime.txt"

UPSTREAM_COMMIT = "3d773bf02fad51c7ab31a5615573fec93b287b30"
UPSTREAM_ARCHIVE_URL = (
    "https://github.com/wgsxm/PartCrafter/archive/"
    f"{UPSTREAM_COMMIT}.tar.gz"
)
UPSTREAM_ARCHIVE_SHA256 = (
    "d7d1cf92c8d642af134f225ab447ff63b3b4784f1516d0c133c41e7cd0e2ccb6"
)
SOURCE_PATCH_VERSION = 1
ARCHIVE_MAX_BYTES = 128 * 1024 * 1024
EXTRACTED_FILE_MAX_BYTES = 8 * 1024 * 1024
EXTRACTED_TOTAL_MAX_BYTES = 16 * 1024 * 1024

BOOTSTRAP_REQUIREMENTS = (
    "pip==25.1.1",
    "setuptools==80.9.0",
    "wheel==0.45.1",
)
SUPPORTED_PYTHON_VERSIONS = ((3, 11), (3, 12))
SUPPORTED_PYTHON_LABEL = "3.11 or 3.12"


class SetupError(RuntimeError):
    """Actionable setup failure safe to surface in Modly."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.public_message = message
        super().__init__(f"{code}: {message}")


class SetupLogger:
    """Mirror setup evidence to stderr and a persistent UTF-8 log."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = path.open("w", encoding="utf-8", newline="\n")

    def log(self, message: str) -> None:
        line = " ".join(str(message).replace("\x00", "").splitlines()).strip()
        rendered = f"[PartCrafter setup] {line}"
        print(rendered, file=sys.stderr, flush=True)
        self._stream.write(rendered + "\n")
        self._stream.flush()

    def close(self) -> None:
        self._stream.close()


@dataclass(frozen=True)
class SetupContext:
    python_exe: Path
    ext_dir: Path
    gpu_sm: int
    cuda_version: int
    accelerator: str
    platform: str
    arch: str


@dataclass(frozen=True)
class TorchProfile:
    profile_id: str
    torch_version: str
    torchvision_version: str
    cuda_lane: str
    torch_index_url: str


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _normalize_platform(value: Any) -> str:
    text = str(value or "").strip().lower()
    aliases = {
        "windows": "win32",
        "win": "win32",
        "linux2": "linux",
        "macos": "darwin",
        "mac": "darwin",
    }
    return aliases.get(text, text)


def _normalize_arch(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    if text in {"x64", "x86_64", "amd64"}:
        return "x64"
    if text in {"arm64", "aarch64"}:
        return "arm64"
    return text


def _parse_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise SetupError("INVALID_ARGUMENT", f"{name} must be an integer")
    if isinstance(value, float) and not value.is_integer():
        raise SetupError("INVALID_ARGUMENT", f"{name} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SetupError("INVALID_ARGUMENT", f"{name} must be an integer") from exc
    if parsed < minimum:
        raise SetupError("INVALID_ARGUMENT", f"{name} must be at least {minimum}")
    return parsed


def _parse_cuda_version(value: Any) -> int:
    if value in (None, ""):
        return 0
    text = str(value).strip().lower().replace("cuda", "").replace("cu", "")
    if text == "0":
        return 0
    if "." in text:
        parts = text.split(".")
        if (
            len(parts) != 2
            or not all(part.isdigit() for part in parts)
            or len(parts[1]) != 1
            or not 10 <= int(parts[0]) < 20
        ):
            raise SetupError(
                "INVALID_ARGUMENT", "cuda_version must look like 128 or 12.8"
            )
        return int(parts[0]) * 10 + int(parts[1])
    if not text.isdigit():
        raise SetupError(
            "INVALID_ARGUMENT", "cuda_version must look like 128 or 12.8"
        )
    parsed = int(text)
    if 10 <= parsed < 20:
        return parsed * 10
    if 100 <= parsed <= 999:
        return parsed
    raise SetupError(
        "INVALID_ARGUMENT", "cuda_version must look like 128 or 12.8"
    )


def parse_setup_args(argv: Sequence[str]) -> SetupContext:
    """Parse Modly's current JSON payload and the historical positional form."""

    if len(argv) == 1:
        try:
            payload = json.loads(argv[0])
        except json.JSONDecodeError as exc:
            raise SetupError(
                "INVALID_ARGUMENT", "the setup payload is not valid JSON"
            ) from exc
        if not isinstance(payload, dict):
            raise SetupError("INVALID_ARGUMENT", "the setup payload must be an object")
        missing = [key for key in ("python_exe", "ext_dir") if not payload.get(key)]
        if missing:
            raise SetupError(
                "INVALID_ARGUMENT",
                "the setup payload is missing: " + ", ".join(missing),
            )
        gpu_sm = _parse_int(payload.get("gpu_sm", 0), "gpu_sm")
        accelerator = str(
            payload.get("accelerator") or ("cuda" if gpu_sm > 0 else "cpu")
        ).strip().lower()
        return SetupContext(
            python_exe=Path(str(payload["python_exe"])).expanduser(),
            ext_dir=Path(str(payload["ext_dir"])).expanduser(),
            gpu_sm=gpu_sm,
            cuda_version=_parse_cuda_version(payload.get("cuda_version", 0)),
            accelerator=accelerator,
            platform=_normalize_platform(payload.get("platform") or sys.platform),
            arch=_normalize_arch(
                payload.get("arch") or platform_module.machine()
            ),
        )

    if len(argv) >= 3 and len(argv) <= 4:
        gpu_sm = _parse_int(argv[2], "gpu_sm")
        platform_name = _normalize_platform(sys.platform)
        arch_name = _normalize_arch(platform_module.machine())
        if len(argv) == 4:
            cuda_version = _parse_cuda_version(argv[3])
        elif gpu_sm <= 0:
            cuda_version = 0
        elif arch_name == "arm64" or gpu_sm >= 100:
            raise SetupError(
                "CUDA_VERSION_REQUIRED",
                "legacy setup arguments cannot safely select the Linux ARM64 "
                "or NVIDIA Blackwell CUDA lane. Run setup from the current "
                "Modly Extensions UI so its JSON payload includes cuda_version.",
            )
        else:
            # CUDA 11.8 is the most conservative x64 wheel lane supported by
            # the upstream runtime and works with the broadest driver range.
            cuda_version = 118
        return SetupContext(
            python_exe=Path(argv[0]).expanduser(),
            ext_dir=Path(argv[1]).expanduser(),
            gpu_sm=gpu_sm,
            cuda_version=cuda_version,
            accelerator="cuda" if gpu_sm > 0 else "cpu",
            platform=platform_name,
            arch=arch_name,
        )

    raise SetupError(
        "INVALID_ARGUMENT",
        "expected one Modly JSON payload or legacy: python_exe ext_dir gpu_sm "
        "[cuda_version]",
    )


def select_torch_profile(context: SetupContext) -> TorchProfile:
    """Select only audited CUDA/Python wheel lanes for the target host."""

    if context.accelerator != "cuda" or context.gpu_sm <= 0:
        raise SetupError(
            "UNSUPPORTED_ACCELERATOR",
            "PartCrafter requires an NVIDIA CUDA GPU; CPU and Apple MPS are "
            "not supported upstream.",
        )
    if context.platform not in {"linux", "win32"}:
        raise SetupError(
            "UNSUPPORTED_PLATFORM",
            "PartCrafter setup supports Linux or Windows with NVIDIA CUDA only.",
        )
    if context.platform == "win32" and context.arch != "x64":
        raise SetupError(
            "UNSUPPORTED_PLATFORM",
            "Windows ARM64 is not supported; use Windows x64 with NVIDIA CUDA.",
        )
    if context.platform == "linux" and context.arch not in {"x64", "arm64"}:
        raise SetupError(
            "UNSUPPORTED_PLATFORM",
            f"Linux architecture {context.arch!r} is not supported.",
        )

    if context.arch == "arm64":
        if context.gpu_sm < 90:
            raise SetupError(
                "UNSUPPORTED_GPU",
                "Linux ARM64 requires a server-class NVIDIA SBSA GPU with "
                "compute capability SM 9.0 or newer.",
            )
        if context.cuda_version < 128:
            raise SetupError(
                "UNSUPPORTED_CUDA",
                "Linux ARM64 is provided through the CUDA 12.8 PyTorch lane. "
                "Update the NVIDIA driver/runtime so Modly reports CUDA 12.8.",
            )
        return TorchProfile(
            profile_id="linux-arm64-cu128-torch271",
            torch_version="2.7.1",
            torchvision_version="0.22.1",
            cuda_lane="cu128",
            torch_index_url="https://download.pytorch.org/whl/cu128",
        )

    if context.gpu_sm >= 100 and context.cuda_version < 128:
        raise SetupError(
            "UNSUPPORTED_CUDA",
            "NVIDIA SM 10.0+ requires the CUDA 12.8 PyTorch lane. Update "
            "the driver so Modly reports CUDA 12.8.",
        )

    # CUDA_VISIBLE_DEVICES can make torch device 0 differ from the first
    # physical GPU reported by Modly's nvidia-smi probe. CUDA 12.8 therefore
    # selects the Blackwell-capable lane even when the advisory SM is older;
    # the smoke test validates the actual visible device after installation.
    if context.cuda_version >= 128:
        return TorchProfile(
            profile_id=f"{context.platform}-x64-cu128-torch271",
            torch_version="2.7.1",
            torchvision_version="0.22.1",
            cuda_lane="cu128",
            torch_index_url="https://download.pytorch.org/whl/cu128",
        )

    if context.cuda_version >= 124:
        lane = "cu124"
    elif context.cuda_version >= 121:
        lane = "cu121"
    elif context.cuda_version >= 118:
        lane = "cu118"
    else:
        raise SetupError(
            "UNSUPPORTED_CUDA",
            "PartCrafter requires an NVIDIA driver compatible with CUDA 11.8 "
            "or newer.",
        )
    return TorchProfile(
        profile_id=f"{context.platform}-x64-{lane}-torch251",
        torch_version="2.5.1",
        torchvision_version="0.20.1",
        cuda_lane=lane,
        torch_index_url=f"https://download.pytorch.org/whl/{lane}",
    )


def _venv_python(venv_dir: Path, platform_name: str) -> Path:
    if platform_name == "win32":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    paths = [root / "LICENSE", *sorted((root / "src").rglob("*.py"))]
    for path in paths:
        if not path.is_file():
            raise SetupError("SOURCE_INVALID", f"source file is missing: {path}")
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(relative + b"\0")
        digest.update(bytes.fromhex(_sha256_file(path)))
    return digest.hexdigest()


def _replace_exact(
    text: str,
    old: str,
    new: str,
    label: str,
    *,
    expected: int = 1,
) -> str:
    count = text.count(old)
    if count != expected:
        raise SetupError(
            "SOURCE_DRIFT",
            f"upstream patch {label!r} expected {expected} match, found {count}; "
            "the pinned source no longer matches the audited revision.",
        )
    return text.replace(old, new)


def _apply_source_patches(source_root: Path) -> list[str]:
    """Apply audited inference fixes, failing closed on any source drift."""

    pipeline_path = source_root / "src" / "pipelines" / "pipeline_partcrafter.py"
    vae_path = (
        source_root
        / "src"
        / "models"
        / "autoencoders"
        / "autoencoder_kl_triposg.py"
    )
    inference_path = source_root / "src" / "utils" / "inference_utils.py"

    pipeline = pipeline_path.read_text(encoding="utf-8")
    pipeline = _replace_exact(
        pipeline,
        """        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)\n        return noise\n""",
        """        if latents is None:\n            latents = randn_tensor(\n                shape, generator=generator, device=device, dtype=dtype\n            )\n        else:\n            if tuple(latents.shape) != shape:\n                raise ValueError(\n                    f\"Unexpected latents shape {tuple(latents.shape)}; expected {shape}\"\n                )\n            latents = latents.to(device=device, dtype=dtype)\n        return latents\n""",
        "honor supplied latents",
    )
    pipeline = _replace_exact(
        pipeline,
        """                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)\n\n                    latents = callback_outputs.pop(\"latents\", latents)\n                    image_embeds_1 = callback_outputs.pop(\n                        \"image_embeds_1\", image_embeds_1\n                    )\n                    negative_image_embeds_1 = callback_outputs.pop(\n                        \"negative_image_embeds_1\", negative_image_embeds_1\n                    )\n                    image_embeds_2 = callback_outputs.pop(\n                        \"image_embeds_2\", image_embeds_2\n                    )\n                    negative_image_embeds_2 = callback_outputs.pop(\n                        \"negative_image_embeds_2\", negative_image_embeds_2\n                    )\n""",
        """                    callback_outputs = (\n                        callback_on_step_end(self, i, t, callback_kwargs) or {}\n                    )\n                    latents = callback_outputs.pop(\"latents\", latents)\n""",
        "repair denoising callback",
    )
    pipeline = _replace_exact(
        pipeline,
        """        self.vae.set_flash_decoder()\n        output, meshes = [], []\n""",
        """        if use_flash_decoder:\n            self.vae.set_flash_decoder()\n        else:\n            self.vae.set_default_attn_processor()\n        output, meshes = [], []\n""",
        "honor and reset flash decoder",
    )
    pipeline = _replace_exact(
        pipeline,
        """                try:\n                    mesh_v_f = hierarchical_extract_geometry(\n                        geometric_func,\n                        device,\n                        dtype=latents.dtype,\n                        bounds=bounds,\n                        dense_octree_depth=dense_octree_depth,\n                        hierarchical_octree_depth=hierarchical_octree_depth,\n                        max_num_expanded_coords=max_num_expanded_coords,\n                        # verbose=True\n                    )\n                    mesh = trimesh.Trimesh(mesh_v_f[0].astype(np.float32), mesh_v_f[1])\n                except:\n                    mesh_v_f = None\n                    mesh = None\n""",
        """                mesh_v_f = hierarchical_extract_geometry(\n                    geometric_func,\n                    device,\n                    dtype=latents.dtype,\n                    bounds=bounds,\n                    dense_octree_depth=dense_octree_depth,\n                    hierarchical_octree_depth=hierarchical_octree_depth,\n                    max_num_expanded_coords=max_num_expanded_coords,\n                    # verbose=True\n                )\n                mesh = trimesh.Trimesh(\n                    mesh_v_f[0].astype(np.float32), mesh_v_f[1]\n                )\n""",
        "propagate mesh decoding errors",
    )
    pipeline_path.write_text(pipeline, encoding="utf-8", newline="\n")

    inference = inference_path.read_text(encoding="utf-8")
    inference = _replace_exact(
        inference,
        "device='cuda'",
        "device=edge_coords.device",
        "remove hard-coded CUDA device",
    )
    inference_path.write_text(inference, encoding="utf-8", newline="\n")

    vae = vae_path.read_text(encoding="utf-8")
    vae = _replace_exact(
        vae,
        """from torch_cluster import fps\nfrom tqdm import tqdm\n""",
        """try:\n    from torch_cluster import fps as _torch_cluster_fps\nexcept (ImportError, OSError):\n    _torch_cluster_fps = None\nfrom tqdm import tqdm\n\nTORCH_CLUSTER_AVAILABLE = _torch_cluster_fps is not None\n\ndef fps(points, batch, ratio, random_start=True):\n    if _torch_cluster_fps is not None:\n        return _torch_cluster_fps(\n            points, batch, ratio=ratio, random_start=random_start\n        )\n    selected = []\n    for batch_id in torch.unique(batch, sorted=True):\n        indices = torch.nonzero(batch == batch_id, as_tuple=False).flatten()\n        if indices.numel() == 0:\n            continue\n        count = max(1, min(indices.numel(), int(np.ceil(indices.numel() * ratio))))\n        local = points.index_select(0, indices)\n        first = (\n            int(torch.randint(indices.numel(), (1,), device=points.device).item())\n            if random_start\n            else 0\n        )\n        distances = torch.full(\n            (indices.numel(),), float(\"inf\"), device=points.device\n        )\n        chosen = first\n        local_selected = []\n        for _ in range(count):\n            local_selected.append(chosen)\n            delta = local - local[chosen]\n            distances = torch.minimum(distances, (delta * delta).sum(dim=-1))\n            chosen = int(torch.argmax(distances).item())\n        local_tensor = torch.tensor(\n            local_selected, dtype=torch.long, device=points.device\n        )\n        selected.append(indices.index_select(0, local_tensor))\n    if not selected:\n        return torch.empty(0, dtype=torch.long, device=points.device)\n    return torch.cat(selected, dim=0)\n""",
        "provide portable FPS fallback",
    )
    vae_path.write_text(vae, encoding="utf-8", newline="\n")

    for path in sorted((source_root / "src").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        try:
            compile(text, str(path), "exec")
        except SyntaxError as exc:
            raise SetupError(
                "SOURCE_PATCH_INVALID", f"patched source does not compile: {path}: {exc}"
            ) from exc

    if any(token in pipeline for token in ("image_embeds_1", "image_embeds_2")):
        raise SetupError(
            "SOURCE_PATCH_INVALID", "undefined callback tensors remain after patching"
        )
    return [
        "honor supplied latents",
        "repair denoising callback",
        "honor and reset flash decoder",
        "propagate mesh decoding errors",
        "remove hard-coded CUDA device",
        "provide portable FPS fallback",
    ]


def _download_source_archive(destination: Path, logger: SetupLogger) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and _sha256_file(destination) == UPSTREAM_ARCHIVE_SHA256:
        logger.log("Reusing verified immutable PartCrafter source archive")
        return
    destination.unlink(missing_ok=True)

    last_error: BaseException | None = None
    for attempt in range(1, 4):
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.part")
        try:
            logger.log(f"Downloading pinned PartCrafter source (attempt {attempt}/3)")
            request = urllib.request.Request(
                UPSTREAM_ARCHIVE_URL,
                headers={"User-Agent": "modly-partcrafter-extension/1.0.0"},
            )
            total = 0
            with urllib.request.urlopen(request, timeout=60) as response, temporary.open(
                "wb"
            ) as stream:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > ARCHIVE_MAX_BYTES:
                        raise SetupError(
                            "SOURCE_DOWNLOAD_INVALID",
                            "the upstream source archive exceeded the safety limit",
                        )
                    stream.write(chunk)
                stream.flush()
                os.fsync(stream.fileno())
            digest = _sha256_file(temporary)
            if digest != UPSTREAM_ARCHIVE_SHA256:
                raise SetupError(
                    "SOURCE_CHECKSUM_MISMATCH",
                    "the PartCrafter source archive checksum did not match the "
                    "audited revision; retry Repair after checking the network path.",
                )
            os.replace(temporary, destination)
            logger.log(f"Verified source archive SHA-256 ({total:,} bytes)")
            return
        except BaseException as exc:
            last_error = exc
            temporary.unlink(missing_ok=True)
            if isinstance(exc, SetupError) and exc.code in {
                "SOURCE_DOWNLOAD_INVALID",
                "SOURCE_CHECKSUM_MISMATCH",
            }:
                break
            if attempt < 3:
                time.sleep(attempt)
    if isinstance(last_error, SetupError):
        raise last_error
    raise SetupError(
        "SOURCE_DOWNLOAD_FAILED",
        f"could not download pinned PartCrafter source: {last_error}. Check "
        "internet access and run Repair.",
    ) from last_error


def _safe_extract_runtime_source(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    extracted = 0
    root_component: str | None = None
    with tarfile.open(archive, mode="r:gz") as bundle:
        for member in bundle.getmembers():
            pure = PurePosixPath(member.name)
            if pure.is_absolute() or ".." in pure.parts or len(pure.parts) < 2:
                continue
            if root_component is None:
                root_component = pure.parts[0]
            if pure.parts[0] != root_component:
                raise SetupError(
                    "SOURCE_ARCHIVE_UNSAFE", "source archive has multiple roots"
                )
            relative = PurePosixPath(*pure.parts[1:])
            wanted = relative == PurePosixPath("LICENSE") or (
                len(relative.parts) >= 2
                and relative.parts[0] == "src"
                and relative.suffix == ".py"
            )
            if not wanted:
                continue
            if not member.isfile() or member.issym() or member.islnk():
                raise SetupError(
                    "SOURCE_ARCHIVE_UNSAFE",
                    f"source entry is not a regular file: {relative.as_posix()}",
                )
            if member.size > EXTRACTED_FILE_MAX_BYTES:
                raise SetupError(
                    "SOURCE_ARCHIVE_UNSAFE",
                    f"source entry is unexpectedly large: {relative.as_posix()}",
                )
            extracted += member.size
            if extracted > EXTRACTED_TOTAL_MAX_BYTES:
                raise SetupError(
                    "SOURCE_ARCHIVE_UNSAFE", "extracted source exceeded safety limit"
                )
            source = bundle.extractfile(member)
            if source is None:
                raise SetupError(
                    "SOURCE_ARCHIVE_INVALID",
                    f"cannot read source entry: {relative.as_posix()}",
                )
            target = destination.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with source, target.open("wb") as stream:
                shutil.copyfileobj(source, stream)

    required = (
        destination / "LICENSE",
        destination / "src" / "pipelines" / "pipeline_partcrafter.py",
        destination
        / "src"
        / "models"
        / "autoencoders"
        / "autoencoder_kl_triposg.py",
        destination / "src" / "utils" / "inference_utils.py",
    )
    missing = [path.relative_to(destination).as_posix() for path in required if not path.is_file()]
    if missing:
        raise SetupError(
            "SOURCE_ARCHIVE_INVALID",
            "source archive is missing: " + ", ".join(missing),
        )


def _source_ready(source_root: Path = UPSTREAM_ROOT) -> tuple[bool, str | None]:
    marker_path = source_root / SOURCE_STATE_NAME
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if not isinstance(marker, dict):
            return False, None
        if marker.get("upstream_commit") != UPSTREAM_COMMIT:
            return False, None
        if marker.get("archive_sha256") != UPSTREAM_ARCHIVE_SHA256:
            return False, None
        if marker.get("patch_version") != SOURCE_PATCH_VERSION:
            return False, None
        tree_hash = _source_tree_hash(source_root)
        return tree_hash == marker.get("tree_sha256"), tree_hash
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError, SetupError):
        return False, None


def prepare_upstream_source(logger: SetupLogger) -> str:
    ready, tree_hash = _source_ready()
    if ready and tree_hash:
        logger.log("Pinned and patched PartCrafter source is already valid")
        return tree_hash

    archive = DOWNLOADS_DIR / f"partcrafter-{UPSTREAM_COMMIT}.tar.gz"
    _download_source_archive(archive, logger)
    UPSTREAM_PARENT.mkdir(parents=True, exist_ok=True)
    staging = UPSTREAM_PARENT / f".partcrafter-{uuid.uuid4().hex}.staging"
    backup = UPSTREAM_PARENT / f".partcrafter-{uuid.uuid4().hex}.backup"
    try:
        _safe_extract_runtime_source(archive, staging)
        patches = _apply_source_patches(staging)
        tree_hash = _source_tree_hash(staging)
        _atomic_write_json(
            staging / SOURCE_STATE_NAME,
            {
                "schema_version": 1,
                "upstream_commit": UPSTREAM_COMMIT,
                "archive_sha256": UPSTREAM_ARCHIVE_SHA256,
                "patch_version": SOURCE_PATCH_VERSION,
                "patches": patches,
                "tree_sha256": tree_hash,
            },
        )
        if UPSTREAM_ROOT.exists():
            os.replace(UPSTREAM_ROOT, backup)
        os.replace(staging, UPSTREAM_ROOT)
        if backup.exists():
            shutil.rmtree(backup)
        logger.log("Installed pinned source and six audited inference fixes")
        return tree_hash
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        if backup.exists() and not UPSTREAM_ROOT.exists():
            os.replace(backup, UPSTREAM_ROOT)
        raise
    finally:
        if backup.exists() and UPSTREAM_ROOT.exists():
            shutil.rmtree(backup, ignore_errors=True)


def _format_command(command: Sequence[str]) -> str:
    return " ".join(str(part) for part in command)


def _run_streamed(
    command: Sequence[str],
    logger: SetupLogger,
    *,
    env: Mapping[str, str] | None = None,
    allow_failure: bool = False,
) -> bool:
    logger.log("Running: " + _format_command(command))
    process = subprocess.Popen(
        [str(part) for part in command],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=dict(env) if env is not None else None,
    )
    assert process.stdout is not None
    for line in process.stdout:
        logger.log(line.rstrip("\r\n"))
    code = process.wait()
    if code != 0 and not allow_failure:
        raise SetupError(
            "COMMAND_FAILED",
            f"command exited with code {code}: {_format_command(command)}",
        )
    return code == 0


def _probe_python(python_exe: Path) -> dict[str, Any]:
    script = (
        "import json,platform,struct,sys;"
        "print(json.dumps({'version':list(sys.version_info[:2]),"
        "'implementation':platform.python_implementation(),"
        "'platform':sys.platform,'machine':platform.machine(),"
        "'bits':struct.calcsize('P')*8}))"
    )
    try:
        result = subprocess.run(
            [str(python_exe), "-c", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise SetupError(
            "PYTHON_UNUSABLE",
            f"cannot execute Python at {python_exe}: {(exc.stderr or '').strip()}",
        ) from exc
    except OSError as exc:
        raise SetupError(
            "PYTHON_UNUSABLE",
            f"cannot execute Python at {python_exe}: {exc}",
        ) from exc
    try:
        value = json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise SetupError("PYTHON_UNUSABLE", "Python probe returned invalid output") from exc
    return value


def _python_version(probe: Mapping[str, Any]) -> tuple[int, int] | None:
    value = probe.get("version")
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        return int(value[0]), int(value[1])
    except (TypeError, ValueError, OverflowError):
        return None


def _python_probe_matches(
    probe: Mapping[str, Any],
    expected_version: tuple[int, int],
    expected_platform: str,
    expected_arch: str,
) -> bool:
    # _probe_python always supplies platform; the fallback keeps synthetic
    # probes used by contract tests concise without weakening real validation.
    probe_platform = _normalize_platform(
        probe.get("platform") or expected_platform
    )
    return (
        _python_version(probe) == expected_version
        and probe.get("implementation") == "CPython"
        and probe.get("bits") == 64
        and probe_platform == expected_platform
        and _normalize_arch(probe.get("machine")) == expected_arch
    )


def _validate_context(context: SetupContext) -> dict[str, Any]:
    if os.path.normcase(str(context.ext_dir.resolve())) != os.path.normcase(
        str(ROOT.resolve())
    ):
        raise SetupError(
            "EXTENSION_PATH_MISMATCH",
            f"Modly passed ext_dir={context.ext_dir}, but setup.py lives in {ROOT}.",
        )
    if not context.python_exe.is_file():
        raise SetupError(
            "PYTHON_NOT_FOUND", f"Modly Python was not found at {context.python_exe}"
        )
    probe = _probe_python(context.python_exe)
    version = _python_version(probe)
    if (
        version not in SUPPORTED_PYTHON_VERSIONS
        or probe.get("implementation") != "CPython"
        or probe.get("bits") != 64
    ):
        raise SetupError(
            "UNSUPPORTED_PYTHON",
            "PartCrafter requires a 64-bit CPython "
            f"{SUPPORTED_PYTHON_LABEL} Modly runtime.",
        )
    actual_arch = _normalize_arch(probe.get("machine"))
    if actual_arch and actual_arch != context.arch:
        raise SetupError(
            "ARCHITECTURE_MISMATCH",
            f"Modly reported {context.arch}, but Python reports {actual_arch}.",
        )
    actual_platform = _normalize_platform(probe.get("platform"))
    if actual_platform != context.platform:
        raise SetupError(
            "PLATFORM_MISMATCH",
            f"Modly reported {context.platform}, but Python reports "
            f"{actual_platform or 'an unknown platform'}.",
        )
    return probe


def _requirements_sha256() -> str:
    if not REQUIREMENTS_PATH.is_file():
        raise SetupError(
            "REQUIREMENTS_MISSING", "requirements-runtime.txt is missing from the extension"
        )
    return _sha256_file(REQUIREMENTS_PATH)


def _read_status() -> dict[str, Any] | None:
    try:
        value = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


SMOKE_SCRIPT = r"""
import json
import os
import platform
import sys

source_root, expected_torch, expected_vision, expected_cuda_lane, expected_sm = sys.argv[1:6]
sys.path.insert(0, source_root)
is_linux = sys.platform.startswith("linux")
if is_linux:
    os.environ["PYOPENGL_PLATFORM"] = "egl"
else:
    # Windows uses pyrender's normal Pyglet/WGL backend. Never inherit a
    # Linux-oriented EGL selection into the worker.
    os.environ.pop("PYOPENGL_PLATFORM", None)

import accelerate
import cv2
import diffusers
import einops
from google import genai
import huggingface_hub
import numpy
import omegaconf
import OpenGL
import peft
from PIL import Image
import pyglet
import pyrender
import safetensors
import scipy
import skimage
import torch
import torchvision
import transformers
import trimesh
from src.utils import render_utils
from src.models.autoencoders.autoencoder_kl_triposg import (
    TORCH_CLUSTER_AVAILABLE,
    TripoSGVAEModel,
)
from src.models.transformers.partcrafter_transformer import PartCrafterDiTModel
from src.pipelines.pipeline_partcrafter import PartCrafterPipeline
from src.schedulers.scheduling_rectified_flow import RectifiedFlowScheduler

# Validate the actual platform context required by the optional upstream
# turntable path. Import-only checks cannot detect a missing EGL/OpenGL runtime.
if not is_linux:
    # Upstream assigns EGL after importing pyrender. Remove that late value
    # before OffscreenRenderer chooses/creates the Windows context.
    os.environ.pop("PYOPENGL_PLATFORM", None)
renderer = pyrender.OffscreenRenderer(viewport_width=1, viewport_height=1)
renderer.delete()

if torch.__version__.split('+', 1)[0] != expected_torch:
    raise RuntimeError(f"torch version drift: {torch.__version__}")
if torchvision.__version__.split('+', 1)[0] != expected_vision:
    raise RuntimeError(f"torchvision version drift: {torchvision.__version__}")
runtime_cuda_lane = str(torch.version.cuda or "").replace(".", "")
if runtime_cuda_lane != expected_cuda_lane.removeprefix("cu"):
    raise RuntimeError(
        f"PyTorch CUDA lane drift: torch reports {torch.version.cuda}, "
        f"expected {expected_cuda_lane}"
    )
if not torch.cuda.is_available():
    raise RuntimeError("torch.cuda.is_available() is false")
if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
    raise RuntimeError("scaled_dot_product_attention is unavailable")
capability = torch.cuda.get_device_capability(0)
detected_sm = capability[0] * 10 + capability[1]
reported_sm = int(expected_sm)
machine = platform.machine().strip().lower()
if machine in {"aarch64", "arm64"} and detected_sm < 90:
    raise RuntimeError(
        "Linux ARM64 requires a server-class NVIDIA SBSA device with SM 9.0 "
        f"or newer; the visible CUDA device is SM {detected_sm}"
    )
if detected_sm >= 100 and expected_cuda_lane != "cu128":
    raise RuntimeError(
        f"visible SM {detected_sm} requires the cu128 PyTorch lane, got "
        f"{expected_cuda_lane}"
    )
matrix = torch.randn((16, 16), device="cuda", dtype=torch.float16)
matmul = matrix @ matrix
query = torch.randn((1, 1, 4, 8), device="cuda", dtype=torch.float16)
attention = torch.nn.functional.scaled_dot_product_attention(query, query, query)
if not torch.isfinite(matmul).all() or not torch.isfinite(attention).all():
    raise RuntimeError("CUDA smoke kernels produced non-finite values")
torch.cuda.synchronize()
print(json.dumps({
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "torch_cuda": torch.version.cuda,
    "device": torch.cuda.get_device_name(0),
    "capability": list(capability),
    "reported_sm": reported_sm,
    "detected_sm": detected_sm,
    "arch_list": torch.cuda.get_arch_list(),
    "vram_bytes": torch.cuda.get_device_properties(0).total_memory,
    "torch_cluster_native": bool(TORCH_CLUSTER_AVAILABLE),
    "pipeline_symbol": PartCrafterPipeline.__name__,
    "vae_symbol": TripoSGVAEModel.__name__,
    "transformer_symbol": PartCrafterDiTModel.__name__,
    "scheduler_symbol": RectifiedFlowScheduler.__name__,
        "render_symbol": render_utils.render_views_around_mesh.__name__,
        "render_context": "ok",
}, sort_keys=True))
"""


def _run_smoke(
    venv_python: Path,
    profile: TorchProfile,
    gpu_sm: int,
) -> dict[str, Any]:
    result = subprocess.run(
        [
            str(venv_python),
            "-c",
            SMOKE_SCRIPT,
            str(UPSTREAM_ROOT),
            profile.torch_version,
            profile.torchvision_version,
            profile.cuda_lane,
            str(gpu_sm),
        ],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        env={
            **os.environ,
            "PYTHONUTF8": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "DIFFUSERS_OFFLINE": "1",
        },
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()[-1800:]
        if sys.platform.startswith("linux") and any(
            token in detail.lower() for token in ("egl", "opengl", "libgl")
        ):
            detail += (
                ". Linux turntable rendering also requires the system packages "
                "libegl1 and libgl1 (package names shown for Debian/Ubuntu)"
            )
        raise SetupError(
            "RUNTIME_SMOKE_FAILED",
            f"the isolated PartCrafter runtime failed validation: {detail}. "
            "Run Repair after checking the selected CUDA profile.",
        )
    for line in reversed(result.stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise SetupError("RUNTIME_SMOKE_FAILED", "runtime smoke test returned no JSON evidence")


def _ensure_venv(
    context: SetupContext,
    profile: TorchProfile,
    source_hash: str,
    host_probe: Mapping[str, Any],
    logger: SetupLogger,
) -> tuple[Path, dict[str, Any], bool]:
    host_version = _python_version(host_probe)
    if host_version not in SUPPORTED_PYTHON_VERSIONS:
        raise SetupError(
            "UNSUPPORTED_PYTHON",
            "PartCrafter requires a 64-bit CPython "
            f"{SUPPORTED_PYTHON_LABEL} Modly runtime.",
        )
    requirements_hash = _requirements_sha256()
    venv_python = _venv_python(VENV_DIR, context.platform)
    status = _read_status()
    expected_state = {
        "python_version": list(host_version),
        "profile_id": profile.profile_id,
        "requirements_sha256": requirements_hash,
        "source_tree_sha256": source_hash,
    }

    if venv_python.is_file() and status and status.get("result") == "ready":
        if all(status.get(key) == value for key, value in expected_state.items()):
            try:
                venv_probe = _probe_python(venv_python)
                if not _python_probe_matches(
                    venv_probe, host_version, context.platform, context.arch
                ):
                    raise SetupError(
                        "VENV_INVALID",
                        "the existing extension venv does not match Modly's "
                        f"64-bit CPython {host_version[0]}.{host_version[1]} ABI",
                    )
                smoke = _run_smoke(venv_python, profile, context.gpu_sm)
                logger.log("Existing extension venv passed full runtime validation")
                return venv_python, smoke, True
            except SetupError as exc:
                logger.log(f"Existing venv needs Repair: {exc.public_message}")

    if VENV_DIR.exists():
        if venv_python.is_file():
            logger.log(
                "Rebuilding extension venv because its pinned state changed or "
                "its runtime smoke failed"
            )
        else:
            logger.log("Removing incomplete extension venv")
        shutil.rmtree(VENV_DIR)
    logger.log(
        "Creating extension-local 64-bit CPython "
        f"{host_version[0]}.{host_version[1]} venv"
    )
    _run_streamed(
        [str(context.python_exe), "-m", "venv", str(VENV_DIR)], logger
    )
    venv_probe = _probe_python(venv_python)
    if not _python_probe_matches(
        venv_probe, host_version, context.platform, context.arch
    ):
        raise SetupError(
            "VENV_INVALID",
            "the extension venv does not match Modly's 64-bit CPython "
            f"{host_version[0]}.{host_version[1]} ABI",
        )

    pip = [str(venv_python), "-m", "pip"]
    pip_env = {**os.environ, "PYTHONUTF8": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1"}
    _run_streamed([*pip, "install", "--upgrade", *BOOTSTRAP_REQUIREMENTS], logger, env=pip_env)

    logger.log(f"Installing PyTorch profile {profile.profile_id}")
    _run_streamed(
        [
            *pip,
            "install",
            "--upgrade",
            f"torch=={profile.torch_version}",
            f"torchvision=={profile.torchvision_version}",
            "--index-url",
            profile.torch_index_url,
        ],
        logger,
        env=pip_env,
    )
    _run_streamed(
        [*pip, "install", "--upgrade", "-r", str(REQUIREMENTS_PATH)],
        logger,
        env=pip_env,
    )

    logger.log(
        "Using the audited pure-PyTorch FPS fallback; PartCrafter inference does "
        "not call its training/encoding FPS helper"
    )

    _run_streamed([*pip, "check"], logger, env=pip_env)
    smoke = _run_smoke(venv_python, profile, context.gpu_sm)
    logger.log(
        "Runtime smoke passed: "
        f"torch={smoke.get('torch')} CUDA={smoke.get('torch_cuda')} "
        f"device={smoke.get('device')}"
    )
    if int(smoke.get("vram_bytes", 0)) < 8 * 1024**3:
        logger.log(
            "WARNING: the CUDA device reports less than 8 GiB VRAM; upstream "
            "recommends at least 8 GB and generation may run out of memory"
        )
    return venv_python, smoke, False


def run_setup(context: SetupContext, logger: SetupLogger) -> dict[str, Any]:
    """Execute one idempotent Install/Repair operation."""

    host_probe = _validate_context(context)
    profile = select_torch_profile(context)
    logger.log(
        f"Target: platform={context.platform} arch={context.arch} "
        f"SM={context.gpu_sm} CUDA={context.cuda_version} profile={profile.profile_id}"
    )
    source_hash = prepare_upstream_source(logger)
    venv_python, smoke, reused = _ensure_venv(
        context, profile, source_hash, host_probe, logger
    )
    return {
        "schema_version": 1,
        "result": "ready",
        "extension_id": "partcrafter",
        "extension_version": "1.0.0",
        "weights": "required-from-modly-models-ui",
        "python_version": list(_python_version(host_probe) or ()),
        "profile_id": profile.profile_id,
        "torch_profile": asdict(profile),
        "requirements_sha256": _requirements_sha256(),
        "source_tree_sha256": source_hash,
        "upstream_commit": UPSTREAM_COMMIT,
        "venv_python": str(venv_python),
        "venv_reused": reused,
        "host_python": host_probe,
        "runtime_smoke": smoke,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    logger = SetupLogger(LOG_PATH)
    started = time.time()
    try:
        logger.log("Starting Install/Repair; model weights will not be downloaded")
        logger.log(
            "LICENSE NOTICE: pinned upstream source contains Tencent Hunyuan-"
            "derived code with separate territory/AUP terms. Review README, "
            "NOTICE, and THIRD_PARTY_NOTICES.md before use."
        )
        context = parse_setup_args(arguments)
        status = run_setup(context, logger)
        status["duration_seconds"] = round(time.time() - started, 3)
        _atomic_write_json(STATUS_PATH, status)
        logger.log(
            "READY: environment and pinned source validated. Download Object, "
            "Scene, or RMBG weights from Modly's Models UI."
        )
        return 0
    except SetupError as exc:
        failure = {
            "schema_version": 1,
            "result": "error",
            "code": exc.code,
            "message": exc.public_message,
            "duration_seconds": round(time.time() - started, 3),
            "failed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        try:
            _atomic_write_json(STATUS_PATH, failure)
        except OSError:
            pass
        logger.log(
            f"FAILED [{exc.code}]: {exc.public_message} "
            "Correct the issue and choose Repair in Modly."
        )
        return 1
    except BaseException as exc:
        message = " ".join(str(exc).split()) or type(exc).__name__
        try:
            _atomic_write_json(
                STATUS_PATH,
                {
                    "schema_version": 1,
                    "result": "error",
                    "code": "UNEXPECTED_SETUP_ERROR",
                    "message": message[:1800],
                    "duration_seconds": round(time.time() - started, 3),
                },
            )
        except OSError:
            pass
        logger.log(
            "Unexpected traceback: "
            + "".join(traceback.format_exception(exc)).strip()[-4_000:]
        )
        logger.log(
            "FAILED [UNEXPECTED_SETUP_ERROR]: "
            f"{message[:1500]}. Inspect {LOG_PATH} and choose Repair in Modly."
        )
        return 1
    finally:
        logger.close()


if __name__ == "__main__":
    raise SystemExit(main())
