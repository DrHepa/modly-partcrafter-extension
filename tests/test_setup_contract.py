from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests._support import load_module


PIPELINE_FIXTURE = '''class Pipeline:
    def prepare_latents(self, shape, generator, device, dtype, latents=None):
        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return noise

    def callback_area(self, callback_on_step_end, i, t, callback_kwargs, latents):
        if True:
            if True:
                if True:
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    image_embeds_1 = callback_outputs.pop(
                        "image_embeds_1", image_embeds_1
                    )
                    negative_image_embeds_1 = callback_outputs.pop(
                        "negative_image_embeds_1", negative_image_embeds_1
                    )
                    image_embeds_2 = callback_outputs.pop(
                        "image_embeds_2", image_embeds_2
                    )
                    negative_image_embeds_2 = callback_outputs.pop(
                        "negative_image_embeds_2", negative_image_embeds_2
                    )

    def flash(self):
        self.vae.set_flash_decoder()
        output, meshes = [], []

    def decode(self, geometric_func, device, latents, bounds, dense_octree_depth,
               hierarchical_octree_depth, max_num_expanded_coords):
        for i in range(1):
            if True:
                try:
                    mesh_v_f = hierarchical_extract_geometry(
                        geometric_func,
                        device,
                        dtype=latents.dtype,
                        bounds=bounds,
                        dense_octree_depth=dense_octree_depth,
                        hierarchical_octree_depth=hierarchical_octree_depth,
                        max_num_expanded_coords=max_num_expanded_coords,
                        # verbose=True
                    )
                    mesh = trimesh.Trimesh(mesh_v_f[0].astype(np.float32), mesh_v_f[1])
                except:
                    mesh_v_f = None
                    mesh = None
'''

VAE_FIXTURE = '''import numpy as np
import torch
from torch_cluster import fps
from tqdm import tqdm

def uses_fps(points, batch):
    return fps(points, batch, ratio=0.25, random_start=False)
'''

INFERENCE_FIXTURE = '''def expand(edge_coords, torch, grid_size, dtype):
    return torch.zeros(grid_size, grid_size, grid_size, device='cuda', dtype=dtype, requires_grad=False)
'''


def write_source_patch_fixture(root: Path) -> None:
    files = {
        "src/pipelines/pipeline_partcrafter.py": PIPELINE_FIXTURE,
        "src/models/autoencoders/autoencoder_kl_triposg.py": VAE_FIXTURE,
        "src/utils/inference_utils.py": INFERENCE_FIXTURE,
    }
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    (root / "LICENSE").write_text("MIT fixture\n", encoding="utf-8")


def write_tar(path: Path, entries: list[tuple[str, bytes, str | None]]) -> None:
    with tarfile.open(path, "w:gz") as bundle:
        for name, payload, linkname in entries:
            member = tarfile.TarInfo(name)
            if linkname is not None:
                member.type = tarfile.SYMTYPE
                member.linkname = linkname
                bundle.addfile(member)
            else:
                member.size = len(payload)
                bundle.addfile(member, io.BytesIO(payload))


class SetupContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            cls.module = load_module("partcrafter_setup_under_test", "setup.py")
        if stdout.getvalue() or stderr.getvalue():
            raise AssertionError("importing setup.py must not run setup or log output")

    def make_context(
        self,
        *,
        gpu_sm: int = 86,
        cuda_version: int = 124,
        accelerator: str = "cuda",
        platform: str = "linux",
        arch: str = "x64",
        ext_dir: Path | None = None,
    ):
        return self.module.SetupContext(
            python_exe=Path(sys.executable),
            ext_dir=ext_dir or Path.cwd(),
            gpu_sm=gpu_sm,
            cuda_version=cuda_version,
            accelerator=accelerator,
            platform=platform,
            arch=arch,
        )

    def test_json_payload_matches_modly_contract_and_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            ext_dir = Path(temp) / "extension"
            config = self.module.parse_setup_args(
                [
                    json.dumps(
                        {
                            "python_exe": sys.executable,
                            "ext_dir": str(ext_dir),
                            "gpu_sm": 100,
                            "cuda_version": "12.8",
                            "accelerator": "cuda",
                            "platform": "linux",
                            "arch": "arm64",
                        }
                    )
                ]
            )
            self.assertEqual(config.python_exe, Path(sys.executable))
            self.assertEqual(config.ext_dir, ext_dir.resolve())
            self.assertEqual(config.gpu_sm, 100)
            self.assertEqual(config.cuda_version, 128)
            self.assertEqual(config.accelerator, "cuda")
            self.assertEqual(config.platform, "linux")
            self.assertEqual(config.arch, "arm64")

    def test_legacy_arguments_remain_supported(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            ext_dir = Path(temp) / "extension"
            for args, expected_cuda in (
                ([sys.executable, str(ext_dir), "86"], 118),
                ([sys.executable, str(ext_dir), "86", "124"], 124),
            ):
                with self.subTest(args=args):
                    with (
                        mock.patch.object(self.module.sys, "platform", "linux"),
                        mock.patch.object(
                            self.module.platform_module,
                            "machine",
                            return_value="x86_64",
                        ),
                    ):
                        config = self.module.parse_setup_args(args)
                    self.assertEqual(config.python_exe, Path(sys.executable))
                    self.assertEqual(config.ext_dir, ext_dir.resolve())
                    self.assertEqual(config.gpu_sm, 86)
                    self.assertEqual(config.cuda_version, expected_cuda)
                    self.assertEqual(config.platform, "linux")
                    self.assertEqual(config.arch, "x64")

    def test_legacy_blackwell_and_arm64_require_explicit_cuda_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            ext_dir = Path(temp) / "extension"
            cases = (("x86_64", "100"), ("aarch64", "90"))
            for machine, gpu_sm in cases:
                with self.subTest(machine=machine, gpu_sm=gpu_sm):
                    with (
                        mock.patch.object(self.module.sys, "platform", "linux"),
                        mock.patch.object(
                            self.module.platform_module,
                            "machine",
                            return_value=machine,
                        ),
                    ):
                        with self.assertRaises(self.module.SetupError) as raised:
                            self.module.parse_setup_args(
                                [sys.executable, str(ext_dir), gpu_sm]
                            )
                    self.assertEqual(raised.exception.code, "CUDA_VERSION_REQUIRED")
                    self.assertIn("JSON payload", raised.exception.public_message)

    def test_json_cuda_version_accepts_numeric_modly_value(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            config = self.module.parse_setup_args(
                [
                    json.dumps(
                        {
                            "python_exe": sys.executable,
                            "ext_dir": temp,
                            "gpu_sm": 100,
                            "cuda_version": 12.8,
                            "accelerator": "cuda",
                            "platform": "linux",
                            "arch": "arm64",
                        }
                    )
                ]
            )
            self.assertEqual(config.cuda_version, 128)

    def test_numeric_setup_values_reject_ambiguous_fractional_forms(self) -> None:
        for value in (99.9, "99.9"):
            with self.subTest(gpu_sm=value), self.assertRaises(
                self.module.SetupError
            ) as raised:
                self.module._parse_int(value, "gpu_sm")
            self.assertEqual(raised.exception.code, "INVALID_ARGUMENT")
        for value in (118.0, "118.0", "12.80", "nonsense"):
            with self.subTest(cuda_version=value), self.assertRaises(
                self.module.SetupError
            ) as raised:
                self.module._parse_cuda_version(value)
            self.assertEqual(raised.exception.code, "INVALID_ARGUMENT")
        for value, expected in (("cu128", 128), ("cuda12.8", 128), (12, 120)):
            with self.subTest(cuda_version=value):
                self.assertEqual(self.module._parse_cuda_version(value), expected)

    def test_invalid_or_incomplete_payload_is_rejected(self) -> None:
        for args in ([], ["{}"], ["not-json"], [sys.executable]):
            with self.subTest(args=args), self.assertRaises(self.module.SetupError):
                self.module.parse_setup_args(args)

    def test_platform_and_arch_aliases_are_normalized(self) -> None:
        expected_platforms = {
            "linux": "linux",
            "linux2": "linux",
            "win32": "win32",
            "windows": "win32",
        }
        for value, expected in expected_platforms.items():
            with self.subTest(platform=value):
                self.assertEqual(self.module._normalize_platform(value), expected)
        expected_arches = {
            "x86_64": "x64",
            "amd64": "x64",
            "x64": "x64",
            "aarch64": "arm64",
            "arm64": "arm64",
        }
        for value, expected in expected_arches.items():
            with self.subTest(arch=value):
                self.assertEqual(self.module._normalize_arch(value), expected)

    def test_pinned_source_constants_are_auditable(self) -> None:
        self.assertEqual(
            self.module.UPSTREAM_COMMIT,
            "3d773bf02fad51c7ab31a5615573fec93b287b30",
        )
        self.assertEqual(
            self.module.UPSTREAM_ARCHIVE_SHA256,
            "d7d1cf92c8d642af134f225ab447ff63b3b4784f1516d0c133c41e7cd0e2ccb6",
        )
        self.assertIn(self.module.UPSTREAM_COMMIT, self.module.UPSTREAM_ARCHIVE_URL)
        self.assertNotIn("huggingface.co", self.module.UPSTREAM_ARCHIVE_URL)

    def test_venv_python_path_is_exact_on_linux_and_windows(self) -> None:
        root = Path("extension") / "venv"
        self.assertEqual(
            self.module._venv_python(root, "linux"), root / "bin" / "python"
        )
        self.assertEqual(
            self.module._venv_python(root, "win32"),
            root / "Scripts" / "python.exe",
        )

    def test_x64_cuda_profiles_are_pinned_per_driver_lane(self) -> None:
        expected_lanes = {118: "cu118", 121: "cu121", 124: "cu124"}
        for platform_name in ("linux", "win32"):
            for cuda_version, lane in expected_lanes.items():
                with self.subTest(platform=platform_name, cuda=cuda_version):
                    profile = self.module.select_torch_profile(
                        self.make_context(
                            platform=platform_name,
                            arch="x64",
                            gpu_sm=86,
                            cuda_version=cuda_version,
                        )
                    )
                    self.assertEqual(profile.torch_version, "2.5.1")
                    self.assertEqual(profile.torchvision_version, "0.20.1")
                    self.assertEqual(profile.cuda_lane, lane)
                    self.assertTrue(profile.torch_index_url.endswith("/" + lane))

    def test_blackwell_x64_uses_pinned_cuda_128_profile(self) -> None:
        for platform_name in ("linux", "win32"):
            for reported_sm in (86, 100, 120):
                with self.subTest(platform=platform_name, reported_sm=reported_sm):
                    profile = self.module.select_torch_profile(
                        self.make_context(
                            platform=platform_name,
                            arch="x64",
                            gpu_sm=reported_sm,
                            cuda_version=128,
                        )
                    )
                    self.assertEqual(profile.torch_version, "2.7.1")
                    self.assertEqual(profile.torchvision_version, "0.22.1")
                    self.assertEqual(profile.cuda_lane, "cu128")

    def test_linux_arm64_sbsa_sm90_or_newer_uses_portable_fps_lane(self) -> None:
        for gpu_sm in (90, 100, 120):
            with self.subTest(gpu_sm=gpu_sm):
                profile = self.module.select_torch_profile(
                    self.make_context(
                        platform="linux",
                        arch="arm64",
                        gpu_sm=gpu_sm,
                        cuda_version=128,
                    )
                )
                self.assertEqual(profile.torch_version, "2.7.1")
                self.assertEqual(profile.torchvision_version, "0.22.1")
                self.assertEqual(profile.cuda_lane, "cu128")

    def test_unsupported_accelerators_platforms_and_cuda_fail_actionably(self) -> None:
        cases = (
            self.make_context(accelerator="cpu", gpu_sm=0, cuda_version=0),
            self.make_context(accelerator="mps", gpu_sm=0, cuda_version=0),
            self.make_context(platform="darwin", arch="arm64"),
            self.make_context(platform="win32", arch="arm64"),
            self.make_context(platform="linux", arch="riscv64"),
            self.make_context(gpu_sm=86, cuda_version=117),
            self.make_context(
                platform="linux", arch="arm64", gpu_sm=90, cuda_version=124
            ),
            self.make_context(
                platform="linux", arch="arm64", gpu_sm=89, cuda_version=128
            ),
            self.make_context(
                platform="linux", arch="x64", gpu_sm=100, cuda_version=124
            ),
        )
        for context in cases:
            with self.subTest(context=context):
                with self.assertRaises(self.module.SetupError) as raised:
                    self.module.select_torch_profile(context)
                self.assertRegex(
                    raised.exception.code,
                    r"^UNSUPPORTED_(?:ACCELERATOR|PLATFORM|CUDA|GPU)$",
                )

    def test_source_patches_apply_all_audited_fixes_and_fail_closed_on_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            source_root = Path(temp) / "partcrafter"
            write_source_patch_fixture(source_root)
            patches = self.module._apply_source_patches(source_root)
            self.assertEqual(
                patches,
                [
                    "honor supplied latents",
                    "repair denoising callback",
                    "honor and reset flash decoder",
                    "propagate mesh decoding errors",
                    "remove hard-coded CUDA device",
                    "provide portable FPS fallback",
                ],
            )
            pipeline = (source_root / "src/pipelines/pipeline_partcrafter.py").read_text(
                encoding="utf-8"
            )
            self.assertIn("if latents is None:", pipeline)
            self.assertNotIn("image_embeds_1", pipeline)
            self.assertIn("callback_on_step_end(self, i, t, callback_kwargs) or {}", pipeline)
            self.assertIn("if use_flash_decoder:", pipeline)
            self.assertIn("self.vae.set_flash_decoder()", pipeline)
            self.assertIn("self.vae.set_default_attn_processor()", pipeline)
            self.assertNotIn("except:", pipeline)
            inference = (source_root / "src/utils/inference_utils.py").read_text(
                encoding="utf-8"
            )
            self.assertIn("device=edge_coords.device", inference)
            vae = (
                source_root
                / "src/models/autoencoders/autoencoder_kl_triposg.py"
            ).read_text(encoding="utf-8")
            self.assertIn("TORCH_CLUSTER_AVAILABLE", vae)
            self.assertIn("_torch_cluster_fps", vae)
            self.assertIn("torch.unique(batch, sorted=True)", vae)

            # Setup never patches an unknown or already-mutated source tree.
            with self.assertRaises(self.module.SetupError) as raised:
                self.module._apply_source_patches(source_root)
            self.assertEqual(raised.exception.code, "SOURCE_DRIFT")

    def test_source_state_detects_idempotent_ready_tree_and_later_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            source_root = Path(temp) / "partcrafter"
            write_source_patch_fixture(source_root)
            patches = self.module._apply_source_patches(source_root)
            tree_hash = self.module._source_tree_hash(source_root)
            (source_root / self.module.SOURCE_STATE_NAME).write_text(
                json.dumps(
                    {
                        "upstream_commit": self.module.UPSTREAM_COMMIT,
                        "archive_sha256": self.module.UPSTREAM_ARCHIVE_SHA256,
                        "patch_version": self.module.SOURCE_PATCH_VERSION,
                        "patches": patches,
                        "tree_sha256": tree_hash,
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                self.module._source_ready(source_root), (True, tree_hash)
            )
            pipeline = source_root / "src/pipelines/pipeline_partcrafter.py"
            pipeline.write_text(
                pipeline.read_text(encoding="utf-8") + "\n# tampered\n",
                encoding="utf-8",
            )
            ready, changed_hash = self.module._source_ready(source_root)
            self.assertFalse(ready)
            self.assertIsNotNone(changed_hash)
            self.assertNotEqual(changed_hash, tree_hash)

    def test_safe_extraction_keeps_only_runtime_source_inside_destination(self) -> None:
        valid = [
            ("PartCrafter-fixture/LICENSE", b"MIT", None),
            (
                "PartCrafter-fixture/src/pipelines/pipeline_partcrafter.py",
                b"pipeline = True\n",
                None,
            ),
            (
                "PartCrafter-fixture/src/models/autoencoders/autoencoder_kl_triposg.py",
                b"vae = True\n",
                None,
            ),
            (
                "PartCrafter-fixture/src/utils/inference_utils.py",
                b"inference = True\n",
                None,
            ),
        ]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "source.tar.gz"
            write_tar(
                archive,
                [
                    ("PartCrafter-fixture/../../escape.py", b"escape", None),
                    ("/absolute.py", b"absolute", None),
                    *valid,
                    ("PartCrafter-fixture/README.md", b"not runtime", None),
                    (
                        "PartCrafter-fixture/pretrained/model.safetensors",
                        b"weights",
                        None,
                    ),
                ],
            )
            destination = root / "extracted"
            self.module._safe_extract_runtime_source(archive, destination)
            self.assertTrue((destination / "LICENSE").is_file())
            self.assertTrue(
                (destination / "src/pipelines/pipeline_partcrafter.py").is_file()
            )
            self.assertFalse((destination / "README.md").exists())
            self.assertFalse((destination / "pretrained").exists())
            self.assertFalse((root / "escape.py").exists())
            self.assertFalse(Path("/absolute.py").exists())

    def test_safe_extraction_rejects_symlinks_and_multiple_roots(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            symlink_archive = root / "symlink.tar.gz"
            write_tar(
                symlink_archive,
                [
                    (
                        "PartCrafter-fixture/src/pipelines/pipeline_partcrafter.py",
                        b"",
                        "../../outside.py",
                    )
                ],
            )
            with self.assertRaises(self.module.SetupError) as raised:
                self.module._safe_extract_runtime_source(
                    symlink_archive, root / "symlink-output"
                )
            self.assertEqual(raised.exception.code, "SOURCE_ARCHIVE_UNSAFE")

            roots_archive = root / "roots.tar.gz"
            write_tar(
                roots_archive,
                [
                    ("first/LICENSE", b"MIT", None),
                    ("second/src/pipelines/pipeline_partcrafter.py", b"x=1\n", None),
                ],
            )
            with self.assertRaises(self.module.SetupError) as raised:
                self.module._safe_extract_runtime_source(
                    roots_archive, root / "roots-output"
                )
            self.assertEqual(raised.exception.code, "SOURCE_ARCHIVE_UNSAFE")

    def test_source_download_verifies_checksum_and_reuses_verified_archive(self) -> None:
        payload = b"small pinned source fixture"

        class Response(io.BytesIO):
            pass

        class Logger:
            def __init__(self) -> None:
                self.lines: list[str] = []

            def log(self, line: str) -> None:
                self.lines.append(line)

        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp) / "source.tar.gz"
            logger = Logger()
            digest = hashlib.sha256(payload).hexdigest()
            with (
                mock.patch.object(
                    self.module, "UPSTREAM_ARCHIVE_SHA256", digest
                ),
                mock.patch.object(
                    self.module.urllib.request,
                    "urlopen",
                    return_value=Response(payload),
                ) as opener,
            ):
                self.module._download_source_archive(destination, logger)
                self.assertEqual(destination.read_bytes(), payload)
                self.module._download_source_archive(destination, logger)
            self.assertEqual(opener.call_count, 1)
            self.assertTrue(any("Reusing verified" in line for line in logger.lines))

    def test_source_download_checksum_mismatch_fails_without_publishing_file(self) -> None:
        class Response(io.BytesIO):
            pass

        logger = mock.Mock()
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp) / "source.tar.gz"
            with mock.patch.object(
                self.module.urllib.request,
                "urlopen",
                return_value=Response(b"wrong archive"),
            ):
                with self.assertRaises(self.module.SetupError) as raised:
                    self.module._download_source_archive(destination, logger)
            self.assertEqual(raised.exception.code, "SOURCE_CHECKSUM_MISMATCH")
            self.assertFalse(destination.exists())

    def test_venv_install_uses_only_the_extension_python_after_creation(self) -> None:
        class Logger:
            def log(self, _line: str) -> None:
                pass

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            venv_dir = root / "venv"
            requirements = root / "requirements-runtime.txt"
            requirements.write_text("example==1.0\n", encoding="utf-8")
            context = self.make_context(ext_dir=root)
            context = self.module.SetupContext(
                python_exe=root / "host-python",
                ext_dir=root,
                gpu_sm=context.gpu_sm,
                cuda_version=context.cuda_version,
                accelerator=context.accelerator,
                platform=context.platform,
                arch=context.arch,
            )
            profile = self.module.select_torch_profile(context)
            commands: list[list[str]] = []

            def fake_run(command, _logger, **_kwargs):
                rendered = [str(part) for part in command]
                commands.append(rendered)
                if rendered[1:3] == ["-m", "venv"]:
                    venv_python = self.module._venv_python(venv_dir, context.platform)
                    venv_python.parent.mkdir(parents=True, exist_ok=True)
                    venv_python.write_text("python fixture", encoding="utf-8")
                return True

            smoke = {
                "torch": "2.5.1+cu124",
                "torch_cuda": "12.4",
                "device": "Fake CUDA",
                "vram_bytes": 12 * 1024**3,
            }
            with (
                mock.patch.object(self.module, "VENV_DIR", venv_dir),
                mock.patch.object(self.module, "REQUIREMENTS_PATH", requirements),
                mock.patch.object(self.module, "_read_status", return_value=None),
                mock.patch.object(self.module, "_run_streamed", side_effect=fake_run),
                mock.patch.object(
                    self.module,
                    "_probe_python",
                    return_value={
                        "version": [3, 11],
                        "implementation": "CPython",
                        "bits": 64,
                        "machine": "x86_64",
                    },
                ),
                mock.patch.object(self.module, "_run_smoke", return_value=smoke),
            ):
                venv_python, evidence, reused = self.module._ensure_venv(
                    context,
                    profile,
                    "source-hash",
                    {
                        "version": [3, 11],
                        "implementation": "CPython",
                        "bits": 64,
                        "machine": "x86_64",
                    },
                    Logger(),
                )

            self.assertFalse(reused)
            self.assertEqual(evidence, smoke)
            self.assertEqual(
                commands[0],
                [str(context.python_exe), "-m", "venv", str(venv_dir)],
            )
            for command in commands[1:]:
                with self.subTest(command=command):
                    self.assertEqual(command[:3], [str(venv_python), "-m", "pip"])
            flattened = [token for command in commands for token in command]
            self.assertIn("torch==2.5.1", flattened)
            self.assertIn("torchvision==0.20.1", flattened)
            self.assertIn(str(requirements), flattened)
            self.assertTrue(
                any(command[-1] == "check" for command in commands),
                "pip check was not run",
            )
            self.assertFalse(
                any("torch-cluster" in token for command in commands for token in command)
            )

    def test_valid_existing_venv_is_reused_without_pip(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            venv_dir = root / "venv"
            venv_python = self.module._venv_python(venv_dir, "linux")
            venv_python.parent.mkdir(parents=True)
            venv_python.write_text("python fixture", encoding="utf-8")
            requirements = root / "requirements-runtime.txt"
            requirements.write_text("example==1.0\n", encoding="utf-8")
            context = self.make_context(ext_dir=root)
            profile = self.module.select_torch_profile(context)
            smoke = {"torch": "2.5.1+cu124", "vram_bytes": 12 * 1024**3}
            requirements_hash = hashlib.sha256(requirements.read_bytes()).hexdigest()
            status = {
                "result": "ready",
                "python_version": [3, 11],
                "profile_id": profile.profile_id,
                "requirements_sha256": requirements_hash,
                "source_tree_sha256": "source-hash",
            }
            logger = mock.Mock()
            with (
                mock.patch.object(self.module, "VENV_DIR", venv_dir),
                mock.patch.object(self.module, "REQUIREMENTS_PATH", requirements),
                mock.patch.object(self.module, "_read_status", return_value=status),
                mock.patch.object(
                    self.module,
                    "_probe_python",
                    return_value={
                        "version": [3, 11],
                        "implementation": "CPython",
                        "bits": 64,
                        "machine": "x86_64",
                    },
                ),
                mock.patch.object(self.module, "_run_smoke", return_value=smoke),
                mock.patch.object(self.module, "_run_streamed") as run_streamed,
            ):
                result = self.module._ensure_venv(
                    context,
                    profile,
                    "source-hash",
                    {
                        "version": [3, 11],
                        "implementation": "CPython",
                        "bits": 64,
                        "machine": "x86_64",
                    },
                    logger,
                )
            self.assertEqual(result, (venv_python, smoke, True))
            run_streamed.assert_not_called()
            self.assertTrue(
                any("passed full runtime validation" in call.args[0] for call in logger.log.call_args_list)
            )

    def test_arm64_venv_uses_fps_fallback_without_native_wheel_install(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            venv_dir = root / "venv"
            requirements = root / "requirements-runtime.txt"
            requirements.write_text("example==1.0\n", encoding="utf-8")
            context = self.make_context(
                platform="linux",
                arch="arm64",
                gpu_sm=90,
                cuda_version=128,
                ext_dir=root,
            )
            profile = self.module.select_torch_profile(context)
            commands: list[list[str]] = []

            def fake_run(command, _logger, **_kwargs):
                rendered = [str(part) for part in command]
                commands.append(rendered)
                if rendered[1:3] == ["-m", "venv"]:
                    python = self.module._venv_python(venv_dir, "linux")
                    python.parent.mkdir(parents=True, exist_ok=True)
                    python.write_text("fixture", encoding="utf-8")
                return True

            with (
                mock.patch.object(self.module, "VENV_DIR", venv_dir),
                mock.patch.object(self.module, "REQUIREMENTS_PATH", requirements),
                mock.patch.object(self.module, "_read_status", return_value=None),
                mock.patch.object(self.module, "_run_streamed", side_effect=fake_run),
                mock.patch.object(
                    self.module,
                    "_probe_python",
                    return_value={
                        "version": [3, 12],
                        "implementation": "CPython",
                        "bits": 64,
                        "machine": "aarch64",
                    },
                ),
                mock.patch.object(
                    self.module,
                    "_run_smoke",
                    return_value={"torch": "2.7.1+cu128", "vram_bytes": 12 * 1024**3},
                ),
            ):
                self.module._ensure_venv(
                    context,
                    profile,
                    "source-hash",
                    {
                        "version": [3, 12],
                        "implementation": "CPython",
                        "bits": 64,
                        "machine": "aarch64",
                    },
                    mock.Mock(),
                )
            installs = [
                command
                for command in commands
                if "install" in command and any("torch-cluster==" in token for token in command)
            ]
            self.assertEqual(installs, [])

    def test_context_accepts_only_supported_64_bit_cpython_versions(self) -> None:
        context = self.make_context(ext_dir=self.module.ROOT)
        for version in ([3, 11], [3, 12]):
            probe = {
                "version": version,
                "implementation": "CPython",
                "platform": "linux",
                "bits": 64,
                "machine": "x86_64",
            }
            with self.subTest(version=version), mock.patch.object(
                self.module, "_probe_python", return_value=probe
            ):
                self.assertEqual(self.module._validate_context(context), probe)

        rejected = (
            {
                "version": [3, 10],
                "implementation": "CPython",
                "bits": 64,
                "machine": "x86_64",
            },
            {
                "version": [3, 13],
                "implementation": "CPython",
                "bits": 64,
                "machine": "x86_64",
            },
            {
                "version": [3, 12],
                "implementation": "PyPy",
                "bits": 64,
                "machine": "x86_64",
            },
            {
                "version": [3, 12],
                "implementation": "CPython",
                "bits": 32,
                "machine": "x86_64",
            },
        )
        for probe in rejected:
            with self.subTest(probe=probe), mock.patch.object(
                self.module, "_probe_python", return_value=probe
            ):
                with self.assertRaises(self.module.SetupError) as raised:
                    self.module._validate_context(context)
                self.assertEqual(raised.exception.code, "UNSUPPORTED_PYTHON")
                self.assertIn("3.11 or 3.12", raised.exception.public_message)

    def test_python_probe_wraps_unusable_interpreter_os_errors(self) -> None:
        with mock.patch.object(
            self.module.subprocess,
            "run",
            side_effect=PermissionError("fixture interpreter is not executable"),
        ):
            with self.assertRaises(self.module.SetupError) as raised:
                self.module._probe_python(Path("broken-python"))
        self.assertEqual(raised.exception.code, "PYTHON_UNUSABLE")
        self.assertIn("not executable", raised.exception.public_message)

    def test_python_minor_change_rebuilds_existing_venv(self) -> None:
        class Logger:
            def __init__(self) -> None:
                self.lines: list[str] = []

            def log(self, line: str) -> None:
                self.lines.append(line)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            venv_dir = root / "venv"
            old_python = self.module._venv_python(venv_dir, "linux")
            old_python.parent.mkdir(parents=True)
            old_python.write_text("CPython 3.11 fixture", encoding="utf-8")
            requirements = root / "requirements-runtime.txt"
            requirements.write_text("example==1.0\n", encoding="utf-8")
            context = self.make_context(ext_dir=root)
            profile = self.module.select_torch_profile(context)
            requirements_hash = hashlib.sha256(requirements.read_bytes()).hexdigest()
            status = {
                "result": "ready",
                "python_version": [3, 11],
                "profile_id": profile.profile_id,
                "requirements_sha256": requirements_hash,
                "source_tree_sha256": "source-hash",
            }
            host_probe = {
                "version": [3, 12],
                "implementation": "CPython",
                "bits": 64,
                "machine": "x86_64",
            }
            commands: list[list[str]] = []

            def fake_run(command, _logger, **_kwargs):
                rendered = [str(part) for part in command]
                commands.append(rendered)
                if rendered[1:3] == ["-m", "venv"]:
                    new_python = self.module._venv_python(venv_dir, "linux")
                    new_python.parent.mkdir(parents=True, exist_ok=True)
                    new_python.write_text("CPython 3.12 fixture", encoding="utf-8")
                return True

            logger = Logger()
            with (
                mock.patch.object(self.module, "VENV_DIR", venv_dir),
                mock.patch.object(self.module, "REQUIREMENTS_PATH", requirements),
                mock.patch.object(self.module, "_read_status", return_value=status),
                mock.patch.object(self.module, "_run_streamed", side_effect=fake_run),
                mock.patch.object(
                    self.module, "_probe_python", return_value=host_probe
                ),
                mock.patch.object(
                    self.module,
                    "_run_smoke",
                    return_value={
                        "torch": "2.5.1+cu124",
                        "torch_cuda": "12.4",
                        "device": "Fake CUDA",
                        "vram_bytes": 12 * 1024**3,
                    },
                ),
            ):
                _, _, reused = self.module._ensure_venv(
                    context,
                    profile,
                    "source-hash",
                    host_probe,
                    logger,
                )

            self.assertFalse(reused)
            self.assertEqual(
                commands[0],
                [str(context.python_exe), "-m", "venv", str(venv_dir)],
            )
            self.assertTrue(
                any("pinned state changed" in line for line in logger.lines)
            )
            self.assertTrue(
                any("CPython 3.12 venv" in line for line in logger.lines)
            )

    def test_unusable_existing_venv_is_rebuilt(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            venv_dir = root / "venv"
            venv_python = self.module._venv_python(venv_dir, "linux")
            venv_python.parent.mkdir(parents=True)
            venv_python.write_text("corrupt fixture", encoding="utf-8")
            requirements = root / "requirements-runtime.txt"
            requirements.write_text("example==1.0\n", encoding="utf-8")
            context = self.make_context(ext_dir=root)
            profile = self.module.select_torch_profile(context)
            requirements_hash = hashlib.sha256(requirements.read_bytes()).hexdigest()
            status = {
                "result": "ready",
                "python_version": [3, 12],
                "profile_id": profile.profile_id,
                "requirements_sha256": requirements_hash,
                "source_tree_sha256": "source-hash",
            }
            host_probe = {
                "version": [3, 12],
                "implementation": "CPython",
                "bits": 64,
                "machine": "x86_64",
            }
            commands: list[list[str]] = []

            def fake_run(command, _logger, **_kwargs):
                rendered = [str(part) for part in command]
                commands.append(rendered)
                if rendered[1:3] == ["-m", "venv"]:
                    new_python = self.module._venv_python(venv_dir, "linux")
                    new_python.parent.mkdir(parents=True, exist_ok=True)
                    new_python.write_text("repaired fixture", encoding="utf-8")
                return True

            logger = mock.Mock()
            with (
                mock.patch.object(self.module, "VENV_DIR", venv_dir),
                mock.patch.object(self.module, "REQUIREMENTS_PATH", requirements),
                mock.patch.object(self.module, "_read_status", return_value=status),
                mock.patch.object(self.module, "_run_streamed", side_effect=fake_run),
                mock.patch.object(
                    self.module,
                    "_probe_python",
                    side_effect=[
                        self.module.SetupError(
                            "PYTHON_UNUSABLE", "existing venv is corrupt"
                        ),
                        host_probe,
                    ],
                ),
                mock.patch.object(
                    self.module,
                    "_run_smoke",
                    return_value={
                        "torch": "2.5.1+cu124",
                        "torch_cuda": "12.4",
                        "device": "Fake CUDA",
                        "vram_bytes": 12 * 1024**3,
                    },
                ),
            ):
                _, _, reused = self.module._ensure_venv(
                    context,
                    profile,
                    "source-hash",
                    host_probe,
                    logger,
                )

            self.assertFalse(reused)
            self.assertEqual(
                commands[0],
                [str(context.python_exe), "-m", "venv", str(venv_dir)],
            )
            self.assertTrue(
                any(
                    "existing venv is corrupt" in call.args[0]
                    for call in logger.log.call_args_list
                )
            )

    def test_context_validation_rejects_path_python_and_arch_mismatches(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            wrong = self.make_context(ext_dir=Path(temp))
            with self.assertRaises(self.module.SetupError) as raised:
                self.module._validate_context(wrong)
            self.assertEqual(raised.exception.code, "EXTENSION_PATH_MISMATCH")

        valid = self.make_context(ext_dir=self.module.ROOT)
        with mock.patch.object(
            self.module,
            "_probe_python",
            return_value={
                "version": [3, 11],
                "implementation": "CPython",
                "bits": 64,
                "machine": "aarch64",
            },
        ):
            with self.assertRaises(self.module.SetupError) as raised:
                self.module._validate_context(valid)
        self.assertEqual(raised.exception.code, "ARCHITECTURE_MISMATCH")

        with mock.patch.object(
            self.module,
            "_probe_python",
            return_value={
                "version": [3, 12],
                "implementation": "CPython",
                "platform": "win32",
                "bits": 64,
                "machine": "x86_64",
            },
        ):
            with self.assertRaises(self.module.SetupError) as raised:
                self.module._validate_context(valid)
        self.assertEqual(raised.exception.code, "PLATFORM_MISMATCH")

    def test_main_streams_actionable_ready_and_failure_status_to_stderr(self) -> None:
        context = self.make_context(ext_dir=self.module.ROOT)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            log_path = root / "logs" / "setup.log"
            status_path = root / "setup-status.json"
            success = {"schema_version": 1, "result": "ready"}
            stdout = io.StringIO()
            stderr = io.StringIO()
            with (
                mock.patch.object(self.module, "LOG_PATH", log_path),
                mock.patch.object(self.module, "STATUS_PATH", status_path),
                mock.patch.object(self.module, "parse_setup_args", return_value=context),
                mock.patch.object(self.module, "run_setup", return_value=success),
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(stderr),
            ):
                code = self.module.main(["fixture"])
            self.assertEqual(code, 0)
            self.assertEqual(stdout.getvalue(), "")
            self.assertIn("READY", stderr.getvalue())
            self.assertIn("weights will not be downloaded", stderr.getvalue())
            self.assertEqual(json.loads(status_path.read_text())["result"], "ready")
            self.assertIn("READY", log_path.read_text(encoding="utf-8"))

            stdout = io.StringIO()
            stderr = io.StringIO()
            failure = self.module.SetupError("TEST_FAILURE", "action required")
            with (
                mock.patch.object(self.module, "LOG_PATH", log_path),
                mock.patch.object(self.module, "STATUS_PATH", status_path),
                mock.patch.object(self.module, "parse_setup_args", return_value=context),
                mock.patch.object(self.module, "run_setup", side_effect=failure),
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(stderr),
            ):
                code = self.module.main(["fixture"])
            status = json.loads(status_path.read_text(encoding="utf-8"))
            self.assertEqual(code, 1)
            self.assertEqual(stdout.getvalue(), "")
            self.assertIn("FAILED [TEST_FAILURE]", stderr.getvalue())
            self.assertIn("choose Repair", stderr.getvalue())
            self.assertEqual(status["result"], "error")
            self.assertEqual(status["code"], "TEST_FAILURE")
            self.assertEqual(status["message"], "action required")


if __name__ == "__main__":
    unittest.main()
