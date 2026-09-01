from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import threading
import traceback
import types
import unittest
from pathlib import Path
from unittest import mock

from tests._support import load_module, minimal_glb, modly_api_fixture


class GeneratorContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = modly_api_fixture()
        cls.base_class, cls.cancelled_error = cls.fixture.__enter__()
        cls.module = load_module("partcrafter_generator_under_test", "generator.py")
        cls.generator_class = cls.module.PartCrafterGenerator

    @classmethod
    def tearDownClass(cls) -> None:
        cls.fixture.__exit__(None, None, None)

    def test_uses_real_modly_base_contract(self) -> None:
        self.assertTrue(issubclass(self.generator_class, self.base_class))
        source = (Path(self.module.__file__)).read_text(encoding="utf-8")
        self.assertIn(
            "from services.generators.base import BaseGenerator, GenerationCancelled",
            source,
        )
        self.assertNotRegex(source, r"(?m)^class\s+BaseGenerator\b")

    def test_constructor_preserves_exact_model_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model_dir = root / "models" / "partcrafter" / "object"
            outputs_dir = root / "outputs"
            generator = self.generator_class(model_dir, outputs_dir)
            self.assertEqual(generator.model_dir, model_dir)
            self.assertEqual(generator.outputs_dir, outputs_dir)

    def test_generator_discovers_from_an_unrelated_working_directory(self) -> None:
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as temp:
            try:
                os.chdir(temp)
                discovered = load_module(
                    "partcrafter_generator_other_cwd", "generator.py"
                )
            finally:
                os.chdir(previous)
        self.assertTrue(issubclass(discovered.PartCrafterGenerator, self.base_class))
        self.assertEqual(discovered.EXTENSION_DIR, Path(discovered.__file__).parent)

    def test_required_weight_validation_is_complete_and_local(self) -> None:
        expected = {
            "model_index.json",
            "feature_extractor_dinov2/preprocessor_config.json",
            "image_encoder_dinov2/config.json",
            "image_encoder_dinov2/model.safetensors",
            "scheduler/scheduler_config.json",
            "transformer/config.json",
            "transformer/diffusion_pytorch_model.safetensors",
            "vae/config.json",
            "vae/diffusion_pytorch_model.safetensors",
        }
        with tempfile.TemporaryDirectory() as temp:
            model_dir = Path(temp) / "exact-model-dir"
            model_dir.mkdir()
            errors = self.module._model_validation_errors(model_dir)
            for relative in expected:
                with self.subTest(path=relative):
                    self.assertTrue(
                        any(relative in error for error in errors),
                        f"missing validation for {relative}",
                    )

            compact_requirements = tuple((relative, 1) for relative in expected)
            for relative in expected:
                path = model_dir / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                if relative.endswith(".safetensors"):
                    header = json.dumps(
                        {
                            "x": {
                                "dtype": "F32",
                                "shape": [1],
                                "data_offsets": [0, 4],
                            }
                        }
                    ).encode("utf-8")
                    path.write_bytes(len(header).to_bytes(8, "little") + header + b"\0" * 4)
                elif relative.endswith(".json"):
                    path.write_text("{}", encoding="utf-8")
                else:
                    path.write_bytes(b"x")
            index = {
                "_class_name": "PartCrafterPipeline",
                **{component: ["src.test", "Test"] for component in self.module._MODEL_INDEX_COMPONENTS},
            }
            (model_dir / "model_index.json").write_text(
                json.dumps(index), encoding="utf-8"
            )
            with mock.patch.object(
                self.module, "REQUIRED_MODEL_FILES", compact_requirements
            ):
                self.assertEqual(self.module._model_validation_errors(model_dir), [])

            # The production thresholds reject a Git-LFS pointer or truncated
            # checkpoint even when every filename exists.
            errors = self.module._model_validation_errors(model_dir)
            self.assertTrue(any("incomplete" in error for error in errors))

    def test_model_index_requires_a_json_object(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            model_dir = Path(temp)
            (model_dir / "model_index.json").write_text("[]", encoding="utf-8")
            with mock.patch.object(
                self.module,
                "REQUIRED_MODEL_FILES",
                (("model_index.json", 1),),
            ):
                errors = self.module._model_validation_errors(model_dir)
            self.assertTrue(any("JSON object" in error for error in errors))

    def test_rmbg_weights_validate_config_exact_size_and_safetensors_header(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            model_dir = Path(temp)
            (model_dir / "config.json").write_text(
                json.dumps({"in_ch": 3, "out_ch": 1}), encoding="utf-8"
            )
            header = json.dumps(
                {
                    "x": {
                        "dtype": "F32",
                        "shape": [1],
                        "data_offsets": [0, 4],
                    }
                }
            ).encode("utf-8")
            weights = model_dir / "model.safetensors"
            with weights.open("wb") as stream:
                stream.write(len(header).to_bytes(8, "little"))
                stream.write(header)
                stream.seek(self.module.RMBG_MODEL_SIZE - 1)
                stream.write(b"\0")
            self.assertEqual(self.module._rmbg_validation_errors(model_dir), [])

            weights.write_bytes(b"version https://git-lfs.github.com/spec/v1\n")
            errors = self.module._rmbg_validation_errors(model_dir)
            self.assertTrue(any("model.safetensors" in error for error in errors))

            (model_dir / "config.json").write_text("{}", encoding="utf-8")
            errors = self.module._rmbg_validation_errors(model_dir)
            self.assertTrue(any("in_ch=3" in error for error in errors))

    def test_missing_weights_fail_with_ui_action_and_without_network(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            generator = self.generator_class(
                root / "models" / "partcrafter" / "object", root / "outputs"
            )
            generator.hf_repo = "wgsxm/PartCrafter"
            with self.assertRaises(RuntimeError) as raised:
                generator.load()
            message = str(raised.exception)
            self.assertIn("wgsxm/PartCrafter", message)
            self.assertIn(str(generator.model_dir), message)
            self.assertRegex(message.lower(), r"models? ui|ui.*models?")

    def test_load_uses_exact_model_dir_and_local_only(self) -> None:
        class DeviceProperties:
            total_memory = 12 * 2**30

        cuda = types.SimpleNamespace(
            is_available=lambda: True,
            get_device_properties=lambda _index: DeviceProperties(),
        )
        fake_torch = types.ModuleType("torch")
        fake_torch.float16 = object()
        fake_torch.cuda = cuda

        calls: dict[str, object] = {}

        class FakePipeline:
            vae = object()
            transformer = object()
            scheduler = object()
            image_encoder_dinov2 = object()
            feature_extractor_dinov2 = object()

            @classmethod
            def from_pretrained(cls, path, **kwargs):
                calls["path"] = path
                calls["kwargs"] = kwargs
                return cls()

            def to(self, **kwargs):
                calls["to"] = kwargs
                return self

            def set_progress_bar_config(self, **kwargs):
                calls["progress"] = kwargs

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model_dir = root / "models" / "partcrafter" / "scene"
            generator = self.generator_class(model_dir, root / "outputs")
            with (
                mock.patch.object(self.module, "_model_validation_errors", return_value=[]),
                mock.patch.object(self.module, "_require_upstream_source"),
                mock.patch.object(
                    self.module, "_load_pipeline_class", return_value=FakePipeline
                ),
                mock.patch.dict(sys.modules, {"torch": fake_torch}),
            ):
                generator.load()

            self.assertEqual(calls["path"], str(model_dir))
            self.assertEqual(calls["kwargs"]["local_files_only"], True)
            self.assertIs(calls["kwargs"]["torch_dtype"], fake_torch.float16)
            self.assertEqual(calls["to"], {"device": "cuda", "dtype": fake_torch.float16})
            self.assertEqual(calls["progress"], {"disable": True})
            self.assertIsInstance(generator._model, FakePipeline)

    def test_rmbg_load_uses_upstream_class_exact_dir_and_local_only(self) -> None:
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = types.SimpleNamespace(is_available=lambda: True)
        calls: dict[str, object] = {}

        class FakeRMBG:
            @classmethod
            def from_pretrained(cls, path, **kwargs):
                calls["path"] = path
                calls["kwargs"] = kwargs
                return cls()

            def to(self, **kwargs):
                calls["to"] = kwargs
                return self

            def eval(self):
                calls["eval"] = True
                return self

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model_dir = root / "models" / "partcrafter" / "rmbg"
            generator = self.generator_class(model_dir, root / "outputs")
            with (
                mock.patch.object(
                    self.module, "_rmbg_validation_errors", return_value=[]
                ),
                mock.patch.object(self.module, "_require_upstream_source"),
                mock.patch.object(
                    self.module, "_load_rmbg_class", return_value=FakeRMBG
                ),
                mock.patch.dict(sys.modules, {"torch": fake_torch}),
            ):
                generator.load()

        self.assertEqual(calls["path"], str(model_dir))
        self.assertEqual(calls["kwargs"], {"local_files_only": True})
        self.assertEqual(calls["to"], {"device": "cuda"})
        self.assertTrue(calls["eval"])
        self.assertIsInstance(generator._model, FakeRMBG)

    def test_load_rejects_non_cuda_without_loading_pipeline(self) -> None:
        fake_torch = types.ModuleType("torch")
        fake_torch.float16 = object()
        fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            generator = self.generator_class(root / "partcrafter" / "object", root / "outputs")
            with (
                mock.patch.object(self.module, "_model_validation_errors", return_value=[]),
                mock.patch.object(self.module, "_require_upstream_source"),
                mock.patch.object(self.module, "_load_pipeline_class") as loader,
                mock.patch.dict(sys.modules, {"torch": fake_torch}),
            ):
                with self.assertRaises(RuntimeError) as raised:
                    generator.load()
            self.assertIn("NVIDIA CUDA", str(raised.exception))
            loader.assert_not_called()

    def test_inherited_weight_download_is_explicitly_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            generator = self.generator_class(root / "partcrafter" / "object", root / "outputs")
            with self.assertRaises(RuntimeError) as raised:
                generator._auto_download()
            self.assertRegex(str(raised.exception).lower(), r"ui|download")

    def test_hugging_face_runtime_is_forced_offline(self) -> None:
        for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "DIFFUSERS_OFFLINE"):
            with self.subTest(variable=name):
                self.assertEqual(os.environ.get(name), "1")

    def test_optional_gemini_features_are_explicit_and_secret_safe(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            generator = self.generator_class(root / "partcrafter" / "object", root / "outputs")
            input_path = root / "input.png"
            input_path.write_bytes(b"input")
            styled_path = root / "styled.png"
            with mock.patch.dict(
                os.environ,
                {"GEMINI_API_KEY": "", "GOOGLE_API_KEY": ""},
                clear=False,
            ):
                self.assertEqual(
                    generator._resolve_part_count(
                        input_path,
                        self.module._parse_params({"num_parts": 5}, "object"),
                        "object",
                    ),
                    5,
                )
                with self.assertRaises(RuntimeError) as raised:
                    generator._resolve_part_count(
                        input_path,
                        self.module._parse_params(
                            {"part_count_mode": "gemini"}, "object"
                        ),
                        "object",
                    )
                self.assertIn("GEMINI_API_KEY", str(raised.exception))
                selected, warning = generator._style_input(
                    input_path, styled_path, "gemini-test"
                )
                self.assertEqual(selected, input_path)
                self.assertIsNotNone(warning)
                self.assertFalse(styled_path.exists())

            secret = "secret-value-123"
            with mock.patch.dict(os.environ, {"GEMINI_API_KEY": secret}):
                sanitized = self.module._safe_error(RuntimeError(f"failure {secret}"))
            self.assertNotIn(secret, sanitized)
            self.assertIn("[REDACTED]", sanitized)

    def test_gemini_failure_traceback_does_not_cross_protocol_with_secret(self) -> None:
        secret = "protocol-secret-value-123"
        source = Path("input.png")
        generator = self.generator_class(Path("models/partcrafter/object"), Path("out"))
        settings = self.module._parse_params(
            {"part_count_mode": "gemini"}, "object"
        )
        fake_vlm = types.ModuleType("src.utils.vlm_utils")

        def fail(*_args, **_kwargs):
            raise RuntimeError(f"provider exposed {secret}")

        fake_vlm.suggest_num_parts = fail
        with (
            mock.patch.dict(os.environ, {"GEMINI_API_KEY": secret}),
            mock.patch.dict(sys.modules, {"src.utils.vlm_utils": fake_vlm}),
        ):
            try:
                generator._resolve_part_count(source, settings, "object")
            except RuntimeError:
                rendered = traceback.format_exc()
            else:
                self.fail("Gemini failure was not propagated")
        self.assertNotIn(secret, rendered)
        self.assertIn("[REDACTED]", rendered)

    def test_parameter_parsing_matches_manifest_defaults_and_upstream_ranges(self) -> None:
        object_values = self.module._parse_params({}, "object")
        scene_values = self.module._parse_params({}, "scene")
        self.assertEqual(object_values["num_parts"], 3)
        self.assertEqual(scene_values["num_parts"], 6)
        self.assertEqual(object_values["num_tokens"], 1024)
        self.assertEqual(scene_values["num_tokens"], 2048)
        self.assertEqual(object_values["num_inference_steps"], 50)
        self.assertEqual(object_values["guidance_scale"], 7.0)
        self.assertEqual(object_values["max_num_expanded_coords"], 1_000_000_000)
        self.assertFalse(object_values["use_flash_decoder"])
        self.assertFalse(object_values["render"])
        self.assertFalse(object_values["remove_background"])
        self.assertFalse(scene_values["remove_background"])
        self.assertTrue(
            self.module._parse_params(
                {"remove_background": "true"}, "object"
            )["remove_background"]
        )
        self.assertFalse(
            self.module._parse_params(
                {"remove_background": "not-visible"}, "scene"
            )["remove_background"]
        )
        self.assertTrue(
            self.module._parse_params({"render": "true"}, "object")["render"]
        )
        self.assertEqual(
            self.module._parse_params({"max_num_expanded_coords": 0}, "object")[
                "max_num_expanded_coords"
            ],
            0,
        )
        self.assertEqual(
            self.module._parse_params(
                {"seed": -1, "guidance_scale": -2.5}, "object"
            )["seed"],
            -1,
        )
        self.assertEqual(
            self.module._parse_params(
                {"seed": -1, "guidance_scale": -2.5}, "object"
            )["guidance_scale"],
            -2.5,
        )
        self.assertEqual(
            self.module._parse_params(
                {"scene_num_tokens": 8_192}, "scene"
            )["num_tokens"],
            8_192,
        )

        with self.assertRaises((TypeError, ValueError)):
            self.module._parse_params({"num_parts": 17}, "object")
        with self.assertRaises((TypeError, ValueError)):
            self.module._parse_params({"scene_num_parts": 9}, "scene")
        with self.assertRaises((TypeError, ValueError)):
            self.module._parse_params({}, "unknown")

        gemini_values = self.module._parse_params(
            {
                "part_count_mode": "gemini",
                "num_parts": 99,
                "part_model": "gemini-custom",
                "style_transfer": "false",
                "style_model": "ignored\nwhile-disabled",
            },
            "object",
        )
        self.assertEqual(gemini_values["num_parts"], 3)
        self.assertEqual(gemini_values["part_model"], "gemini-custom")
        self.assertFalse(gemini_values["style_transfer"])

    def test_rmbg_decoder_preserves_source_alpha(self) -> None:
        from PIL import Image

        source = Image.new("RGBA", (4, 4), (20, 30, 40, 0))
        source.putpixel((2, 2), (20, 30, 40, 255))
        buffer = io.BytesIO()
        source.save(buffer, format="PNG")
        decoded = self.module._decode_rmbg_input_image(buffer.getvalue())
        self.assertEqual(decoded.mode, "RGBA")
        self.assertEqual(decoded.getpixel((0, 0))[3], 0)
        self.assertEqual(decoded.getpixel((2, 2))[3], 255)

    def test_prepare_with_rmbg_calls_exact_upstream_public_function(self) -> None:
        import numpy as np
        from PIL import Image

        calls: dict[str, object] = {}
        expected = Image.new("RGB", (8, 8), "white")

        def prepare_image(path, **kwargs):
            calls["path"] = path
            calls["kwargs"] = kwargs
            return expected

        class InferenceMode:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

        fake_image_utils = types.ModuleType("src.utils.image_utils")
        fake_image_utils.prepare_image = prepare_image
        fake_src = types.ModuleType("src")
        fake_src.__path__ = []
        fake_utils = types.ModuleType("src.utils")
        fake_utils.__path__ = []
        fake_torch = types.SimpleNamespace(inference_mode=InferenceMode)

        with tempfile.TemporaryDirectory() as temp, mock.patch.dict(
            sys.modules,
            {
                "src": fake_src,
                "src.utils": fake_utils,
                "src.utils.image_utils": fake_image_utils,
            },
        ):
            root = Path(temp)
            input_path = root / "input.png"
            input_path.write_bytes(b"PNG")
            generator = self.generator_class(root / "partcrafter" / "rmbg", root)
            generator._torch = fake_torch
            model = object()
            result = generator._prepare_with_rmbg(input_path, model)

        self.assertIs(result, expected)
        self.assertEqual(calls["path"], str(input_path))
        kwargs = calls["kwargs"]
        np.testing.assert_array_equal(kwargs["bg_color"], np.array([1.0, 1.0, 1.0]))
        self.assertIs(kwargs["rmbg_net"], model)
        self.assertEqual(kwargs["padding_ratio"], 0.1)
        self.assertEqual(kwargs["device"], "cuda")

    def test_mesh_validation_rejects_missing_parts_and_wrong_count(self) -> None:
        fake_numpy = types.ModuleType("numpy")
        fake_trimesh = types.ModuleType("trimesh")
        fake_trimesh.Trimesh = type("Trimesh", (), {})
        with mock.patch.dict(
            sys.modules, {"numpy": fake_numpy, "trimesh": fake_trimesh}
        ):
            with self.assertRaises(RuntimeError):
                self.module._validate_meshes([None], 1)
            with self.assertRaises(RuntimeError):
                self.module._validate_meshes([], 1)

    def test_output_stem_is_sanitized_and_atomic_writer_publishes_bytes(self) -> None:
        self.assertEqual(
            self.module._safe_output_stem("  My unsafe/name?!  "),
            "My-unsafe-name",
        )
        self.assertEqual(self.module._safe_output_stem("CON"), "partcrafter-CON")
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp) / "nested" / "result.glb"
            self.module._atomic_write_bytes(destination, b"glTF-test")
            self.assertEqual(destination.read_bytes(), b"glTF-test")
            self.assertEqual(list(destination.parent.glob("*.tmp")), [])

    def test_generate_honours_cancellation_before_work(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            generator = self.generator_class(root / "partcrafter" / "object", root / "outputs")
            generator._model = object()
            event = threading.Event()
            event.set()
            with self.assertRaises(self.cancelled_error):
                generator.generate(b"not-decoded", {}, cancel_event=event)

    def test_pipeline_double_receives_upstream_parameters_and_reports_steps(self) -> None:
        class InferenceMode:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

        class TorchGenerator:
            def __init__(self, device):
                self.device = device
                self.seed = None

            def manual_seed(self, seed):
                self.seed = seed
                return self

        seeds: list[int] = []
        fake_torch = types.SimpleNamespace(
            Generator=TorchGenerator,
            manual_seed=lambda seed: seeds.append(seed),
            cuda=types.SimpleNamespace(manual_seed_all=lambda seed: seeds.append(seed)),
            inference_mode=InferenceMode,
        )
        calls: dict[str, object] = {}

        class Result:
            meshes = [object(), object()]

        class FakePipeline:
            def __call__(self, **kwargs):
                calls.update(kwargs)
                for index in range(kwargs["num_inference_steps"]):
                    returned = kwargs["callback_on_step_end"](
                        self, index, None, {"latents": object()}
                    )
                    self.assert_callback_result = returned
                return Result()

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            generator = self.generator_class(root / "partcrafter" / "object", root / "outputs")
            generator._torch = fake_torch
            generator._model = FakePipeline()
            settings = self.module._parse_params(
                {
                    "num_parts": 2,
                    "num_tokens": 1536,
                    "num_inference_steps": 3,
                    "guidance_scale": 8.5,
                    "max_num_expanded_coords": 123456,
                    "use_flash_decoder": "true",
                    "seed": 42,
                },
                "object",
            )
            progress: list[tuple[int, str]] = []
            expected_meshes = Result.meshes
            with mock.patch.object(
                self.module, "_validate_meshes", return_value=expected_meshes
            ) as validator:
                meshes = generator._run_pipeline(
                    object(), settings, 2, lambda pct, label: progress.append((pct, label)), None
                )

        self.assertIs(meshes, expected_meshes)
        self.assertEqual(len(calls["image"]), 2)
        self.assertEqual(calls["attention_kwargs"], {"num_parts": 2})
        self.assertEqual(calls["num_tokens"], 1536)
        self.assertEqual(calls["num_inference_steps"], 3)
        self.assertEqual(calls["guidance_scale"], 8.5)
        self.assertEqual(calls["max_num_expanded_coords"], 123456)
        self.assertTrue(calls["use_flash_decoder"])
        self.assertEqual(calls["callback_on_step_end_tensor_inputs"], ["latents"])
        self.assertEqual(calls["generator"].device, "cuda")
        self.assertEqual(calls["generator"].seed, 42)
        self.assertEqual(seeds, [42, 42])
        self.assertEqual([pct for pct, _ in progress], sorted(pct for pct, _ in progress))
        validator.assert_called_once_with(expected_meshes, 2)

    def test_pipeline_callback_honours_mid_generation_cancellation(self) -> None:
        class InferenceMode:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

        class TorchGenerator:
            def __init__(self, device=None):
                pass

            def manual_seed(self, _seed):
                return self

        event = threading.Event()

        class FakePipeline:
            def __call__(self, **kwargs):
                event.set()
                kwargs["callback_on_step_end"](
                    self, 0, None, {"latents": object()}
                )
                raise AssertionError("callback should have cancelled generation")

        fake_torch = types.SimpleNamespace(
            Generator=TorchGenerator,
            manual_seed=lambda _seed: None,
            cuda=types.SimpleNamespace(manual_seed_all=lambda _seed: None),
            inference_mode=InferenceMode,
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            generator = self.generator_class(root / "partcrafter" / "object", root / "outputs")
            generator._torch = fake_torch
            generator._model = FakePipeline()
            with self.assertRaises(self.cancelled_error):
                generator._run_pipeline(
                    object(), self.module._parse_params({}, "object"), 1, None, event
                )

    def test_render_loader_selects_egl_on_linux_but_not_windows(self) -> None:
        fake_module = types.SimpleNamespace(
            render_views_around_mesh=lambda *_args, **_kwargs: None,
            render_normal_views_around_mesh=lambda *_args, **_kwargs: None,
            make_grid_for_images_or_videos=lambda *_args, **_kwargs: None,
            export_renderings=lambda *_args, **_kwargs: None,
        )

        linux_seen: list[str | None] = []

        def linux_import(_name):
            linux_seen.append(os.environ.get("PYOPENGL_PLATFORM"))
            return fake_module

        with (
            mock.patch.object(self.module.sys, "platform", "linux"),
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.object(
                self.module.importlib, "import_module", side_effect=linux_import
            ),
        ):
            self.assertIs(self.module._load_render_utils(), fake_module)
            self.assertIsNone(os.environ.get("PYOPENGL_PLATFORM"))
        self.assertEqual(linux_seen, ["egl"])

        windows_seen: list[str | None] = []

        def windows_import(_name):
            windows_seen.append(os.environ.get("PYOPENGL_PLATFORM"))
            # Simulate upstream's late environment assignment.
            os.environ["PYOPENGL_PLATFORM"] = "egl"
            return fake_module

        with (
            mock.patch.object(self.module.sys, "platform", "win32"),
            mock.patch.dict(os.environ, {"PYOPENGL_PLATFORM": "egl"}, clear=True),
            mock.patch.object(
                self.module.importlib, "import_module", side_effect=windows_import
            ),
        ):
            self.assertIs(self.module._load_render_utils(), fake_module)
            self.assertEqual(os.environ.get("PYOPENGL_PLATFORM"), "egl")
        self.assertEqual(windows_seen, [None])

    def test_render_sidecars_use_exact_upstream_cli_values_and_no_stdout(self) -> None:
        calls: dict[str, object] = {}
        platforms_seen: list[str | None] = []

        class Frame:
            def save(self, path):
                Path(path).write_bytes(b"PNG-render")

        color_frames = [Frame() for _ in range(36)]
        normal_frames = [Frame() for _ in range(36)]
        grid_frames = [Frame() for _ in range(36)]

        def render_color(scene, **kwargs):
            platforms_seen.append(os.environ.get("PYOPENGL_PLATFORM"))
            calls["color"] = (scene, kwargs)
            return color_frames

        def render_normal(scene, **kwargs):
            platforms_seen.append(os.environ.get("PYOPENGL_PLATFORM"))
            calls["normal"] = (scene, kwargs)
            return normal_frames

        def make_grid(inputs, **kwargs):
            calls["grid"] = (inputs, kwargs)
            return grid_frames

        exports: list[tuple[object, str, int]] = []

        def export(frames, path, fps):
            platforms_seen.append(os.environ.get("PYOPENGL_PLATFORM"))
            exports.append((frames, Path(path).name, fps))
            Path(path).write_bytes(b"GIF89a-render")

        fake_utils = types.SimpleNamespace(
            render_views_around_mesh=render_color,
            render_normal_views_around_mesh=render_normal,
            make_grid_for_images_or_videos=make_grid,
            export_renderings=export,
        )
        scene = object()
        processed_image = object()
        stdout = io.StringIO()
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(
            self.module.sys, "platform", "win32"
        ), mock.patch.dict(
            os.environ, {"PYOPENGL_PLATFORM": "egl"}, clear=True
        ), mock.patch.object(
            self.module, "_load_render_utils", return_value=fake_utils
        ), contextlib.redirect_stdout(stdout):
            root = Path(temp)
            files = self.module._render_sidecars(scene, processed_image, root)
            self.assertEqual(set(files.values()), {
                "rendering.gif",
                "rendering_normal.gif",
                "rendering_grid.gif",
                "rendering.png",
                "rendering_normal.png",
                "rendering_grid.png",
            })
            self.assertTrue(all((root / name).is_file() for name in files.values()))
            self.assertEqual(os.environ.get("PYOPENGL_PLATFORM"), "egl")

        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(platforms_seen, [None, None, None, None, None])
        self.assertEqual(calls["color"], (scene, {"num_views": 36, "radius": 4}))
        self.assertEqual(calls["normal"], (scene, {"num_views": 36, "radius": 4}))
        grid_inputs, grid_kwargs = calls["grid"]
        self.assertEqual(grid_kwargs, {"nrow": 3})
        self.assertEqual(len(grid_inputs), 3)
        self.assertTrue(all(item is processed_image for item in grid_inputs[0]))
        self.assertEqual(
            [(name, fps) for _frames, name, fps in exports],
            [
                ("rendering.gif", 18),
                ("rendering_normal.gif", 18),
                ("rendering_grid.gif", 18),
            ],
        )

    def test_render_failure_is_actionable(self) -> None:
        with (
            tempfile.TemporaryDirectory() as temp,
            mock.patch.object(self.module.sys, "platform", "linux"),
            mock.patch.object(
                self.module,
                "_load_render_utils",
                side_effect=ImportError("Unable to load EGL"),
            ),
        ):
            with self.assertRaises(RuntimeError) as raised:
                self.module._render_sidecars(object(), object(), Path(temp))
        message = str(raised.exception)
        self.assertIn("turntable rendering failed", message)
        self.assertIn("libegl1", message)

    def test_generation_double_writes_valid_primary_and_sidecars(self) -> None:
        class PreparedImage:
            def save(self, path, format=None):
                self.saved_format = format
                Path(path).write_bytes(b"PNG")

        class Mesh:
            vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
            faces = [[0, 1, 2]]

        render_files = {
            "color_gif": "rendering.gif",
            "normal_gif": "rendering_normal.gif",
            "grid_gif": "rendering_grid.gif",
            "color_png": "rendering.png",
            "normal_png": "rendering_normal.png",
            "grid_png": "rendering_grid.png",
        }

        def fake_render(_scene, _image, output_dir):
            for filename in render_files.values():
                Path(output_dir, filename).write_bytes(b"render")
            return render_files

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            outputs = root / "outputs"
            generator = self.generator_class(root / "partcrafter" / "object", outputs)
            generator._model = object()
            progress: list[tuple[int, str]] = []
            stderr = io.StringIO()
            stdout = io.StringIO()
            with (
                mock.patch.object(
                    self.module, "_decode_input_image", return_value=PreparedImage()
                ),
                mock.patch.object(
                    self.module, "_open_rgb_image", return_value=PreparedImage()
                ),
                mock.patch.object(
                    generator, "_run_pipeline", return_value=[Mesh(), Mesh()]
                ),
                mock.patch.object(self.module, "_build_colored_scene", return_value=object()),
                mock.patch.object(self.module, "_glb_bytes", return_value=minimal_glb()),
                mock.patch.object(self.module, "_validate_composite_glb"),
                mock.patch.object(
                    self.module, "_render_sidecars", side_effect=fake_render
                ) as renderer,
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(stderr),
            ):
                output = generator.generate(
                    b"image bytes",
                    {"num_parts": 2, "output_name": "smoke", "render": "true"},
                    lambda pct, label: progress.append((pct, label)),
                )

            self.assertTrue(output.is_absolute())
            self.assertTrue(output.is_file())
            self.assertEqual(output.read_bytes(), minimal_glb())
            self.assertEqual(stdout.getvalue(), "")
            self.assertIn("Generated 2 named meshes", stderr.getvalue())
            run_dir = output.parent
            self.assertEqual(output.name, "smoke.glb")
            self.assertTrue((run_dir / "part_00.glb").is_file())
            self.assertTrue((run_dir / "part_01.glb").is_file())
            metadata = json.loads((run_dir / "generation.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["adapter"]["author"], "DrHepa")
            self.assertEqual(metadata["generation"]["num_parts"], 2)
            self.assertTrue(metadata["generation"]["render_requested"])
            self.assertEqual(metadata["outputs"]["renderings"], render_files)
            self.assertTrue(
                all((run_dir / filename).is_file() for filename in render_files.values())
            )
            renderer.assert_called_once()
            self.assertEqual(
                [part["name"] for part in metadata["outputs"]["parts"]],
                ["part_00", "part_01"],
            )
            percentages = [pct for pct, _label in progress]
            self.assertEqual(percentages, sorted(percentages))
            self.assertEqual(percentages[-1], 100)
            self.assertFalse(any(path.name.endswith(".tmp") for path in outputs.iterdir()))

    def test_standalone_rmbg_returns_atomic_png_for_workflows(self) -> None:
        from PIL import Image

        source = Image.new("RGBA", (8, 8), (100, 80, 60, 0))
        source.putpixel((4, 4), (100, 80, 60, 255))
        input_buffer = io.BytesIO()
        source.save(input_buffer, format="PNG")
        prepared = Image.new("RGB", (10, 10), (255, 255, 255))

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            outputs = root / "outputs"
            generator = self.generator_class(
                root / "models" / "partcrafter" / "rmbg", outputs
            )
            generator._model = object()
            progress: list[tuple[int, str]] = []
            with mock.patch.object(
                generator, "_prepare_with_rmbg", return_value=prepared
            ) as preprocessor:
                output = generator.generate(
                    input_buffer.getvalue(),
                    {},
                    lambda pct, step: progress.append((pct, step)),
                )

            self.assertTrue(output.is_absolute())
            self.assertEqual(output.suffix, ".png")
            self.assertTrue(output.is_file())
            with Image.open(output) as result:
                result.load()
                self.assertEqual(result.mode, "RGB")
                self.assertEqual(result.size, (10, 10))
            input_path = preprocessor.call_args.args[0]
            self.assertFalse(input_path.exists())
            self.assertEqual(progress[-1][0], 100)
            self.assertFalse(any(path.name.startswith(".") for path in outputs.iterdir()))

    def test_object_order_is_style_then_suggestion_then_rmbg_then_pipeline(self) -> None:
        class PreparedImage:
            def save(self, path, format=None):
                Path(path).write_bytes(b"PNG")

        class Mesh:
            vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
            faces = [[0, 1, 2]]

        order: list[str] = []
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            generator = self.generator_class(
                root / "models" / "partcrafter" / "object", root / "outputs"
            )
            generator._model = object()

            def style(_input, styled, _model):
                order.append("style")
                Path(styled).write_bytes(b"PNG-styled")
                return Path(styled), None

            def suggest(*_args):
                order.append("suggest")
                return 2

            def rmbg(*_args):
                order.append("rmbg")
                return PreparedImage()

            def pipeline(*_args):
                order.append("pipeline")
                return [Mesh(), Mesh()]

            with (
                mock.patch.object(
                    self.module,
                    "_decode_rmbg_input_image",
                    return_value=PreparedImage(),
                ),
                mock.patch.object(generator, "_style_input", side_effect=style),
                mock.patch.object(
                    generator, "_resolve_part_count", side_effect=suggest
                ),
                mock.patch.object(
                    generator, "_get_auxiliary_rmbg_model", return_value=object()
                ),
                mock.patch.object(generator, "_prepare_with_rmbg", side_effect=rmbg),
                mock.patch.object(generator, "_run_pipeline", side_effect=pipeline),
                mock.patch.object(
                    self.module, "_build_colored_scene", return_value=object()
                ),
                mock.patch.object(
                    self.module, "_glb_bytes", return_value=minimal_glb()
                ),
                mock.patch.object(self.module, "_validate_composite_glb"),
            ):
                generator.generate(
                    b"image",
                    {
                        "part_count_mode": "gemini",
                        "style_transfer": "true",
                        "remove_background": "true",
                    },
                )

        self.assertEqual(order, ["style", "suggest", "rmbg", "pipeline"])

    def test_object_rmbg_missing_sibling_has_download_action(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model_dir = root / "models" / "partcrafter" / "object"
            generator = self.generator_class(model_dir, root / "outputs")
            with mock.patch.object(
                self.module,
                "_rmbg_validation_errors",
                return_value=["missing model.safetensors"],
            ):
                with self.assertRaises(RuntimeError) as raised:
                    generator._get_auxiliary_rmbg_model()
            message = str(raised.exception)
            self.assertIn(str(model_dir.parent / "rmbg"), message)
            self.assertIn("Download the PartCrafter RMBG Preprocess node", message)
            self.assertIn("Models UI", message)

    def test_generation_failure_reports_stage_and_removes_temporary_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            outputs = root / "outputs"
            generator = self.generator_class(root / "partcrafter" / "scene", outputs)
            generator._model = object()
            stdout = io.StringIO()
            stderr = io.StringIO()
            with (
                mock.patch.object(
                    self.module,
                    "_decode_input_image",
                    side_effect=ValueError("invalid pixels"),
                ),
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(stderr),
            ):
                with self.assertRaises(RuntimeError) as raised:
                    generator.generate(b"bad image", {})
            self.assertIn("input preparation", str(raised.exception))
            self.assertIn("invalid pixels", str(raised.exception))
            self.assertEqual(stdout.getvalue(), "")
            self.assertIn("Generation failed during input preparation", stderr.getvalue())
            self.assertTrue(outputs.is_dir())
            self.assertEqual(list(outputs.iterdir()), [])

    def test_logs_and_upstream_prints_never_use_protocol_stdout(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            self.module._log("diagnostic")
            with self.module._upstream_output_to_stderr():
                print("upstream diagnostic")
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("diagnostic", stderr.getvalue())
        self.assertIn("upstream diagnostic", stderr.getvalue())

    def test_unload_releases_pipeline_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            generator = self.generator_class(root / "partcrafter" / "object", root / "outputs")
            generator._model = object()
            generator._rmbg_model = object()
            generator.unload()
            self.assertIsNone(generator._model)
            self.assertIsNone(generator._rmbg_model)


if __name__ == "__main__":
    unittest.main()
