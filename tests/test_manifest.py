from __future__ import annotations

import unittest
from pathlib import PurePosixPath
from urllib.parse import urlparse

from tests._support import read_json


class ManifestContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = read_json("manifest.json")
        cls.nodes = {node["id"]: node for node in cls.manifest["nodes"]}

    def test_root_identity_and_entrypoint(self) -> None:
        manifest = self.manifest
        self.assertEqual(manifest["id"], "partcrafter")
        self.assertEqual(manifest["type"], "model")
        self.assertEqual(manifest["author"], "DrHepa")
        self.assertEqual(manifest["generator_class"], "PartCrafterGenerator")
        self.assertEqual(manifest["vram_gb"], 8)
        self.assertRegex(manifest["version"], r"^\d+\.\d+\.\d+$")
        source = urlparse(manifest["source"])
        self.assertEqual(source.scheme, "https")
        self.assertEqual(source.netloc, "github.com")
        self.assertIn("DrHepa", source.path)

    def test_nodes_are_separate_weight_owners(self) -> None:
        self.assertEqual(set(self.nodes), {"object", "scene", "rmbg"})
        expected = {
            "object": "wgsxm/PartCrafter",
            "scene": "wgsxm/PartCrafter-Scene",
        }
        for node_id, hf_repo in expected.items():
            with self.subTest(node=node_id):
                node = self.nodes[node_id]
                self.assertEqual(node["hf_repo"], hf_repo)
                self.assertEqual(node["input"], "image")
                self.assertEqual(node["output"], "mesh")
                self.assertEqual(
                    node["download_check"],
                    "vae/diffusion_pytorch_model.safetensors",
                )
                sentinel = PurePosixPath(node["download_check"])
                self.assertFalse(sentinel.is_absolute())
                self.assertNotIn("..", sentinel.parts)
                self.assertNotIn("hf_skip_prefixes", node)
                self.assertNotIn("hf_include_prefixes", node)

        rmbg = self.nodes["rmbg"]
        self.assertEqual(rmbg["hf_repo"], "briaai/RMBG-1.4")
        self.assertEqual(rmbg["input"], "image")
        self.assertEqual(rmbg["output"], "image")
        self.assertEqual(rmbg["download_check"], "model.safetensors")
        self.assertEqual(
            rmbg["hf_include_prefixes"],
            ["config.json", "model.safetensors"],
        )
        self.assertEqual(rmbg["params_schema"], [])
        self.assertNotIn("hf_skip_prefixes", rmbg)

    def test_parameter_schemas_use_supported_types_and_inline_defaults(self) -> None:
        allowed_types = {"select", "int", "float", "string", "file-select"}
        for node_id, node in self.nodes.items():
            schema = node.get("params_schema")
            self.assertIsInstance(schema, list)
            if node_id == "rmbg":
                self.assertEqual(schema, [])
                continue
            self.assertTrue(schema)
            seen: set[str] = set()
            for param in schema:
                with self.subTest(node=node_id, parameter=param.get("id")):
                    self.assertIsInstance(param.get("id"), str)
                    self.assertIsInstance(param.get("label"), str)
                    self.assertTrue(param["label"].strip())
                    self.assertNotIn(param["id"], seen)
                    seen.add(param["id"])
                    self.assertIn(param.get("type"), allowed_types)
                    self.assertIn("default", param)
                    if param["type"] == "select":
                        self.assertTrue(param.get("options"))
                        values = [option["value"] for option in param["options"]]
                        self.assertTrue(
                            all(
                                "value" in option
                                and isinstance(option.get("label"), str)
                                and option["label"].strip()
                                for option in param["options"]
                            )
                        )
                        self.assertIn(param["default"], values)

            by_id = {param["id"]: param for param in schema}
            for param in schema:
                condition = param.get("show_if")
                if not condition:
                    continue
                self.assertIsInstance(condition, dict)
                self.assertEqual(len(condition), 1)
                controller_id, expected = next(iter(condition.items()))
                self.assertIn(controller_id, by_id)
                controller = by_id[controller_id]
                if controller["type"] == "select":
                    self.assertIn(
                        expected,
                        [option["value"] for option in controller["options"]],
                    )

    def test_defaults_match_upstream_modes(self) -> None:
        def defaults(node_id: str) -> dict:
            return {
                param["id"]: param["default"]
                for param in self.nodes[node_id]["params_schema"]
            }

        object_defaults = defaults("object")
        scene_defaults = defaults("scene")
        for values in (object_defaults, scene_defaults):
            self.assertEqual(values["seed"], 0)
            self.assertEqual(values["num_inference_steps"], 50)
            self.assertEqual(values["guidance_scale"], 7.0)
            self.assertEqual(values["max_num_expanded_coords"], 1_000_000_000)
            self.assertEqual(values["use_flash_decoder"], "false")
            self.assertEqual(values["render"], "false")
        self.assertEqual(object_defaults["num_tokens"], 1024)
        self.assertEqual(scene_defaults["scene_num_tokens"], 2048)
        self.assertLessEqual(object_defaults["num_parts"], 16)
        self.assertLessEqual(scene_defaults["scene_num_parts"], 8)
        self.assertEqual(object_defaults["remove_background"], "false")
        self.assertNotIn("remove_background", scene_defaults)

    def test_each_node_exposes_the_complete_supported_inference_surface(self) -> None:
        expected = {
            "part_count_mode",
            "part_model",
            "style_transfer",
            "style_model",
            "num_inference_steps",
            "guidance_scale",
            "max_num_expanded_coords",
            "use_flash_decoder",
            "seed",
            "render",
            "output_name",
        }
        for node_id in ("object", "scene"):
            node = self.nodes[node_id]
            schema = {param["id"]: param for param in node["params_schema"]}
            with self.subTest(node=node_id):
                node_expected = set(expected)
                if node_id == "object":
                    node_expected.update(
                        {"num_parts", "num_tokens", "remove_background"}
                    )
                    count_id = "num_parts"
                    tokens_id = "num_tokens"
                else:
                    node_expected.update({"scene_num_parts", "scene_num_tokens"})
                    count_id = "scene_num_parts"
                    tokens_id = "scene_num_tokens"
                self.assertEqual(set(schema), node_expected)
                self.assertEqual(schema[count_id]["min"], 1)
                self.assertEqual(
                    schema[count_id]["max"], 16 if node_id == "object" else 8
                )
                self.assertEqual(
                    schema[count_id]["show_if"], {"part_count_mode": "manual"}
                )
                self.assertEqual(
                    schema["part_model"]["show_if"], {"part_count_mode": "gemini"}
                )
                self.assertEqual(
                    schema["style_model"]["show_if"], {"style_transfer": "true"}
                )
                self.assertEqual(schema[tokens_id]["min"], 1)
                self.assertEqual(schema["num_inference_steps"]["min"], 1)
                self.assertNotIn("min", schema["guidance_scale"])
                self.assertNotIn("min", schema["seed"])
                self.assertNotIn("max", schema["seed"])
                self.assertEqual(schema["max_num_expanded_coords"]["min"], 0)

    def test_no_nonfunctional_upstream_controls_are_advertised(self) -> None:
        ids = {
            param["id"]
            for node in self.nodes.values()
            for param in node["params_schema"]
        }
        self.assertNotIn("flash_octree_depth", ids)
        self.assertIn("remove_background", ids)
        self.assertIn("render", ids)

if __name__ == "__main__":
    unittest.main()
