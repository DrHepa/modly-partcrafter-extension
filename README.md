# PartCrafter for Modly

A Modly `model` extension for [PartCrafter](https://github.com/wgsxm/PartCrafter). It turns one image into a structured GLB whose generated parts remain separate, and exposes the upstream image preprocessing route:

- **PartCrafter Object** — 1–16 parts with [`wgsxm/PartCrafter`](https://huggingface.co/wgsxm/PartCrafter).
- **PartCrafter Scene** — 1–8 scene objects with [`wgsxm/PartCrafter-Scene`](https://huggingface.co/wgsxm/PartCrafter-Scene).
- **PartCrafter RMBG Preprocess** — image-to-image background removal and upstream white-background framing with [`briaai/RMBG-1.4`](https://huggingface.co/briaai/RMBG-1.4).

The Modly integration is maintained by **DrHepa**. PartCrafter and its weights remain work by their upstream authors.

> **Third-party license warning:** the pinned PartCrafter source includes copied or adapted Tencent Hunyuan code whose embedded Community License states that it does not apply in the European Union. Review [Limitations](#limitations), [NOTICE](NOTICE), and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) before installation. This wrapper cannot grant rights that the upstream licenses withhold.

## Installation

PartCrafter requires 64-bit CPython 3.11 or 3.12 and a supported NVIDIA CUDA system.

1. Open **Extensions** in Modly.
2. Choose **Install from GitHub** and enter:

   `https://github.com/DrHepa/modly-partcrafter-extension`

3. Wait for setup to finish. Setup creates the extension-local `venv`, installs the pinned runtime, retrieves the immutable upstream source revision, applies audited inference fixes, and validates the environment. **It does not download model weights.**

When the repository is copied directly into Modly's extensions runtime, its directory must be named `partcrafter` so it matches `manifest.id`. Opening its drawer and choosing **Repair** then runs the same setup path. Modly invokes `setup.py` with its JSON environment payload; no separate global Python environment should be activated.

## Download weights in Modly

In the PartCrafter extension card, download the nodes you need. Modly owns the download and places each snapshot under the configured **Models Directory**:

```text
<Models Directory>/partcrafter/object
<Models Directory>/partcrafter/scene
<Models Directory>/partcrafter/rmbg
```

For example, with the requested Linux location:

```text
/home/drhepa/Documentos/Modly/models/partcrafter/object
/home/drhepa/Documentos/Modly/models/partcrafter/scene
/home/drhepa/Documentos/Modly/models/partcrafter/rmbg
```

Object and Scene are about 3.701 GiB each. RMBG downloads only its configuration and 176,381,984-byte checkpoint. If you change **Models Directory**, restart Modly before starting the download. Generation loads only exact UI-managed directories in local-only mode; it never falls back to a cache or downloads missing files.

## Usage

Create a workflow with an image-producing/input node, connect it to **PartCrafter Object** or **PartCrafter Scene**, and run it. Use **PartCrafter RMBG Preprocess** directly before Object when you want the image-to-image preprocessing node.

## Outputs

The primary Object/Scene result returned to Modly is a composite GLB scene. Each run also writes beside it:

- `part_00.glb`, `part_01.glb`, … — one mesh per generated part/object;
- `generation.json` — resolved parameters, output inventory, timings, and any sanitized style-transfer warning;
- `styled_input.png` — only when optional Gemini style transfer succeeds.

When **Turntable Rendering** is enabled, the same run also contains upstream's `rendering.gif`, `rendering_normal.gif`, `rendering_grid.gif` and the first-frame `rendering.png`, `rendering_normal.png`, `rendering_grid.png`. The GLB remains Modly's primary result.

The composite keeps parts as separately named scene geometries instead of flattening them into one mesh. A failed decoder part is reported as an error; the extension never inserts a synthetic degenerate mesh.

**PartCrafter RMBG Preprocess** is a workflow-only image-to-image node. It preserves a useful RGBA input alpha channel, calls upstream `prepare_image` with a white background and padding ratio `0.1`, and returns an atomically published PNG. Connect that PNG to Object, or enable Object's **Remove Background** control after downloading the RMBG node. The combined Object route follows upstream order exactly: style transfer → count suggestion → RMBG → 3D pipeline.

## Parameters

| Control | Upstream behavior |
| --- | --- |
| Part/Object Count Mode | Manual count, or Gemini suggestion using `gemini-3-flash-preview` by default. |
| Number of Parts/Objects | Object: 1–16, extension default 3. Scene: 1–8, extension default 6. Upstream requires a count or VLM suggestion rather than defining a CLI default. |
| Objaverse Style Transfer | Optional Gemini preprocessing with `gemini-3.1-flash-image-preview` by default. |
| Remove Background | Object only. Runs the separate BRIA RMBG-1.4 model after style/count preprocessing; default disabled. |
| Tokens per Part/Object | Object default 1024; scene default 2048. Reducing this can lower VRAM use. |
| Inference Steps | Denoising steps; upstream default 50. |
| Guidance Scale | Image-conditioning strength; upstream default 7.0. |
| Maximum Expanded Coordinates | Mesh-extraction safety limit; upstream CLI default 1,000,000,000. `0` disables the limit. |
| Flash Decoder | Selects PartCrafter's flash decoder; upstream CLI default is disabled. |
| Seed | Reproducible pipeline seed; upstream default 0. |
| Turntable Rendering | Optional upstream 36-view color, normal, and comparison rendering at radius 4 and 18 fps; default disabled. |
| Run Name | Optional sanitized run-directory and GLB stem, analogous to upstream `--tag`; upstream itself always names the composite `object.glb`. |

The workflow node also exposes the upstream Gemini model-name overrides as conditional text fields. Modly 0.4.2's classic Generate controls neither display string fields nor apply `show_if`; use a workflow for Gemini mode and its model-name overrides.

## Optional Gemini features

Set either `GEMINI_API_KEY` or `GOOGLE_API_KEY` in the environment that launches Modly, then restart Modly. Gemini is contacted only when count suggestion or style transfer is selected.

- Count suggestion must return a valid count; API/authentication failures stop the run with an actionable error.
- Style transfer follows upstream behavior: a failure is logged as a warning and generation continues with the original image. The sanitized reason is recorded in `generation.json`.

API access, billing, availability, and data handling are governed by Google's service terms.

## Compatibility

Target host: **upstream Modly 0.4.2**. Its standard lane uses CPython 3.11; the setup and contract-test matrix also support CPython 3.12 for compatible hosts and forks such as Modly Private. Upstream PartCrafter recommends at least **8 GB VRAM**. More parts, more tokens, and scene generation can require more.

| Platform | Status in this release |
| --- | --- |
| Linux x86_64 + NVIDIA CUDA | Upstream route implemented. |
| Windows x64 + NVIDIA CUDA | Upstream community-supported route implemented. |
| Linux ARM64 NVIDIA SBSA (SM90+) + CUDA 12.8 | Experimental implementation using the official PyTorch ARM64 lane and a PyTorch FPS fallback. Not a generic Jetson claim. |
| CPU, Apple MPS/macOS, ROCm, Windows ARM64 | Unsupported. Setup fails with an explicit diagnostic instead of installing a nonfunctional environment. |

Full-checkpoint generation was not run in this build workspace. Repository validation uses protocol, offline, manifest, setup, and controlled pipeline-double tests; the real GPU/weights test is intentionally left for the target Modly installation.

Turntable rendering uses EGL on Linux and the normal OpenGL backend on Windows. Linux hosts must provide the system OpenGL libraries; on Debian/Ubuntu install `libegl1` and `libgl1` before setup. Python rendering packages are pinned and installed inside the extension venv.

## Limitations

All upstream object/scene inference capabilities are exposed: manual or Gemini counts, Gemini style transfer, optional BRIA RMBG preprocessing, seeding, token/step/guidance controls, extraction limits, flash-decoder selection, separate part/composite exports, and turntable rendering. RMBG is a separate model node because Modly assigns one Hugging Face repository and weight directory to each node.

Training and dataset preprocessing are also outside a Modly inference extension.

The pinned source file `src/models/transformers/partcrafter_transformer.py` says it contains code copied or adapted from Tencent HunyuanDiT and embeds the Tencent Hunyuan Community License. That embedded version defines a territory excluding the European Union and says use outside that territory is unlicensed. It also imposes an Acceptable Use Policy and distribution/notice conditions. Do not install or use the affected code where those terms do not authorize it; obtain separate permission or qualified legal advice if needed. The source also contains a BSD-3-Clause smoothing utility. These file-level terms are not replaced by PartCrafter's root MIT file.

BRIA RMBG-1.4 is limited to non-commercial use unless the user obtains a commercial agreement from BRIA.

## Troubleshooting

Setup progress and failures are streamed to Modly through `stderr`. Complete evidence is kept at:

```text
.modly/setup/logs/setup.log
.modly/setup/setup-status.json
```

Generation exceptions are returned through Modly's model-runner protocol; upstream diagnostic output is redirected to extension `stderr` so it remains visible without corrupting the protocol stream.

- **Incomplete weights:** use **Remove model weights** for the affected Object, Scene, or RMBG node, then download it again. The generator validates all required local files even when Modly's single sentinel already exists.
- **Object RMBG error:** download **PartCrafter RMBG Preprocess** before enabling **Remove Background**; its weights must be in the sibling `partcrafter/rmbg` directory.
- **Interrupted or broken setup:** open the extension drawer and choose **Repair**.
- **CUDA/profile rejection:** use a supported platform above and inspect the setup log; setup does not silently switch to CPU.
- **Gemini error:** verify that the key is exported to the process that launches Modly and that the selected model is available to that key.
- **Turntable rendering error:** on Linux verify EGL/OpenGL system libraries; on Windows update the NVIDIA/OpenGL driver, then restart Modly and run **Repair**. A requested render failure stops the run instead of publishing incomplete sidecars.

## Upstream and credits

- This extension targets [Modly](https://github.com/lightningpixel/modly), created by **Lightning Pixel**, and is authored and maintained by **DrHepa**.
- Upstream source is prepared from commit `3d773bf02fad51c7ab31a5615573fec93b287b30`.
- Its transformer contains Tencent Hunyuan-derived portions under the embedded Tencent Hunyuan Community License; `src/utils/smoothing.py` retains BSD-3-Clause terms.
- Object weights were audited at revision `69a0ffc1dad5e48e7e5ed91c0609f2b1276eb31f`.
- Scene weights were audited at revision `0454bb8e595a2765e8cb1f17ffacad9ba159777a`.
- RMBG-1.4 was audited at public revision `2ceba5a5efaec153162aedea169f76caf9b46cf8`; its `model.safetensors` SHA-256 is `46ef7fe46f2ae284d8f1aaa24bfa5fca5ef25a34e2c7caa890a0029eb100e87f`.

Paper: [PartCrafter: Structured 3D Mesh Generation via Compositional Latent Diffusion Transformers](https://arxiv.org/abs/2506.05573).

## License

- This wrapper is MIT licensed, Copyright 2026 DrHepa.
- PartCrafter's repository root and Object/Scene model cards declare MIT, but specific upstream source files retain Tencent Hunyuan Community and BSD-3-Clause terms. The wrapper MIT license does not override them.
- RMBG-1.4 has BRIA's separate custom non-commercial license and is not covered by this wrapper's MIT license. The extension does not redistribute RMBG weights or relabel its license. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
