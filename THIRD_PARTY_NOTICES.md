# Third-party notices

This repository is the Modly integration maintained by DrHepa. PartCrafter itself is third-party work and is not authored by DrHepa.

## PartCrafter source code

- Project: [wgsxm/PartCrafter](https://github.com/wgsxm/PartCrafter)
- Setup revision: `3d773bf02fad51c7ab31a5615573fec93b287b30`
- Source archive SHA-256: `d7d1cf92c8d642af134f225ab447ff63b3b4784f1516d0c133c41e7cd0e2ccb6`
- Copyright: 2025 Yuchen Lin
- Root project license declaration: MIT; that text is reproduced in [`LICENSES/PARTCRAFTER-MIT.txt`](LICENSES/PARTCRAFTER-MIT.txt). This root declaration does not override the file-level terms below.

`setup.py` retrieves the source archive for that immutable revision, verifies its SHA-256 digest, and installs only the runtime source required by the extension. The extension contains compatibility fixes around upstream inference callbacks, latent handling, decoder selection, CUDA device placement, mesh-decode errors, and the optional `torch-cluster` path; those changes do not alter the upstream license.

### Tencent Hunyuan-derived transformer code

The pinned PartCrafter file [`src/models/transformers/partcrafter_transformer.py`](https://github.com/wgsxm/PartCrafter/blob/3d773bf02fad51c7ab31a5615573fec93b287b30/src/models/transformers/partcrafter_transformer.py) states that portions are copied or adapted from Tencent HunyuanDiT and reproduces the applicable Tencent Hunyuan Community License Agreement and Acceptable Use Policy.

That embedded agreement states that it does not apply in the European Union, defines its licensed territory as excluding the European Union, and says use outside that territory is unlicensed and unauthorized. It also imposes use restrictions, downstream notice obligations, and an exact `NOTICE` requirement for distribution. The required notice is included at repository root in [`NOTICE`](NOTICE). The full agreement remains embedded in the pinned source file installed by setup.

Neither PartCrafter's root MIT declaration nor this wrapper's MIT license replaces those terms. This project does not grant additional Tencent rights, determine whether a particular use is authorized, or cure the stated territorial gap. Users must review the embedded agreement and obtain any separate permission or qualified legal advice they need before installation or use.

### BSD-3-Clause smoothing utility

PartCrafter's `src/utils/smoothing.py` is Copyright (c) 2012–2015 P. M. Neila and carries the BSD 3-Clause License. Its license text is reproduced in [`LICENSES/SMOOTHING-BSD-3-CLAUSE.txt`](LICENSES/SMOOTHING-BSD-3-CLAUSE.txt) and remains in the source installed by setup.

## PartCrafter model weights

The weights remain hosted by their publisher and are downloaded only when the user requests them in Modly's Extensions UI.

| Node | Hugging Face repository | Revision audited for this release | Declared license |
| --- | --- | --- | --- |
| Object | [wgsxm/PartCrafter](https://huggingface.co/wgsxm/PartCrafter) | `69a0ffc1dad5e48e7e5ed91c0609f2b1276eb31f` | MIT |
| Scene | [wgsxm/PartCrafter-Scene](https://huggingface.co/wgsxm/PartCrafter-Scene) | `0454bb8e595a2765e8cb1f17ffacad9ba159777a` | MIT |

Modly 0.4.2's manifest contract does not expose a Hugging Face revision field, so the UI downloads the repository snapshot available at download time. The revisions above are audit evidence, not a claim that the Modly downloader pins them. The model-card declarations do not supersede file-level or derivative-work terms that may also apply through the required inference source.

## BRIA RMBG-1.4

- Model: [briaai/RMBG-1.4](https://huggingface.co/briaai/RMBG-1.4)
- Public revision audited for this release: `2ceba5a5efaec153162aedea169f76caf9b46cf8`
- Audited `model.safetensors` size: 176,381,984 bytes
- Audited `model.safetensors` ETag/SHA-256: `46ef7fe46f2ae284d8f1aaa24bfa5fca5ef25a34e2c7caa890a0029eb100e87f`
- License: BRIA's separate custom non-commercial model license, linked from the model repository; it is not MIT.

Modly downloads only `config.json` and `model.safetensors` from the publisher when the user requests the RMBG node. Neither file is redistributed in this repository. The RMBG architecture used by the pinned PartCrafter source retains its upstream BRIA source and copyright notice. The wrapper's MIT license does not cover or relicense RMBG source, weights, or use rights; users must review and comply with BRIA's terms.

## Optional Gemini integration

Part-count suggestion and style transfer use Google's Gemini API through the optional upstream provider. No API key or Gemini response is bundled. Users supply `GEMINI_API_KEY` or `GOOGLE_API_KEY` and are responsible for the applicable Google API terms, access, billing, and data handling.

## Optional rendering stack

The exposed upstream `--render` path calls PartCrafter's unmodified public functions in `src.utils.render_utils`. Its Python OpenGL stack is installed from PyPI with exact versions, including [pyrender](https://github.com/mmatl/pyrender), [PyOpenGL](https://pyopengl.sourceforge.net/), pyglet, freetype-py, imageio, NetworkX, and six. Those packages remain under their respective upstream licenses. Linux EGL/OpenGL shared libraries are supplied by the host operating system and are not redistributed here.
