# PromptDA Version Reference

## Upstream

- **Repository**: https://github.com/DepthAnything/PromptDA
- **Paper**: Prompting Depth Anything for 4K Resolution Accurate Metric Depth Estimation (Lin et al., 2026, arXiv:2412.14015)

## Local Copy Used

The thesis experiments used a local copy of PromptDA at:
`PromptDA-main/PromptDA-main/`

The exact upstream commit could not be verified from the local copy
(no `.git` metadata present; likely downloaded as a ZIP archive).

Manual inspection of the core model source files found no obvious
thesis-specific modifications:
- `promptda/promptda.py` — model class, forward pass, normalize/denormalize
- `promptda/model/dpt.py` — DPT head implementation
- `promptda/model/blocks.py` — building blocks
- `promptda/model/config.py` — encoder configurations

All adaptation for DSM refinement is in the training and evaluation
wrappers provided in this repository (training/, evaluation/).

## Pretrained Checkpoint

- **File**: `model.ckpt`
- **Usage**: Loaded via `PromptDA(encoder="vitl", ckpt_path="...")`
- **Backbone**: DINOv2 ViT-Large (loaded via `torch.hub.load` from local torchhub)

## Note

Exact equality with the upstream PromptDA version has not been verified.
Future work should pin a specific commit or release tag.
