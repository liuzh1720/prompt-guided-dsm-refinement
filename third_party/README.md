# PromptDA External Dependency

## Source

- **Repository**: [DepthAnything/PromptDA](https://github.com/DepthAnything/PromptDA)
- **Paper**: Lin et al., "Prompting Depth Anything for 4K Resolution Accurate Metric Depth Estimation" (arXiv:2412.14015, 2026)
- **Local copy used in thesis**: `PromptDA-main/PromptDA-main/`

## Setup

1. Clone PromptDA into a directory outside this repository:
   ```bash
   git clone https://github.com/DepthAnything/PromptDA.git
   ```

2. Download the pretrained checkpoint (model.ckpt) and place it in:
   ```
   PromptDA/checkpoints/model.ckpt
   ```

3. When running training or evaluation, pass the PromptDA path:
   ```bash
   python training/train_hk_only.py --promptda-path /path/to/PromptDA ...
   ```

## Expected directory structure

```
PromptDA/
├── checkpoints/
│   └── model.ckpt
├── promptda/
│   ├── promptda.py
│   ├── model/
│   │   ├── dpt.py
│   │   ├── blocks.py
│   │   └── config.py
│   └── ...
├── torchhub/
│   └── facebookresearch_dinov2_main/
└── ...
```

## Version

The exact upstream commit used for the thesis experiments could not be verified
(no git metadata in the local copy).  Manual inspection of the core model files
(promptda.py, dpt.py, blocks.py, config.py) found no obvious thesis-specific
modifications.  All adaptation is done in the training/evaluation wrappers in
this repository.

## License

The PromptDA license has not been confirmed from the local copy.  Check the
upstream repository for license terms.  PromptDA source code is NOT
redistributed in this repository.

## Integration

This repository's training and evaluation scripts use PromptDA as follows:

```python
import sys
sys.path.insert(0, "/path/to/PromptDA")
from promptda.promptda import PromptDA

model = PromptDA(encoder="vitl", ckpt_path="path/to/model.ckpt")
```

The model is instantiated with `encoder="vitl"` (ViT-Large DINOv2 backbone).
Only the `depth_head` parameters are trained; the encoder is frozen.
