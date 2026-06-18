# Checkpoints

Trained model checkpoints are not stored in this repository (~2 GB each).
Download from the links below and place in `training_outputs/`.

## Pretrained PromptDA

- **File**: `model.ckpt` (~1 GB)
- **Source**: DepthAnything/PromptDA HuggingFace or GitHub releases
- **Place in**: `PromptDA/checkpoints/model.ckpt`

## Trained Checkpoints

| Experiment | File | Description |
|---|---|---|
| HK-only | `training_outputs/hk_only/best_model.pth` | Trained on 139 HK patches |
| HK+Austin | `training_outputs/mixed/hk_austin/best_model.pth` | Trained on 139 HK + 311 Austin |
| HK+Paris | `training_outputs/mixed/hk_paris/best_model.pth` | Trained on 139 HK + 136 Paris |
| Austin+Paris | `training_outputs/mixed/austin_paris/best_model.pth` | Trained on 311 Austin + 136 Paris |

*Download links to be added after uploading to Zenodo/Google Drive.*
