# Environment Setup

## Conda

```bash
conda env create -f environment.yml
conda activate dsm-refine
```

## PyTorch with CUDA

Verify GPU is available:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If CUDA is not available, the scripts fall back to CPU automatically
(pass `--device cpu` to force CPU mode).

## VS Code

In VS Code, select the conda environment:
`Ctrl+Shift+P` → `Python: Select Interpreter` → choose `dsm-refine`

Do not commit `.vscode/settings.json` with absolute interpreter paths.

## Rasterio / GDAL

On Windows, rasterio and GDAL are easiest to install via conda:
```bash
conda install -c conda-forge rasterio gdal
```

On Linux/macOS, you may need system GDAL libraries first:
```bash
# Ubuntu
sudo apt-get install libgdal-dev
# macOS
brew install gdal
```

## PromptDA

PromptDA must be installed separately. Clone the repository and place
the pretrained checkpoint as described in `third_party/README.md`.
The path is passed via `--promptda-path` to all scripts.

## Tested Environment

- **OS**: Windows 11
- **Python**: 3.10
- **PyTorch**: 2.0.1+cu118
- **CUDA**: 11.8
- **rasterio**: 1.3.x
- **geopandas**: 0.14.x

Other OS/PyTorch/CUDA combinations should work but have not been tested.
