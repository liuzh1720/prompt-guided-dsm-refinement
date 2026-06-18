"""
CSV-based dataset for paired RGB-prompt-DSM patches.

Reads sample paths from a CSV file with columns:
    city, patch_id, rgb_path, dsm_path, prompt_path

All paths are resolved relative to data_root.
"""
from pathlib import Path
from typing import Optional, Dict, Union

import numpy as np
import pandas as pd
import rasterio
import torch
from torch.utils.data import Dataset


class CSVPromptDataset(Dataset):
    """
    Dataset that reads paired (RGB, prompt DSM, reference DSM) samples
    from a CSV split file.

    The CSV must contain columns:
        city, patch_id, rgb_path, dsm_path, prompt_path

    Paths in the CSV are resolved relative to data_root if they are
    not absolute.

    RGB images are read as 3-channel float32 and normalized per-patch
    by max value when max > 1.  Prompt and reference DSM are read as
    single-channel float32 with shape [1, H, W].

    Parameters
    ----------
    csv_path : str or Path
        Path to the CSV split file.
    data_root : str or Path, optional
        Root directory for resolving relative paths in the CSV.
        Ignored for absolute paths.
    """

    def __init__(
        self,
        csv_path: Union[str, Path],
        data_root: Optional[Union[str, Path]] = None,
    ):
        self.csv_path = Path(csv_path)
        self.data_root = Path(data_root) if data_root else None

        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")

        self.df = pd.read_csv(self.csv_path)

        required_cols = ["city", "patch_id", "rgb_path", "dsm_path", "prompt_path"]
        missing = [c for c in required_cols if c not in self.df.columns]
        if missing:
            raise ValueError(
                f"CSV missing columns: {missing}\n"
                f"CSV: {self.csv_path}\n"
                f"Found: {list(self.df.columns)}"
            )

        for col in required_cols:
            self.df[col] = self.df[col].astype(str)

        self._validate_files()
        self._log_summary()

    def _resolve(self, rel_path: str) -> Path:
        """Resolve a path from the CSV, using data_root for relative paths."""
        p = Path(rel_path)
        if p.is_absolute():
            return p
        if self.data_root is None:
            return p
        return self.data_root / p

    def _validate_files(self):
        missing = []
        for idx, row in self.df.iterrows():
            for key in ["rgb_path", "dsm_path", "prompt_path"]:
                p = self._resolve(row[key])
                if not p.exists():
                    missing.append((key, idx, p))
        if missing:
            print(f"\nWarning: {len(missing)} missing file(s) in {self.csv_path}")
            for key, idx, p in missing[:10]:
                print(f"  row {idx}, {key}: {p}")
            if len(missing) > 10:
                print(f"  ... and {len(missing) - 10} more")
            raise FileNotFoundError(f"{len(missing)} missing files in {self.csv_path}")

    def _log_summary(self):
        print(f"Loaded {len(self.df)} samples from {self.csv_path}")
        if "city" in self.df.columns:
            counts = self.df["city"].value_counts().to_dict()
            print(f"  City counts: {counts}")

    def __len__(self) -> int:
        return len(self.df)

    def _read_rgb(self, path: Path) -> torch.Tensor:
        """Read RGB bands [1,2,3], float32, normalize per-patch by max."""
        with rasterio.open(path) as src:
            rgb = src.read([1, 2, 3]).astype(np.float32)  # [3, H, W]
        rgb_max = rgb.max()
        if rgb_max > 1:
            rgb = rgb / rgb_max
        return torch.from_numpy(rgb).float()

    def _read_single_band(self, path: Path) -> torch.Tensor:
        """Read single-band raster, float32, add channel dim: [H,W] -> [1,H,W]."""
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.float32)
        arr = np.expand_dims(arr, axis=0)
        return torch.from_numpy(arr).float()

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        row = self.df.iloc[idx]

        rgb = self._read_rgb(self._resolve(row["rgb_path"]))
        prompt_depth = self._read_single_band(self._resolve(row["prompt_path"]))
        gt_dsm = self._read_single_band(self._resolve(row["dsm_path"]))

        return {
            "rgb": rgb,
            "prompt_depth": prompt_depth,
            "gt_dsm": gt_dsm,
            "city": row.get("city", ""),
            "patch_id": row.get("patch_id", ""),
            "rgb_path": row["rgb_path"],
            "prompt_path": row["prompt_path"],
            "dsm_path": row["dsm_path"],
        }
