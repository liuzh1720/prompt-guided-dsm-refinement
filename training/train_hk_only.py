"""
HK-only model training.
Uses a fixed 72/139 evaluation split: reads HK_train.csv (139 patches) for
training with an internal validation split.  HK_eval.csv (72 patches) is
excluded from training.

Usage:
    python training/train_hk_only.py --config configs/hk_only.yaml
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, random_split

# --- add datasets module to path ---
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets import CSVPromptDataset


def parse_args():
    p = argparse.ArgumentParser(description="HK-only training")
    p.add_argument("--config", required=True, help="YAML config file")
    p.add_argument("--promptda-path", required=True,
                   help="Path to PromptDA package (containing promptda/promptda.py)")
    p.add_argument("--data-root", default="data", help="Root data directory")
    p.add_argument("--output-dir", default=None, help="Override save directory")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    return p.parse_args()


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def get_device(pref):
    if pref == "cuda" and torch.cuda.is_available():
        return "cuda"
    if pref == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def masked_l1_loss(pred, gt, nodata_threshold=-9990):
    valid_mask = gt > nodata_threshold
    if valid_mask.sum() == 0:
        return None, valid_mask
    loss = torch.abs(pred - gt)[valid_mask].mean()
    return loss, valid_mask


def run_epoch(model, loader, optimizer, device, trainable_params, training):
    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    valid_batches = 0
    valid_ratio_sum = 0.0

    for batch_idx, batch in enumerate(loader):
        rgb = batch["rgb"].to(device)
        prompt_depth = batch["prompt_depth"].to(device)
        gt_dsm = batch["gt_dsm"].to(device)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            pred = model(rgb, prompt_depth)
            loss, valid_mask = masked_l1_loss(
                pred, gt_dsm, nodata_threshold=-9990
            )

        if loss is None:
            continue

        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        valid_batches += 1
        valid_ratio_sum += valid_mask.float().mean().item()

    if valid_batches == 0:
        return 0.0, 0.0
    return total_loss / valid_batches, valid_ratio_sum / valid_batches


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # --- paths ---
    data_root = Path(args.data_root) if args.data_root else Path(".")
    train_csv = Path(cfg["data"]["train_csv"])  # relative to project root, not data_root
    output_dir = Path(args.output_dir) if args.output_dir else Path(cfg["output"]["save_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- device ---
    device = get_device(args.device)
    print(f"Device: {device}")

    # --- PromptDA ---
    promptda_path = Path(args.promptda_path)
    if not promptda_path.exists():
        sys.exit(f"PromptDA path not found: {promptda_path}")
    sys.path.insert(0, str(promptda_path))
    try:
        from promptda.promptda import PromptDA
    except ImportError:
        sys.exit("Failed to import PromptDA. Check --promptda-path.")

    # --- dataset ---
    full_dataset = CSVPromptDataset(train_csv, data_root=data_root)
    val_size = max(1, int(round(len(full_dataset) * cfg["training"]["val_ratio_from_train"])))
    train_dataset, val_dataset = random_split(
        full_dataset,
        [len(full_dataset) - val_size, val_size],
        generator=torch.Generator().manual_seed(cfg["random_seed"]),
    )

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # --- model ---
    ckpt_path = cfg.get("model", {}).get("pretrained_checkpoint",
                                          str(promptda_path / "checkpoints" / "model.ckpt"))
    model = PromptDA(encoder=cfg["model"]["encoder"], ckpt_path=ckpt_path).to(device)

    for name, param in model.named_parameters():
        prefix = cfg["model"]["freeze_strategy"]["trainable_prefix"]
        param.requires_grad = name.startswith(prefix)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable params: {len(trainable_params)}")

    # --- optimizer & scheduler ---
    opt_cfg = cfg["optimizer"]
    optimizer = torch.optim.AdamW(
        trainable_params, lr=opt_cfg["lr"], weight_decay=opt_cfg["weight_decay"]
    )
    sch_cfg = cfg["scheduler"]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode=sch_cfg["mode"], factor=sch_cfg["factor"],
        patience=sch_cfg["patience"]
    )

    # --- logging ---
    log_csv = output_dir / "train_log.csv"
    with open(log_csv, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "lr", "val_valid_ratio"])

    # --- save config snapshot ---
    import copy
    run_cfg = copy.deepcopy(cfg)
    run_cfg["_train_csv"] = str(train_csv)
    run_cfg["_data_root"] = str(data_root)
    run_cfg["_promptda_path"] = str(promptda_path)
    run_cfg["_device"] = device
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(run_cfg, f)

    # --- save used split ---
    full_dataset.df.to_csv(output_dir / "used_training_split.csv", index=False)

    # --- training loop ---
    t_cfg = cfg["training"]
    best_val_loss = float("inf")
    best_epoch = -1
    no_improve = 0
    start_time = time.time()

    for epoch in range(t_cfg["num_epochs"]):
        print(f"\n=== Epoch {epoch+1}/{t_cfg['num_epochs']} ===")
        train_loss, _ = run_epoch(model, train_loader, optimizer, device,
                                  trainable_params, training=True)
        val_loss, val_vr = run_epoch(model, val_loader, None, device,
                                     trainable_params, training=False)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Train={train_loss:.4f} Val={val_loss:.4f} VR={val_vr:.4f} LR={current_lr:.2e}")
        with open(log_csv, "a", newline="") as f:
            csv.writer(f).writerow([epoch+1, train_loss, val_loss, current_lr, val_vr])

        ckpt = {
            "epoch": epoch+1, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss, "val_loss": val_loss, "lr": current_lr,
            "best_val_loss": best_val_loss, "best_epoch": best_epoch,
            "train_csv": str(train_csv), "config": run_cfg,
        }
        torch.save(ckpt, output_dir / "last_model.pth")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            no_improve = 0
            torch.save(ckpt, output_dir / "best_model.pth")
            print(f"  -> best model (val_loss={val_loss:.4f})")
        else:
            no_improve += 1
            if no_improve >= t_cfg["early_stopping_patience"]:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # --- run metadata ---
    elapsed = time.time() - start_time
    meta = {
        "experiment": cfg.get("experiment", "hk_only"),
        "train_csv": str(train_csv),
        "train_patches": len(full_dataset),
        "actual_train": len(train_dataset),
        "actual_val": len(val_dataset),
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "elapsed_seconds": elapsed,
        "device": device,
        "promptda_path": str(promptda_path),
    }
    with open(output_dir / "run_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone. Best val={best_val_loss:.4f} at epoch {best_epoch}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
