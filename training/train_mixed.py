"""
Mixed-region model training.
Trains on two cities simultaneously using a combined training CSV.
The fixed evaluation groups are excluded.

Usage:
    python training/train_mixed.py --config configs/hk_austin.yaml
"""
import argparse
import csv
import copy
import json
import os
import sys
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets import CSVPromptDataset


def parse_args():
    p = argparse.ArgumentParser(description="Mixed-region training")
    p.add_argument("--config", required=True, help="YAML config file")
    p.add_argument("--promptda-path", required=True,
                   help="Path to PromptDA package")
    p.add_argument("--data-root", default="data")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    return p.parse_args()


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
    return torch.abs(pred - gt)[valid_mask].mean(), valid_mask


def _batch_str(batch, key):
    val = batch[key]
    return str(val[0]) if isinstance(val, (list, tuple)) else str(val)


def run_epoch(model, loader, optimizer, device, trainable_params, training):
    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    valid_batches = 0
    valid_ratio_sum = 0.0
    city_loss = {}

    for batch in loader:
        rgb = batch["rgb"].to(device)
        prompt_depth = batch["prompt_depth"].to(device)
        gt_dsm = batch["gt_dsm"].to(device)
        city = _batch_str(batch, "city")

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            pred = model(rgb, prompt_depth)
            loss, valid_mask = masked_l1_loss(pred, gt_dsm)

        if loss is None:
            continue

        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

        lv = loss.item()
        total_loss += lv
        valid_batches += 1
        valid_ratio_sum += valid_mask.float().mean().item()
        city_loss[city] = city_loss.get(city, 0.0) + lv

    if valid_batches == 0:
        return 0.0, 0.0, {}
    avg = total_loss / valid_batches
    city_avg = {c: s / valid_batches for c, s in city_loss.items()}
    return avg, valid_ratio_sum / valid_batches, city_avg


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))

    data_root = Path(args.data_root)
    train_csv = data_root / cfg["data"]["train_csv"]
    output_dir = Path(args.output_dir) if args.output_dir else Path(cfg["output"]["save_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)
    print(f"Device: {device}")

    # PromptDA
    promptda_path = Path(args.promptda_path)
    sys.path.insert(0, str(promptda_path))
    from promptda.promptda import PromptDA

    # Dataset
    full_dataset = CSVPromptDataset(train_csv, data_root=data_root)
    vs = max(1, int(round(len(full_dataset) * cfg["training"]["val_ratio_from_train"])))
    train_ds, val_ds = random_split(
        full_dataset, [len(full_dataset) - vs, vs],
        generator=torch.Generator().manual_seed(cfg["random_seed"]),
    )
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    # Model
    ckpt_path = cfg.get("model", {}).get("pretrained_checkpoint",
                                          str(promptda_path / "checkpoints" / "model.ckpt"))
    model = PromptDA(encoder=cfg["model"]["encoder"], ckpt_path=ckpt_path).to(device)
    prefix = cfg["model"]["freeze_strategy"]["trainable_prefix"]
    for n, p in model.named_parameters():
        p.requires_grad = n.startswith(prefix)
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    # Optimizer / Scheduler
    oc = cfg["optimizer"]
    optimizer = torch.optim.AdamW(trainable_params, lr=oc["lr"], weight_decay=oc["weight_decay"])
    sc = cfg["scheduler"]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode=sc["mode"], factor=sc["factor"], patience=sc["patience"])

    # Logging
    with open(output_dir / "train_log.csv", "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "lr", "val_valid_ratio", "city_loss"])

    # Config snapshot
    run_cfg = copy.deepcopy(cfg)
    run_cfg["_train_csv"] = str(train_csv)
    run_cfg["_data_root"] = str(data_root)
    run_cfg["_device"] = device
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(run_cfg, f)
    full_dataset.df.to_csv(output_dir / "used_training_split.csv", index=False)

    # Training loop
    tc = cfg["training"]
    best_val, best_ep, no_imp = float("inf"), -1, 0
    t0 = time.time()

    for ep in range(tc["num_epochs"]):
        print(f"\n=== Epoch {ep+1}/{tc['num_epochs']} ===")
        tl, _, tcl = run_epoch(model, train_loader, optimizer, device, trainable_params, True)
        vl, vvr, vcl = run_epoch(model, val_loader, None, device, trainable_params, False)
        scheduler.step(vl)
        lr = optimizer.param_groups[0]["lr"]
        print(f"Train={tl:.4f} Val={vl:.4f} VR={vvr:.4f} LR={lr:.2e}")
        if tcl:
            print(f"  Train city: {tcl}")
        if vcl:
            print(f"  Val city: {vcl}")

        with open(output_dir / "train_log.csv", "a", newline="") as f:
            csv.writer(f).writerow([ep+1, tl, vl, lr, vvr, str(vcl)])

        ckpt = {
            "epoch": ep+1, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": tl, "val_loss": vl, "lr": lr,
            "best_val_loss": best_val, "best_epoch": best_ep,
            "train_csv": str(train_csv), "config": run_cfg,
        }
        torch.save(ckpt, output_dir / "last_model.pth")

        if vl < best_val:
            best_val, best_ep, no_imp = vl, ep+1, 0
            torch.save(ckpt, output_dir / "best_model.pth")
            print(f"  -> best (val={vl:.4f})")
        else:
            no_imp += 1
            if no_imp >= tc["early_stopping_patience"]:
                print(f"Early stopping at epoch {ep+1}")
                break

    meta = {
        "experiment": cfg.get("experiment"), "train_csv": str(train_csv),
        "train_patches": len(full_dataset), "actual_train": len(train_ds),
        "actual_val": len(val_ds), "best_val_loss": best_val, "best_epoch": best_ep,
        "elapsed_seconds": time.time() - t0, "device": device,
    }
    with open(output_dir / "run_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone. Best val={best_val:.4f} epoch={best_ep}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
