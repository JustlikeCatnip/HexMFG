from root import ROOT_DIR

import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)
from lightning.pytorch import loggers
from lightning.pytorch.tuner import Tuner
import argparse
import torch
from models import unet_precip_regression_lightning as unet_regr
import models.regression_HexMFG as gan


def train_regression(hparams):
    if hparams.model == "HexMF-UNet":
        net = unet_regr.HexMF_UNet(hparams=hparams)
    elif hparams.model == "HexMF-GNet":
        net = unet_regr.HexMF_GNet(hparams=hparams)
    elif hparams.model == "HexMFG":
        net = gan.GAN(hparams=hparams)
    else:
        raise Exception(f"{hparams.model} is not a valid model name")

    default_save_path = hparams.default_save_path

    checkpoint_callback = ModelCheckpoint(
        dirpath=default_save_path / net.__class__.__name__,
        filename=net.__class__.__name__ + "_rain_threshhold_50_{epoch}-{val_loss:.6f}",
        save_top_k=3,
        save_last=True,
        verbose=False,
        monitor="val_loss",
        mode="min",
    )
    lr_monitor = LearningRateMonitor()
    tb_logger = loggers.TensorBoardLogger(save_dir=default_save_path, name=net.__class__.__name__)

    earlystopping_callback = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=hparams.es_patience,
    )
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        fast_dev_run=hparams.fast_dev_run,
        max_epochs=hparams.epochs,
        default_root_dir=default_save_path,
        logger=tb_logger,
        callbacks=[checkpoint_callback, earlystopping_callback, lr_monitor],
        val_check_interval=hparams.val_check_interval,
    )
    trainer.fit(model=net, ckpt_path=hparams.resume_from_checkpoint)

def get_best_checkpoint(checkpoint_dir):
    from pathlib import Path
    checkpoint_dir = Path(checkpoint_dir)

    if not checkpoint_dir.exists():
        print(f"Checkpoint directory not found: {checkpoint_dir}")
        return None

    ckpt_files = list(checkpoint_dir.glob("*.ckpt"))
    if not ckpt_files:
        print(f"No checkpoint files found in: {checkpoint_dir}")
        return None

    best_ckpt = None
    best_val_loss = float('inf')

    for ckpt_file in ckpt_files:
        if ckpt_file.name == "last.ckpt":
            continue
        filename = ckpt_file.name
        if "val_loss=" in filename:
            try:
                val_loss_str = filename.split("val_loss=")[1].split(".ckpt")[0]
                val_loss = float(val_loss_str)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_ckpt = ckpt_file
            except:
                continue

    if best_ckpt is None:
        best_ckpt = checkpoint_dir / "last.ckpt"
    else:
        print(f"Found best checkpoint: {best_ckpt.name}, val_loss: {best_val_loss:.6f}")

    return str(best_ckpt)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = gan.GAN.add_model_specific_args(parser)

    parser.add_argument(
        "--dataset_folder",
        default=ROOT_DIR / "data" / "precipitation" / "train_test_2020-2025_input-length_5_img-ahead_20_rain-threshold_20_norm.h5",
        type=str,
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--fast_dev_run", type=bool, default=False)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--val_check_interval", type=float, default=None)

    args = parser.parse_args()

    args.n_channels = 5
    args.n_classes = 20
    args.n_output_images = 20
    args.n_masks = 25

    args.gpus = 1
    args.model = "HexMFG"
    args.lr_patience = 10
    args.es_patience = 30

    # GAN options
    args.l = 1000000
    args.disc_every_n_steps = 2

    args.freq_loss_weight = 0.5
    args.high_freq_weight = 0.0
    args.structure_loss_weight = 0.0
    args.temporal_growth_rate = 0.03

    args.kernels_per_layer = 2
    args.dataset_folder = (
        ROOT_DIR / "data" / "precipitation" / "train_test_2020-2025_input-length_5_img-ahead_20_rain-threshold_20_norm.h5"
    )
    args.dataset = "train"
    args.default_save_path = ROOT_DIR / "lightning" / "2020-2025" / f"{args.model}_batch-{args.batch_size}_v1.0"

    args.dropout = 0.5
    torch.multiprocessing.set_sharing_strategy('file_system')

    print("=" * 60)
    print(f"  freq_loss_weight    = {args.freq_loss_weight}")
    print(f"  high_freq_weight    = {args.high_freq_weight}")
    print(f"  structure_loss_weight = {args.structure_loss_weight}")
    print(f"  temporal_growth_rate = {args.temporal_growth_rate}")
    print("=" * 60)

    train_regression(args)
