from root import ROOT_DIR

import lightning.pytorch as pl

from torch import nn, optim, multiprocessing
import torch
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

import torchvision
from utils import dataset_precip
from models.discriminator import *
from models.unet_precip_regression_lightning import HexMF_UNet, HexMF_GNet

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import math


class GAN_base(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--n_channels", type=int, default=5)
        parser.add_argument("--n_classes", type=int, default=20)
        parser.add_argument("--kernels_per_layer", type=int, default=1)
        parser.add_argument("--bilinear", type=bool, default=True)
        parser.add_argument("--reduction_ratio", type=int, default=16)
        parser.add_argument("--lr_patience", type=int, default=5)
        parser.add_argument("--freq_loss_weight", type=float, default=5.0)
        parser.add_argument("--structure_loss_weight", type=float, default=10.0)
        parser.add_argument("--temporal_growth_rate", type=float, default=0.05)
        parser.add_argument("--high_freq_weight", type=float, default=2.0)
        parser.add_argument("--runs", type=int, default=1)
        return parser

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.automatic_optimization = False
        if isinstance(hparams, dict):
            n_channels = hparams.get('n_channels', 5)
            n_classes = hparams.get('n_classes', 20)
        else:
            n_channels = getattr(hparams, 'n_channels', 5)
            n_classes = getattr(hparams, 'n_classes', 20)
        in_channels = n_channels + n_classes

        self.generator = HexMF_UNet(hparams=hparams)

        self.discriminator = HexMF_DNet(in_channels=in_channels)

        self.g_losses = []
        self.d_losses = []

        self.loss_history = {
            'temporal_mse': [],
            'freq_loss': [],
            'struct_loss': [],
            'g_total': [],
            'g_loss': []
        }


        self.visualize_losses = getattr(hparams, 'visualize_losses', True)
        self.log("val_g_loss", float("inf"), prog_bar=False)
        self.log("val_d_loss", float("inf"), prog_bar=False)

        # ========== Sobel==========
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        # =====================================================

    def forward(self, x):
        return self.generator(x)

    def configure_optimizers(self):
        lr = self.hparams.learning_rate

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
        scheduler_g = {
            "scheduler": optim.lr_scheduler.ReduceLROnPlateau(
                opt_g, mode="min", factor=0.1, patience=self.hparams.lr_patience,
            ),
            "monitor": "val_g_loss",
        }
        scheduler_d = {
            "scheduler": optim.lr_scheduler.ReduceLROnPlateau(
                opt_d, mode="min", factor=0.1, patience=self.hparams.lr_patience
            ),
            "monitor": "val_d_loss",
        }
        return [opt_g, opt_d], [scheduler_g, scheduler_d]

    def on_validation_epoch_end(self):
        plt.close("all")

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def loss_func(self, y_pred, y_true):
        return nn.functional.mse_loss(
            y_pred, y_true, reduction="mean"
        )

    def frequency_loss(self, y_pred, y_true):
        """
        y_pred, y_true: (B, C, H, W)
        """
        high_freq_weight = self.hparams.high_freq_weight

        # 2D FFT
        fft_pred = torch.fft.rfft2(y_pred, norm="ortho")
        fft_true = torch.fft.rfft2(y_true, norm="ortho")

        diff = torch.abs(fft_pred) - torch.abs(fft_true)

        B, C, H, W_freq = diff.shape
        freq_y = torch.arange(H, device=diff.device, dtype=torch.float32).unsqueeze(1)
        freq_x = torch.arange(W_freq, device=diff.device, dtype=torch.float32).unsqueeze(0)
        # [0, 1]
        max_dist = math.sqrt((H / 2) ** 2 + W_freq ** 2)
        dist = torch.sqrt((freq_y - H / 2) ** 2 + freq_x ** 2) / max_dist  # (H, W_freq)

        weight = 1.0 + (high_freq_weight - 1.0) * dist  # (H, W_freq)
        weight = weight.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W_freq)

        freq_loss = torch.mean(weight * diff ** 2)
        return freq_loss

    def structure_loss(self, y_pred, y_true):
        """Sobel
        y_pred, y_true: (B, C, H, W)
        """
        B, C, H, W = y_pred.shape
        # batch
        pred_flat = y_pred.reshape(B * C, 1, H, W)
        true_flat = y_true.reshape(B * C, 1, H, W)

        # Sobel
        pred_gx = F.conv2d(pred_flat, self.sobel_x, padding=1)
        pred_gy = F.conv2d(pred_flat, self.sobel_y, padding=1)
        true_gx = F.conv2d(true_flat, self.sobel_x, padding=1)
        true_gy = F.conv2d(true_flat, self.sobel_y, padding=1)

        pred_grad = torch.sqrt(pred_gx ** 2 + pred_gy ** 2 + 1e-8)
        true_grad = torch.sqrt(true_gx ** 2 + true_gy ** 2 + 1e-8)

        return F.mse_loss(pred_grad, true_grad)

    def temporal_weighted_loss(self, y_pred, y_true, visualize=False):
        """
        y_pred, y_true: (B, C, H, W), C = num_output_images
        visualize: True/False
        """
        growth_rate = self.hparams.temporal_growth_rate
        num_steps = y_pred.shape[1]

        #  w_t = exp(growth_rate * t), t = 0, 1, ..., num_steps-1
        t = torch.arange(num_steps, device=y_pred.device, dtype=torch.float32)
        weights = torch.exp(growth_rate * t)  # (C,)
        weights = weights / weights.mean()
        weights = weights.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # (1, C, 1, 1)

        # weighted MSE
        per_pixel_loss = (y_pred - y_true) ** 2  # (B, C, H, W)
        weighted_loss = (weights * per_pixel_loss).mean()

        if visualize and hasattr(self.hparams, 'default_save_path') and self.hparams.default_save_path:
            import matplotlib.pyplot as plt
            import numpy as np

            weights_np = torch.exp(growth_rate * t).numpy()
            weights_np_normalized = weights_np / np.mean(weights_np)

            plt.figure(figsize=(10, 6))
            plt.plot(t.cpu().numpy(), weights_np_normalized, 'o-', linewidth=2)
            plt.title(f'Temporal Weights Distribution (growth_rate={growth_rate})')
            plt.xlabel('Time Step')
            plt.ylabel('Normalized Weight')
            plt.grid(True, alpha=0.3)

            save_path = self.hparams.default_save_path / f'temporal_weights_growth_rate_{growth_rate:.4f}.png'
            plt.savefig(save_path)
            plt.close()

        return weighted_loss

    def visualize_growth_rate_impact(self, num_steps=20, growth_rates=[0.01, 0.03, 0.05, 0.07, 0.1]):
        """
        growth_rate
        num_steps
        growth_rates
        """
        import matplotlib.pyplot as plt
        import numpy as np

        plt.figure(figsize=(12, 8))

        for growth_rate in growth_rates:
            t = np.arange(num_steps)
            weights = np.exp(growth_rate * t)
            weights_normalized = weights / np.mean(weights)
            plt.plot(t, weights_normalized, 'o-', linewidth=2, label=f'growth_rate={growth_rate}')

        plt.title('Impact of growth_rate on Temporal Weights')
        plt.xlabel('Time Step')
        plt.ylabel('Normalized Weight')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if hasattr(self.hparams, 'default_save_path') and self.hparams.default_save_path:
            save_path = self.hparams.default_save_path / 'growth_rate_impact.png'
            plt.savefig(save_path)
            print(f"Growth rate impact visualization saved to: {save_path}")
        else:
            save_path = 'growth_rate_impact.png'
            plt.savefig(save_path)
            print(f"Growth rate impact visualization saved to: {save_path}")

        plt.close()

    def get_temporal_weights(self, num_steps, growth_rate):
        """
        num_steps
        growth_rate
        """
        t = torch.arange(num_steps, dtype=torch.float32)
        weights = torch.exp(growth_rate * t)
        weights = weights / weights.mean()
        return weights.numpy()

    def compute_generator_structural_loss(self, y_pred, y_true):
        freq_w = self.hparams.freq_loss_weight
        struct_w = self.hparams.structure_loss_weight

        # 1
        temporal_mse = self.temporal_weighted_loss(y_pred, y_true)

        # 2
        freq_loss = self.frequency_loss(y_pred, y_true)

        # 3
        struct_loss = self.structure_loss(y_pred, y_true)

        total = temporal_mse + freq_w * freq_loss + struct_w * struct_loss

        return total, temporal_mse, freq_loss, struct_loss

    # ==================================================================

    def training_step(self, batch, batchid):
        # lamda param for generator loss
        l = self.hparams.l

        imgs, masks_in, tar, masks_true = batch
        optimizer_g, optimizer_d = self.optimizers()
        scheduler_g, scheduler_d = self.lr_schedulers()

        # how well can it label as real?
        valid = torch.ones((imgs.size(0), 1, 4, 4))
        valid = valid.type_as(imgs)

        # how well can it label as fake?
        fake = torch.zeros((imgs.size(0), 1, 4, 4))
        fake = fake.type_as(imgs)

        # train discriminator
        generated_imgs = self.generator(imgs)
        self.toggle_optimizer(optimizer_d)

        real_loss = self.adversarial_loss(self.discriminator(imgs, tar), valid)
        fake_loss = self.adversarial_loss(self.discriminator(imgs, generated_imgs.detach()), fake)

        d_loss = (real_loss + fake_loss)
        self.log("d_loss", d_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        if batchid % self.hparams.disc_every_n_steps == 0:
            optimizer_d.zero_grad()
            self.manual_backward(d_loss)
            optimizer_d.step()

        self.untoggle_optimizer(optimizer_d)

        # Train generator
        self.toggle_optimizer(optimizer_g)
        optimizer_g.zero_grad()

        generated_imgs = self(imgs)

        g_loss = self.adversarial_loss(self.discriminator(imgs, generated_imgs), valid)
        structural_g_loss, temporal_mse, freq_loss, struct_loss = self.compute_generator_structural_loss(
            generated_imgs, tar
        )
        total_g_loss = g_loss + l * structural_g_loss

        self.log("g_total", total_g_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("g_loss", g_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("g_temporal_mse", temporal_mse, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("g_freq_loss", freq_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("g_struct_loss", struct_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        # ===========================================================

        self.manual_backward(total_g_loss)
        optimizer_g.step()
        self.untoggle_optimizer(optimizer_g)

        if batchid % 100 == 0:
            display_list = [imgs[0], tar[0], generated_imgs[0]]
            title = ['Input Image', 'Ground Truth', 'Predicted Image']
            n_display = min(5, self.hparams.n_channels, self.hparams.n_classes)
            f, axarr = plt.subplots(3, n_display)
            f.set_figwidth(20)
            for i in range(3):
                for j in range(n_display):
                    axarr[i, j].imshow(display_list[i][j, :, :].detach().cpu().numpy())
            plt.axis("off")
            plt.savefig(self.hparams.default_save_path / "imgs.png")
            plt.close()

            xs = [x * 10 for x in range(len(self.g_losses))]
            xs = xs[10:]
            fig, ax1 = plt.subplots()
            color = 'tab:red'
            ax1.set_xlabel('steps')
            ax1.set_ylabel('g_loss', color=color)
            g_losses = self.g_losses[10:]
            ax1.plot(xs, g_losses, color=color)
            ax1.tick_params(axis='y', labelcolor=color)

            ax2 = ax1.twinx()
            color = 'tab:blue'
            ax2.set_ylabel('d_loss', color=color)
            d_losses = self.d_losses[10:]
            ax2.plot(xs, d_losses, color=color)
            ax2.tick_params(axis='y', labelcolor=color)
            fig.tight_layout()
            plt.savefig(self.hparams.default_save_path / "losses.png")
            plt.close()

            if self.visualize_losses:
                self.plot_individual_losses(xs)

        if batchid > 0 and batchid % 10 == 0:
            self.g_losses.append(total_g_loss.item())
            self.d_losses.append(d_loss.item())
            self.loss_history['temporal_mse'].append(temporal_mse.item())
            self.loss_history['freq_loss'].append(freq_loss.item())
            self.loss_history['struct_loss'].append(struct_loss.item())
            self.loss_history['g_total'].append(total_g_loss.item())
            self.loss_history['g_loss'].append(g_loss.item())

        if self.trainer.is_last_batch and (self.trainer.current_epoch + 1) % 1 == 0 and self.trainer.current_epoch > 0:
            scheduler_d.step(self.trainer.callback_metrics["val_d_loss"])
            scheduler_g.step(self.trainer.callback_metrics["val_g_loss"])

        return total_g_loss

    def validation_step(self, batch, batch_idx):
        l = self.hparams.l

        imgs, masks_in, tar, masks_true = batch

        valid = torch.ones((imgs.size(0), 1, 4, 4))
        valid = valid.type_as(imgs)

        fake = torch.zeros((imgs.size(0), 1, 4, 4))
        fake = fake.type_as(imgs)

        generated_imgs = self.generator(imgs)

        real_loss = self.adversarial_loss(self.discriminator(imgs, tar), valid)
        fake_loss = self.adversarial_loss(self.discriminator(imgs, generated_imgs.detach()), fake)

        d_loss = (real_loss + fake_loss)
        self.log("val_d_loss", d_loss, prog_bar=True)

        generated_imgs = self(imgs)
        g_loss = self.adversarial_loss(self.discriminator(imgs, generated_imgs), valid)

        structural_g_loss, temporal_mse, freq_loss, struct_loss = self.compute_generator_structural_loss(
            generated_imgs, tar
        )
        total_g_loss = g_loss + l * structural_g_loss

        num_steps = generated_imgs.shape[1]
        mid = num_steps // 2
        with torch.no_grad():
            mae_short = F.l1_loss(generated_imgs[:, :mid], tar[:, :mid])
            mae_long = F.l1_loss(generated_imgs[:, mid:], tar[:, mid:])

        self.log("val_loss", structural_g_loss * 100, prog_bar=True)
        self.log("val_g_loss", total_g_loss, prog_bar=True)
        self.log("val_freq_loss", freq_loss, prog_bar=False, logger=True)
        self.log("val_struct_loss", struct_loss, prog_bar=False, logger=True)
        self.log("val_temporal_mse", temporal_mse, prog_bar=False, logger=True)
        self.log("val_mae_short", mae_short, prog_bar=True, logger=True)
        self.log("val_mae_long", mae_long, prog_bar=True, logger=True)
        # ===================================================

    def plot_individual_losses(self, steps):
        """
        steps
        """
        import matplotlib.pyplot as plt

        for loss_name, loss_values in self.loss_history.items():
            if len(loss_values) > 10:
                plot_steps = steps[:len(loss_values) - 10]
                plot_values = loss_values[10:]

                plt.figure(figsize=(12, 6))
                plt.plot(plot_steps, plot_values, 'o-', linewidth=1, markersize=2)
                plt.title(f'{loss_name} Loss Over Time')
                plt.xlabel('Steps')
                plt.ylabel('Loss Value')
                plt.grid(True, alpha=0.3)

                save_path = self.hparams.default_save_path / f"{loss_name}_loss.png"
                plt.savefig(save_path)
                plt.close()
                print(f"Saved {loss_name} loss plot to: {save_path}")


class GAN(GAN_base):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parent_parser = GAN_base.add_model_specific_args(parent_parser)
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--num_input_images", type=int, default=5)
        parser.add_argument("--num_output_images", type=int, default=20)
        parser.add_argument("--valid_size", type=float, default=0.1)
        parser.add_argument("--visualize_losses", type=bool, default=True,
                            help="Whether to generate loss visualization plots")
        parser.n_channels = parser.parse_args().num_input_images
        parser.n_classes = 20
        return parser

    def __init__(self, hparams):
        super(GAN, self).__init__(hparams=hparams)
        self.train_dataset = None
        self.valid_dataset = None
        self.train_sampler = None
        self.valid_sampler = None

    def prepare_data(self):
        train_transform = None
        valid_transform = None
        precip_dataset = dataset_precip.precipitation_maps_masked_h5
        self.train_dataset = precip_dataset(
            in_file=self.hparams.dataset_folder,
            num_input_images=self.hparams.num_input_images,
            num_output_images=self.hparams.num_output_images,
            mode=self.hparams.dataset,
            transform=train_transform,
        )
        self.valid_dataset = precip_dataset(
            in_file=self.hparams.dataset_folder,
            num_input_images=self.hparams.num_input_images,
            num_output_images=self.hparams.num_output_images,
            mode=self.hparams.dataset,
            transform=valid_transform,
        )

        num_train = len(self.train_dataset)
        indices = list(range(num_train))
        split = int(np.floor(self.hparams.valid_size * num_train))

        np.random.seed(123)
        np.random.shuffle(indices)

        train_idx, valid_idx = indices[split:], indices[:split]
        self.train_sampler = SubsetRandomSampler(train_idx)
        self.valid_sampler = SubsetRandomSampler(valid_idx)

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            sampler=self.train_sampler,
            num_workers=16,
            pin_memory=True,
        )
        return train_loader

    def val_dataloader(self):
        valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=self.hparams.batch_size,
            sampler=self.valid_sampler,
            num_workers=16,
            pin_memory=True,
        )
        return valid_loader