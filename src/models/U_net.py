import torch
import torch.nn as nn
import pytorch_lightning as pl
from src.loss_function import ComboLoss
import torchmetrics
import numpy as np

class MyAccuracy(torchmetrics.Metric):
    def __init__(self):
        super().__init__()
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, targets):
        preds = (preds > 0.5).float()
        assert preds.shape == targets.shape
        self.correct += torch.sum(preds == targets)
        self.total += targets.numel()

    def compute(self):
        return self.correct.float() / self.total.float()

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropouts=0, max_pooling=True):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2) if max_pooling else nn.Identity()
        self.dropout = nn.Dropout(dropouts) if dropouts > 0 else nn.Identity()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.dropout(x)
        skip_connection = x
        x = self.max_pool(x)
        return x, skip_connection

class UpsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropouts=0):
        super(UpsamplingBlock, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout(dropouts) if dropouts > 0 else nn.Identity()

    def forward(self, x, skip_connection):
        x = self.upsample(x)
        x = torch.cat([x, skip_connection], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.dropout(x)
        return x

class UNetModel(pl.LightningModule):
    def __init__(self, in_channels, out_channels, dropouts=0.1, learning_rate=1e-3):
        super(UNetModel, self).__init__()
        self.save_hyperparameters()

        # Encoder
        self.conv_block1 = ConvBlock(in_channels, 64, dropouts)
        self.conv_block2 = ConvBlock(64, 128, dropouts)
        self.conv_block3 = ConvBlock(128, 256, dropouts)
        self.conv_block4 = ConvBlock(256, 512, dropouts)
        self.conv_block5 = ConvBlock(512, 1024, dropouts, max_pooling=False)

        # Decoder
        self.upsampling_block1 = UpsamplingBlock(1024, 512, dropouts)
        self.upsampling_block2 = UpsamplingBlock(512, 256, dropouts)
        self.upsampling_block3 = UpsamplingBlock(256, 128, dropouts)
        self.upsampling_block4 = UpsamplingBlock(128, 64, dropouts)

        # Final Convolution
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=1),
            nn.Sigmoid(),  # For binary segmentation
        )

        # Loss function
        self.loss_fn = ComboLoss(smooth=1, alpha=0.5)
        self.accuracy = MyAccuracy()
        self.f1_score = torchmetrics.F1Score(
            task="binary" if out_channels == 1 else "multiclass", num_classes=out_channels
        )

    def forward(self, x):
        x1, skip1 = self.conv_block1(x)
        x2, skip2 = self.conv_block2(x1)
        x3, skip3 = self.conv_block3(x2)
        x4, skip4 = self.conv_block4(x3)
        x5, _ = self.conv_block5(x4)

        x6 = self.upsampling_block1(x5, skip4)
        x7 = self.upsampling_block2(x6, skip3)
        x8 = self.upsampling_block3(x7, skip2)
        x9 = self.upsampling_block4(x8, skip1)

        return self.final_conv(x9)

    def common_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)
        loss = self.loss_fn(preds, y)
        if torch.isnan(loss).any():
            print(f"NaN detected in loss at batch {batch_idx}")
        return loss, preds, y

    def training_step(self, batch, batch_idx):
        loss, preds, y = self.common_step(batch, batch_idx)
        y = (y>=0.5).float()
        self.accuracy.update(preds, y)
        self.f1_score.update(preds, y)

        # Compute metrics
        accuracy = self.accuracy.compute()
        f1_score = self.f1_score.compute()

        # Log metrics
        self.log_dict(
            {"train_loss": loss, "train_accuracy": accuracy, "train_f1_score": f1_score},
            prog_bar=True,
        )

        # Reset metrics
        self.accuracy.reset()
        self.f1_score.reset()
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, y = self.common_step(batch, batch_idx)
        y = (y>=0.5).float()
        self.accuracy.update(preds, y)
        self.f1_score.update(preds, y)

        # Compute metrics
        accuracy = self.accuracy.compute()
        f1_score = self.f1_score.compute()

        # Log metrics
        self.log_dict(
            {"val_loss": loss, "val_accuracy": accuracy, "val_f1_score": f1_score},
            prog_bar=True,
        )

        # Reset metrics
        self.accuracy.reset()
        self.f1_score.reset()
        return loss

    def test_step(self, batch, batch_idx):
        loss, preds, y = self.common_step(batch, batch_idx)
        y = (y>=0.5).float()
        self.accuracy.update(preds, y)
        self.f1_score.update(preds, y)

        # Compute metrics
        accuracy = self.accuracy.compute()
        f1_score = self.f1_score.compute()

        # Log metrics
        self.log_dict(
            {"test_loss": loss, "test_accuracy": accuracy, "test_f1_score": f1_score},
            prog_bar=True,
        )

        # Reset metrics
        self.accuracy.reset()
        self.f1_score.reset()
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

def get_model_size(model):
    param_size = sum(p.numel() for p in model.parameters())
    param_size_mb = param_size * 4 / (1024**2)
    return param_size_mb

if __name__ == "__main__":
    model = UNetModel(3, 1, dropouts=0.1)
    print(f"Model size: {get_model_size(model):.2f} MB")
    sample_input = torch.randn(1, 3, 96, 128)
    output = model(sample_input)
    print(output)
    print("Output shape:", output.shape)
