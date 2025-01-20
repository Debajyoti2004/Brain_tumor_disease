import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics


class MyAccuracy(torchmetrics.Metric):
    def __init__(self):
        super().__init__()
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, targets):
        preds = torch.argmax(preds, dim=1) 
        self.correct += (preds == targets).sum()
        self.total += targets.numel()

    def compute(self):
        return self.correct.float() / self.total.float()


class Image_Classifier(pl.LightningModule):
    def __init__(self, in_channels, num_classes, learning_rate=1e-3):
        super(Image_Classifier, self).__init__()
        self.save_hyperparameters()

        # Model definition
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 12 * 16, num_classes)

        # Loss and metrics
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = MyAccuracy()
        self.f1_score = torchmetrics.F1Score(
            task="binary" if num_classes == 1 else "multiclass", num_classes=num_classes
        )

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.relu(self.conv3(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return x

    def common_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = self.loss_fn(y_pred, y)
        return loss, y_pred, y

    def training_step(self, batch, batch_idx):
        loss, y_pred, y = self.common_step(batch, batch_idx)

        # Update metrics
        self.accuracy.update(y_pred, y)
        self.f1_score.update(y_pred, y)

        # Compute metrics
        accuracy = self.accuracy.compute()
        f1_score = self.f1_score.compute()

        # Log metrics
        self.log_dict(
            {"train_loss": loss, "train_accuracy": accuracy, "train_f1_score": f1_score},
            prog_bar=True,
        )

        # Reset metrics for next step
        self.accuracy.reset()
        self.f1_score.reset()

        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_pred, y = self.common_step(batch, batch_idx)

        # Update metrics
        self.accuracy.update(y_pred, y)
        self.f1_score.update(y_pred, y)

        # Compute metrics
        accuracy = self.accuracy.compute()
        f1_score = self.f1_score.compute()

        # Log metrics
        self.log_dict(
            {"val_loss": loss, "val_accuracy": accuracy, "val_f1_score": f1_score},
            prog_bar=True,
        )

        # Reset metrics for next step
        self.accuracy.reset()
        self.f1_score.reset()

        return loss

    def test_step(self, batch, batch_idx):
        loss, y_pred, y = self.common_step(batch, batch_idx)

        # Update metrics
        self.accuracy.update(y_pred, y)
        self.f1_score.update(y_pred, y)

        # Compute metrics
        accuracy = self.accuracy.compute()
        f1_score = self.f1_score.compute()

        # Log metrics
        self.log_dict(
            {"test_loss": loss, "test_accuracy": accuracy, "test_f1_score": f1_score},
            prog_bar=True,
        )

        # Reset metrics for next step
        self.accuracy.reset()
        self.f1_score.reset()

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


# Example usage
if __name__ == "__main__":
    model = Image_Classifier(3, 4)
    sample_input = torch.randn(1, 3, 96, 128)
    output = model(sample_input)
    print("Output shape:", output.shape)
