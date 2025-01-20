from pytorch_lightning.callbacks import Callback, ModelCheckpoint, LearningRateMonitor, EarlyStopping
import os

class CustomCallback(Callback):
    def __init__(self, log_dir=None):
        super().__init__()
        self.log_dir = log_dir

    def on_train_start(self, trainer, pl_module):
        print("Training started.")
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            print(f"Logging directory: {self.log_dir}")

    def on_train_epoch_end(self, trainer, pl_module):
        print(f"Epoch {trainer.current_epoch} finished.")

    def on_validation_end(self, trainer, pl_module):
        print("Validation finished.")

checkpoint_callback = ModelCheckpoint(
    dirpath=r"C:\Users\Debajyoti\OneDrive\Desktop\Brain tumor disease\checkpoints",
    filename="{epoch:02d}-{train_accuracy:.4f}",
    save_top_k=3,
    verbose=True,
    monitor="train_accuracy",
    mode="max",
)

lr_monitor = LearningRateMonitor(logging_interval="step")

early_stopping = EarlyStopping(
    monitor="train_accuracy",
    patience=5,
    verbose=True,
    mode="max",
)
