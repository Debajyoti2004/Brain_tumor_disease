import pytorch_lightning as pl
from dataloader import BrainTumorDataModule,BrainTumorClassificationModule
from models.U_net import UNetModel
from models.Image_classifier import Image_Classifier
from torchvision import transforms
import torch
from callbacks import lr_monitor,early_stopping,checkpoint_callback,CustomCallback
from pytorch_lightning.loggers import TensorBoardLogger

categories = {
    'glioma_tumor': 0,
    'meningioma_tumor': 1,
    'pituitary_tumor':2

}
image_transform = transforms.Compose([
    transforms.ToTensor(),
])

mask_transform = transforms.Compose([
    transforms.ToTensor(),
])

# Instantiate the DataModule
data_module1 = BrainTumorDataModule(
    train_folder=r'C:\Users\Debajyoti\OneDrive\Desktop\Brain tumor disease\Dataset\Train folder',
    train_csv=r'C:\Users\Debajyoti\OneDrive\Desktop\Brain tumor disease\Dataset\Train folder\train.csv',
    test_folder=r'C:\Users\Debajyoti\OneDrive\Desktop\Brain tumor disease\Dataset\Test folder',
    test_csv=r'C:\Users\Debajyoti\OneDrive\Desktop\Brain tumor disease\Dataset\Test folder\test.csv',
    batch_size=4,
    transform=image_transform,
    mask_transform=mask_transform
)
data_module2 = BrainTumorClassificationModule(
    train_folder = r'C:\Users\Debajyoti\OneDrive\Desktop\Brain tumor disease\Dataset\Train folder',
    train_csv = r'C:\Users\Debajyoti\OneDrive\Desktop\Brain tumor disease\Dataset\Train folder\train.csv',
    test_folder = r'C:\Users\Debajyoti\OneDrive\Desktop\Brain tumor disease\Dataset\Test folder',
    test_csv = r'C:\Users\Debajyoti\OneDrive\Desktop\Brain tumor disease\Dataset\Test folder\test.csv',
    batch_size = 4,
    categories = categories,
    transform = image_transform
)

segmentation_model = UNetModel(3, 1, dropouts=0.1, learning_rate=1e-3)
classifier_model = Image_Classifier(3,3)


logger = TensorBoardLogger("tb_logs",name = "Brain_tumor_disease ")
trainer1 = pl.Trainer(
    accelerator="gpu",
    devices=1, 
    min_epochs=3,
    max_epochs=10,
    precision='16-mixed',
    callbacks=[lr_monitor, early_stopping, checkpoint_callback, CustomCallback(log_dir="logs")]
)

# Train and Test
trainer1.fit(segmentation_model, datamodule=data_module1)
trainer1.test(segmentation_model, datamodule=data_module1)
torch.save(segmentation_model.state_dict(), r"C:\Users\Debajyoti\OneDrive\Desktop\Brain tumor disease\src\models\segmentation_weight.h5")
print(r"Model weights saved to 'C:\Users\Debajyoti\OneDrive\Desktop\Brain tumor disease\src\models\segmentation_weight.h5'")


trainer2 = pl.Trainer(
    accelerator="gpu",
    devices=1, 
    min_epochs=3,
    max_epochs=10,
    precision='16-mixed',
    callbacks=[lr_monitor, early_stopping, checkpoint_callback, CustomCallback(log_dir="logs")]
)

trainer2.fit(classifier_model,datamodule=data_module2)
trainer2.test(classifier_model,datamodule=data_module2)
saved_path = r'C:\Users\Debajyoti\OneDrive\Desktop\Brain tumor disease\src\models\classifier_model.h5'
torch.save(classifier_model.state_dict(),saved_path)
print("Model saved at set path successfully!")
