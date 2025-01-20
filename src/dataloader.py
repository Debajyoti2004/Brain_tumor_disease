import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataset_class import ImageSegmentationDataset, ImageClassifierDataset, image_transform, mask_transform


class BrainTumorDataModule(pl.LightningDataModule):
    def __init__(self, train_folder, train_csv, test_folder, test_csv, batch_size, transform=None, mask_transform=None):
        super().__init__()
        self.train_folder = train_folder
        self.train_csv = train_csv
        self.test_folder = test_folder
        self.test_csv = test_csv
        self.batch_size = batch_size
        self.transform = transform
        self.mask_transform = mask_transform

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_data = ImageSegmentationDataset(
                folder_path=self.train_folder,
                csv_file=self.train_csv,
                transform=self.transform,
                mask_transform=self.mask_transform
            )
        if stage == 'test' or stage is None:
            self.test_data = ImageSegmentationDataset(
                folder_path=self.test_folder,
                csv_file=self.test_csv,
                transform=self.transform,
                mask_transform=self.mask_transform
            )

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)

class BrainTumorClassificationModule(pl.LightningDataModule):
    def __init__(self, train_folder, train_csv, test_folder, test_csv, batch_size, categories, transform=None):
        super().__init__()
        self.train_folder = train_folder
        self.train_csv = train_csv
        self.test_folder = test_folder
        self.test_csv = test_csv
        self.batch_size = batch_size
        self.transform = transform
        self.categories = categories

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_data = ImageClassifierDataset(
                folder_path=self.train_folder,
                csv_file=self.train_csv,
                categories=self.categories,
                transform=self.transform
            )
        if stage == "test" or stage is None:
            self.test_data = ImageClassifierDataset(
                folder_path=self.test_folder,
                csv_file=self.test_csv,
                categories=self.categories,
                transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)


if __name__ == "__main__":
    train_folder = r'C:\Users\Debajyoti\OneDrive\Desktop\Brain tumor disease\Dataset\Train folder'
    train_csv = r'C:\Users\Debajyoti\OneDrive\Desktop\Brain tumor disease\Dataset\Train folder\train.csv'
    test_folder = r'C:\Users\Debajyoti\OneDrive\Desktop\Brain tumor disease\Dataset\Test folder'
    test_csv = r'C:\Users\Debajyoti\OneDrive\Desktop\Brain tumor disease\Dataset\Test folder\test.csv'
    batch_size = 4
    categories = {
      f'glioma_tumor': 0,
      'meningioma_tumor': 1,
      'pituitary_tumor':2

}
    # Segmentation DataModule
    seg_data_module = BrainTumorDataModule(
        train_folder=train_folder,
        train_csv=train_csv,
        test_folder=test_folder,
        test_csv=test_csv,
        batch_size=batch_size,
        transform=image_transform,
        mask_transform=mask_transform
    )

    # Classification DataModule
    cls_data_module = BrainTumorClassificationModule(
        train_folder=train_folder,
        train_csv=train_csv,
        test_folder=test_folder,
        test_csv=test_csv,
        batch_size=batch_size,
        categories=categories,
        transform=image_transform
    )

    seg_data_module.setup('fit')
    train_loader = seg_data_module.train_dataloader()
    print("Segmentation Train Data:", len(train_loader.dataset))
    print("Segmentation Shape:", next(iter(train_loader))[0].shape, next(iter(train_loader))[1].shape)

    seg_data_module.setup('test')
    test_loader = seg_data_module.test_dataloader()
    print("Segmentation Test Data:", len(test_loader.dataset))

    cls_data_module.setup('fit')
    train_loader_cls = cls_data_module.train_dataloader()
    print("Classification Train Data:", len(train_loader_cls.dataset))
    print("Classification Example Labels:", next(iter(train_loader_cls))[1].shape)

    cls_data_module.setup('test')
    test_loader_cls = cls_data_module.test_dataloader()
    print("Classification Test Data:", len(test_loader_cls.dataset))
