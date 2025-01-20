import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Define transformations
image_transform = transforms.Compose([
    transforms.ToTensor(),
])

mask_transform = transforms.Compose([
    transforms.ToTensor(),
])

categories = {
    'glioma_tumor': 0,
    'meningioma_tumor': 1,
    'pituitary_tumor':2

}

# Image Segmentation Dataset
class ImageSegmentationDataset(Dataset):
    def __init__(self, folder_path, csv_file, transform=None, mask_transform=None):
        self.folder_path = folder_path
        self.csv_file = csv_file
        self.transform = transform
        self.mask_transform = mask_transform
        self.data = pd.read_csv(self.csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        try:
            image_path = os.path.join(self.folder_path, 'images', self.data.iloc[index, 0])
            image = Image.open(image_path).convert('RGB')
            mask_path = os.path.join(self.folder_path, 'masks', self.data.iloc[index, 0])
            mask = Image.open(mask_path).convert('L')
            
            if self.transform:
                image = self.transform(image)
            if self.mask_transform:
                mask = self.mask_transform(mask)
                
            return image, mask
        except Exception as e:
            raise RuntimeError(f"Error loading data at index {index}: {e}")

# Image Classification Dataset
class ImageClassifierDataset(Dataset):
    def __init__(self, folder_path, csv_file, categories, transform=None):
        self.folder_path = folder_path
        self.csv_file = csv_file
        self.transform = transform
        self.categories = categories
        self.data = pd.read_csv(self.csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        try:
            image_path = os.path.join(self.folder_path, 'images', self.data.iloc[index, 0])
            image = Image.open(image_path).convert('RGB')
            class_name = self.data.iloc[index, 1]
            label = self.categories[class_name]
            
            if self.transform:
                image = self.transform(image)
                
            return image, label
        except Exception as e:
            raise RuntimeError(f"Error loading data at index {index}: {e}")


# Main script
if __name__ == "__main__":
    folder_path = r'C:\Users\Debajyoti\OneDrive\Desktop\Brain tumor disease\Dataset\Train folder'
    csv_file = r'C:\Users\Debajyoti\OneDrive\Desktop\Brain tumor disease\Dataset\Train folder\train.csv'

    # Segmentation Dataset
    segmentation_dataset = ImageSegmentationDataset(
        folder_path=folder_path,
        csv_file=csv_file,
        transform=image_transform,
        mask_transform=mask_transform
    )
    
    segmentation_loader = DataLoader(segmentation_dataset, batch_size=4, shuffle=True)

    print("Segmentation Dataset:")
    for images, masks in segmentation_loader:
        print(f"Images shape: {images.shape}, Masks shape: {masks.shape}")
        break

    # Classification Dataset
    classification_dataset = ImageClassifierDataset(
        folder_path=folder_path,
        csv_file=csv_file,
        categories=categories,
        transform=image_transform
    )
    
    classification_loader = DataLoader(classification_dataset, batch_size=4, shuffle=True)

    print("\nClassification Dataset:")
    for images, labels in classification_loader:
        print(f"Images shape: {images.shape}, Labels: {labels.shape}")
        break
