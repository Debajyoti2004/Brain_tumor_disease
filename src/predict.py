import torch
from models.U_net import UNetModel
from models.Image_classifier import Image_Classifier
import os
import pandas as pd
from PIL import Image
from torchvision import transforms
import numpy as np

image_transform = transforms.Compose([
    transforms.ToTensor(),
])

categories = [] 
num_classes = len(categories)
train_folder_path = r"C:\Users\Debajyoti\OneDrive\Desktop\Brain tumor disease\Dataset\Train folder"
test_folder_path = r"C:\Users\Debajyoti\OneDrive\Desktop\Brain tumor disease\Dataset\Test folder"
predict_folder_path = r"C:\Users\Debajyoti\OneDrive\Desktop\Brain tumor disease\Dataset\Predict_data"

model_segmented_weights_path = r"C:\Users\Debajyoti\OneDrive\Desktop\Brain tumor disease\src\models\segmentation_weight.h5"
model_classification_weights_path = r"C:\Users\Debajyoti\OneDrive\Desktop\Brain tumor disease\src\models\classifier_model.h5"
model1 = UNetModel(3, 1, 0.1)

checkpoints = torch.load(model_segmented_weights_path)
model1.load_state_dict(checkpoints)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model1.to(device)

model2 = Image_Classifier(3,3)
model2.load_state_dict(torch.load(model_classification_weights_path))
model2.to(device)


def predict(model1, model2, train_folder_path, test_folder_path, predict_folder, categories):
    os.makedirs(predict_folder, exist_ok=True)
    
    train_pred_path = os.path.join(predict_folder, "Train_folder_pred_mask")
    test_pred_path = os.path.join(predict_folder, "Test_folder_pred_mask")
    
    os.makedirs(train_pred_path, exist_ok=True)
    os.makedirs(test_pred_path, exist_ok=True)
    
    def predict_and_save_segmentation(model1, folder_path, pred_path,type):
        df = pd.read_csv(os.path.join(folder_path,type))
        
        for root, dirs, files in os.walk(folder_path):
            for dir in dirs:
                if dir == "images":
                    dir_path = os.path.join(root, dir)
                    
                    for image_name in df['image_name']:
                        image_file_path = os.path.join(dir_path, image_name)
                        image = Image.open(image_file_path).convert('RGB')
                        image_tensor = image_transform(image).unsqueeze(0).to(device)
                        
                        with torch.no_grad():
                            pred_mask = model1(image_tensor)
                        
                        pred_mask = pred_mask.squeeze(0).cpu().numpy()
                        pred_mask = (pred_mask * 255).astype(np.uint8)

                        pred_mask_image = Image.fromarray(pred_mask[0], mode='L')
                        pred_mask_file_path = os.path.join(pred_path, image_name)
                        pred_mask_image.save(pred_mask_file_path)
                        
                        print(f"Saved predicted mask for {image_name} at {pred_mask_file_path}")
                        
    def predict_csv(model2, folder_path, pred_path,type):
        df = pd.read_csv(os.path.join(folder_path, type))
        predictions = []
        image_names = []
        
        for root, dirs, files in os.walk(folder_path):
            for dir in dirs:
                if dir == "images":
                    dir_path = os.path.join(root, dir)
                    
                    for image_name in df['image_name']:
                        image_file_path = os.path.join(dir_path, image_name)
                        image = Image.open(image_file_path)
                        image_tensor = image_transform(image).unsqueeze(0).to(device)
                        
                        with torch.no_grad():
                            pred = model2(image_tensor)
                            
                        pred_category = categories[torch.argmax(pred)]
                        image_names.append(image_name)
                        predictions.append(pred_category)
        return image_names, predictions
                            

    predict_and_save_segmentation(model1, train_folder_path, train_pred_path,type="train.csv")
    predict_and_save_segmentation(model1, test_folder_path, test_pred_path,type="test.csv")
    train_pred_csv = predict_csv(model2, train_folder_path, train_pred_path,type = "train.csv")
    test_pred_csv = predict_csv(model2, test_folder_path, test_pred_path,type="test.csv")
    
    train_image_names, train_predicts = train_pred_csv 
    test_image_names, test_predicts = test_pred_csv
    
    train_prediction = {
        'image_name': train_image_names,
        'category_pred': train_predicts
    }
    test_prediction = {
        'image_name': test_image_names,
        'category_pred': test_predicts
    }
    
    train_pred_df = pd.DataFrame(train_prediction)
    test_pred_df = pd.DataFrame(test_prediction)
    train_pred_df.to_csv(os.path.join(train_pred_path, "train_pred.csv"), index=False)
    test_pred_df.to_csv(os.path.join(test_pred_path, "test_pred.csv"), index=False)

predict(model1, model2, train_folder_path, test_folder_path, predict_folder_path, categories)
