import os
import numpy as np
from PIL import Image,ImageDraw
from pycocotools.coco import COCO
import matplotlib.pyplot as plt


class Create_masks_and_images:
    def __init__(self, train_json_path, test_json_path, train_image_dir, test_image_dir):
        self.train_json_path = train_json_path
        self.test_json_path = test_json_path
        self.train_image_dir = train_image_dir
        self.test_image_dir = test_image_dir
        self.coco_train = COCO(self.train_json_path)
        self.coco_test = COCO(self.test_json_path)

    def create_dir_for_train_test_images(self):
        if not os.path.exists(self.train_image_dir):
            os.makedirs(self.train_image_dir)

        if not os.path.exists(self.test_image_dir):
            os.makedirs(self.test_image_dir)

    def create_masks_and_images_for_train(self, train_images_dir):
        sub_dir_images = "images"
        sub_dir_masks = "masks"
        images_path = os.path.join(self.train_image_dir, sub_dir_images)
        mask_path = os.path.join(self.train_image_dir, sub_dir_masks)

        os.makedirs(images_path, exist_ok=True)
        os.makedirs(mask_path, exist_ok=True)

        img_ids = self.coco_train.getImgIds()

        print("Fetching.... Training Image")
        for img_id in img_ids:
            img_info = self.coco_train.loadImgs(img_id)[0]
            img_path = os.path.join(train_images_dir, img_info['file_name'])

            img = Image.open(img_path).convert("RGB")
            width, height = img_info["width"], img_info["height"]

            ann_ids = self.coco_train.getAnnIds(imgIds=img_id)
            anns = self.coco_train.loadAnns(ann_ids)

            mask = Image.new('L', (width, height), 0)

            for ann in anns:
                segmentation = ann['segmentation']

                if isinstance(segmentation, list):
                    for seg in segmentation:
                        draw = ImageDraw.Draw(mask)
                        draw.polygon(seg, fill=1)
                else:
                    draw = ImageDraw.Draw(mask)
                    draw.polygon(segmentation, fill=1)

            img.save(os.path.join(images_path, img_info['file_name']))
            mask.save(os.path.join(mask_path, img_info['file_name']))

            print(f"Processed {img_info['file_name']}")

    def create_masks_and_images_for_test(self, test_images_dir):
        sub_dir_images = "images"
        sub_dir_masks = "masks"
        images_path = os.path.join(self.test_image_dir, sub_dir_images)
        mask_path = os.path.join(self.test_image_dir, sub_dir_masks)

        os.makedirs(images_path, exist_ok=True)
        os.makedirs(mask_path, exist_ok=True)

        img_ids = self.coco_test.getImgIds()

        print("Fetching... Testing Image")
        for img_id in img_ids:
            img_info = self.coco_test.loadImgs(img_id)[0]
            img_path = os.path.join(test_images_dir, img_info['file_name'])

            img = Image.open(img_path).convert("RGB")
            width, height = img_info["width"], img_info["height"]

            ann_ids = self.coco_test.getAnnIds(imgIds=img_id)
            anns = self.coco_test.loadAnns(ann_ids)

            mask = Image.new('L', (width, height), 0)

            for ann in anns:
                segmentation = ann['segmentation']

                if isinstance(segmentation, list):
                    for seg in segmentation:
                        draw = ImageDraw.Draw(mask)
                        draw.polygon(seg, fill=1)
                else:
                    draw = ImageDraw.Draw(mask)
                    draw.polygon(segmentation, fill=1)

            img.save(os.path.join(images_path, img_info['file_name']))
            mask.save(os.path.join(mask_path, img_info['file_name']))

            print(f"Processed {img_info['file_name']}")
