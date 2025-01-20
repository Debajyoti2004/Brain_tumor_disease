from pycocotools.coco import COCO
import os
import pandas as pd

def create_csv_for_train_and_test(create_masks_and_images, train_path, test_path, train_images_dir, test_images_dir):
    train_csv_path = os.path.join(train_path, "train.csv")
    test_csv_path = os.path.join(test_path, "test.csv")

    def for_train(train_images_dir, train_csv_path, train_path):
        cat_ids = create_masks_and_images.coco_train.getCatIds()
        categories = create_masks_and_images.coco_train.loadCats(cat_ids)
        category_names = {cat['id']: cat['name'] for cat in categories}

        img_file_names = []
        cat_names = []

        for cat_id in cat_ids:
            img_ids = create_masks_and_images.coco_train.getImgIds(catIds=cat_id)

            for img_id in img_ids:
                img_info = create_masks_and_images.coco_train.loadImgs(img_id)[0]
                img_file_name = img_info['file_name']
                img_file_names.append(img_file_name)
                cat_names.append(category_names[cat_id])

        return img_file_names, cat_names

    def for_test(test_images_dir, test_csv_path, test_path):
        cat_ids = create_masks_and_images.coco_test.getCatIds()
        categories = create_masks_and_images.coco_test.loadCats(cat_ids)
        category_names = {cat['id']: cat['name'] for cat in categories}

        img_file_names = []
        cat_names = []

        for cat_id in cat_ids:
            img_ids = create_masks_and_images.coco_test.getImgIds(catIds=cat_id)

            for img_id in img_ids:
                img_info = create_masks_and_images.coco_test.loadImgs(img_id)[0]
                img_file_name = img_info['file_name']
                img_file_names.append(img_file_name)
                cat_names.append(category_names[cat_id])

        return img_file_names, cat_names

    train_image_names, train_cat_names = for_train(train_images_dir, train_csv_path, train_path)
    test_image_names, test_cat_names = for_test(test_images_dir, test_csv_path, test_path)

    train_data = {'image_name': train_image_names, 'category_name': train_cat_names}
    test_data = {'image_name': test_image_names, 'category_name': test_cat_names}

    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)

    train_df.to_csv(train_csv_path, index=False)
    test_df.to_csv(test_csv_path, index=False)
