import json
import os
import shutil

import cv2
from COCODataclass import (  # noqa: F401
    COCOImage,
    COCOInfo,
    COCOInstanceAnnotation,
    COCOInstanceCategory,
    COCOLicense,
)
from pycocotools.coco import COCO


class COCOInstance:
    def __init__(self, dataset_root: str) -> None:
        self.dataset_root = dataset_root
        self.image_root = os.path.join(self.dataset_root, "images")
        self.ann_root = os.path.join(self.dataset_root, "annotations")
        os.makedirs(self.image_root, exist_ok=True)
        os.makedirs(self.ann_root, exist_ok=True)
        self.imgId = 0
        self.annId = 0
        self.info = COCOInfo()
        self.licenses = COCOLicense()
        self.images: list[COCOImage] = []
        self.annotations: list[COCOInstanceAnnotation] = []
        self.categories: list[COCOInstanceCategory] = []

    def get_ImgId(self) -> int:
        return self.imgId

    def get_AnnId(self) -> int:
        return self.annId

    def add_image_from_path(self, image_path: str) -> int:
        self.imgId += 1
        image = cv2.imread(image_path)
        height, width = image.shape[:2]
        file_name = os.path.basename(image_path)
        dst_image_path = os.path.join(self.image_root, file_name)
        image_data = COCOImage(
            id=self.imgId,
            file_name=file_name,
            height=height,
            width=width,
        )
        self.images.append(image_data)
        if not os.path.exists(dst_image_path):
            shutil.copy2(image_path, dst_image_path)
        return self.imgId

    def add_image(self, image_data: COCOImage) -> int:
        if image_data.id == -1:
            self.imgId += 1
            image_data.id = self.imgId
        self.images.append(image_data)
        return self.imgId

    def add_instance_annotation(self, ann_data: COCOInstanceAnnotation) -> int:
        if ann_data.id == -1:
            self.annId += 1
            ann_data.id = self.annId
        self.annotations.append(ann_data)
        return self.annId

    def add_category(self, category_data: COCOInstanceCategory) -> None:
        self.categories.append(category_data)

    def write(self) -> None:
        coco_dict = {
            "info": self.info.to_dict(),
            "licenses": self.licenses.to_dict(),
            "images": [image.to_dict() for image in self.images],
            "annotations": [ann.to_dict() for ann in self.annotations],
            "categories": [category.to_dict() for category in self.categories],
        }
        json_path = os.path.join(self.ann_root, "instances.json")
        with open(json_path, "w") as f:
            json.dump(coco_dict, f)
