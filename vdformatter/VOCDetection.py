from __future__ import annotations

import os
import re
import shutil
import xml.etree.ElementTree as ET
from xml.dom import minidom

import cv2
from VOCDataClass import VOCObject


class VOCDetection:
    def __init__(
        self, image_root: str = "JPEGImages", ann_root: str = "Annotations"
    ) -> None:
        self.image_root = image_root
        self.ann_root = ann_root
        os.makedirs(self.image_root, exist_ok=True)
        os.makedirs(self.ann_root, exist_ok=True)

    def write(self, src_image_path: str, ann_data: list[VOCObject]) -> None:
        root = ET.Element("annotation")
        self.add_image_meta(root, src_image_path)
        self.add_annotation(root, ann_data)

        image_name = os.path.basename(src_image_path)
        name, _ = os.path.splitext(image_name)

        # Copy image file
        dst_image_path = os.path.join(self.image_root, image_name)
        if not os.path.exists(dst_image_path):
            shutil.copy2(src_image_path, dst_image_path)

        # Write annotation file
        xml_path = os.path.join(self.ann_root, f"{name}.xml")
        with open(xml_path, mode="w") as f:
            f.write(to_prettify(root))

    def add_image_meta(self, root: ET.Element, image_path: str) -> None:
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        filename = os.path.basename(image_path)
        height, width, channels = image.shape
        ET.SubElement(root, "filename").text = filename
        size = ET.SubElement(root, "size")
        ET.SubElement(size, "width").text = str(width)
        ET.SubElement(size, "height").text = str(height)
        ET.SubElement(size, "depth").text = str(channels)

    def add_annotation(self, root: ET.Element, ann_data: list[VOCObject]) -> None:
        for ann in ann_data:
            object = ET.SubElement(root, "object")
            ET.SubElement(object, "name").text = ann.name
            ET.SubElement(object, "pose").text = ann.pose
            ET.SubElement(object, "truncated").text = str(ann.truncated)
            ET.SubElement(object, "occluded").text = str(ann.occluded)
            bndbox = ET.SubElement(object, "bndbox")
            ET.SubElement(bndbox, "xmin").text = str(ann.bndbox.xmin)
            ET.SubElement(bndbox, "ymin").text = str(ann.bndbox.ymin)
            ET.SubElement(bndbox, "xmax").text = str(ann.bndbox.xmax)
            ET.SubElement(bndbox, "ymax").text = str(ann.bndbox.ymax)
            ET.SubElement(object, "difficult").text = str(ann.difficult)


def to_prettify(elem: ET.Element) -> str:
    rough_string = ET.tostring(elem, "utf-8")
    reparsed = minidom.parseString(rough_string)
    pretty: str = re.sub(r"[\t ]+\n", "", reparsed.toprettyxml(indent="\t"))
    pretty = pretty.replace(">\n\n\t<", ">\n\t<")
    pretty = pretty.replace('<?xml version="1.0" ?>\n', "")
    return pretty
