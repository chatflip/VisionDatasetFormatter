from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class COCOInfo:
    def __init__(
        self,
        description: str = "",
        url: str = "",
        version: str = "1.0",
        year: int = 2017,
        contributor: str = "",
        date_created: str = "",
    ):
        self.description = description
        self.url = url
        self.version = version
        self.year = year
        self.contributor = contributor
        if date_created:
            self.date_created = date_created
        else:
            self.date_created = datetime.now().strftime("%Y/%m/%d")


@dataclass
class COCOLicense:
    url: str = ""
    id: int = 0
    name: str = ""


@dataclass
class COCOImage:
    file_name: str
    height: int
    width: int
    id: int = -1
    coco_url: str = ""
    license: int = 0
    data_captured: str = ""
    flickr_url: str = ""


@dataclass
class COCOInstanceAnnotation:
    category_id: int
    id: int = -1
    image_id: int = -1
    area: float = -1.0
    segmentation: list[list[float]] = field(default_factory=list)
    iscrowd: int = 0
    bbox: list[float] = field(default_factory=list)


@dataclass
class COCOInstanceCategory:
    id: int
    name: str
    supercategory: str = ""
