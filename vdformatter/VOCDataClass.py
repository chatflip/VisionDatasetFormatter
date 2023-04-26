from dataclasses import dataclass


@dataclass
class VOCBndBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int


@dataclass
class VOCObject:
    name: str
    pose: str = ""
    truncated: int = 0
    occluded: int = 0
    bndbox: VOCBndBox  # type: ignore
    difficult: int = 0
