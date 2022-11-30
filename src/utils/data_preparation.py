"""Utilitary functions to process data."""


from typing import Any, Dict

from PIL import Image

import torch
from torchvision.transforms import Compose


def pil_loader(path: str) -> Image:
    """Open path as file to avoid ResourceWarning.

    See (https://github.com/python-pillow/Pillow/issues/835).
    """
    with open(path, "rb") as f:
        with Image.open(f) as img:
            return img.convert("RGB")


def apply_transform(transform: Compose, examples: Dict[str, Any]) -> Dict[str, Any]:
    """Utilitary for applying transforms to *huggingface* datasets.

    Parameters
    ----------
    transform : torchvision.transforms.Compose
        A sequence of transformations to apply to input images.

    examples : dict of {str: any}
        A dictionary containing images and labels.

    Returns
    -------
    examples : dict of {str: any}
        Same dictionary with an appended key for transformed images.
    """
    examples["pixel_values"] = [transform(image.convert("RGB")) for image in examples["image"]]

    return examples


def collate_fn(examples: Dict[str, Any]) -> Dict[str, Any]:
    """Utilitary for collating *huggingface* dataset into *torch* dataloaders.

    Parameters
    ----------
    examples : dict of {str: any}
        A dictionary containing images and labels.

    Returns
    -------
    collated_examples : dict of {str: any}
        A dictionary containing images and labels in *torch* format.
    """
    images = []
    labels = []
    for example in examples:
        images.append((example["pixel_values"]))
        labels.append(example["label"])

    pixel_values = torch.stack(images)
    labels = torch.tensor(labels)

    collated_examples = {"pixel_values": pixel_values, "labels": labels}

    return collated_examples
