import torch
import torchvision.transforms.functional as F
from torchvision import transforms


class FixedHeightResize(object):
    def __init__(self, height):
        self.height = height

    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            img = transforms.ToPILImage()(img)
        size = (self.height, self._calc_new_width(img))
        return F.resize(img, size)

    def _calc_new_width(self, img):
        old_width, old_height = img.size
        aspect_ratio = old_width / old_height
        return round(self.height * aspect_ratio)
