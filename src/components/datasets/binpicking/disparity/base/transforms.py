import torch
import numpy as np


class ToTensor:
    def __call__(self, sample):
        sample = torch.from_numpy(sample)

        return sample


class Padding:
    def __init__(self, img_height, img_width, no_disparity_value):
        self.img_height = img_height
        self.img_width = img_width
        self.no_disparity_value = no_disparity_value

    def __call__(self, sample):
        ori_height, ori_width = sample.shape[-2:]
        bottom_pad = self.img_height - ori_height
        right_pad = self.img_width - ori_width

        assert bottom_pad >= 0 and right_pad >= 0

        sample = np.lib.pad(
            sample,
            ((0, bottom_pad), (0, right_pad)),
            mode="constant",
            constant_values=self.no_disparity_value,
        )

        return sample


class VerticalFlip:
    def __call__(self, sample):
        sample = np.copy(np.flipud(sample))

        return sample


class HorizontalFlip:
    def __call__(self, sample):
        """
        disparity sample is 2d.
        """
        sample = np.copy(np.fliplr(sample))

        return sample
