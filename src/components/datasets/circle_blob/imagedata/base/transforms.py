import torch
import numpy
import cv2


class ToTensor:
    def __call__(self, sample):
        sample = torch.from_numpy(sample)

        return sample


class Crop:
    def __init__(self, crop_height, crop_width, no_value):
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.no_value = no_value

    def __call__(self, sample, offset_x, offset_y):
        start_y, end_y = offset_y, offset_y + self.crop_height
        start_x, end_x = offset_x, offset_x + self.crop_width
        h_orig, w_orig = sample.shape[:2]
        # print("h_orig: {}, w_orig: {}, start_x: {}, start_y: {}".format(h_orig, w_orig, start_x, start_y))
        if start_y < 0:
            sample = numpy.pad(
                sample,
                ((-start_y, offset_y + self.crop_height - h_orig), (0, 0), (0, 0)),
                mode="constant",
                constant_values=self.no_value,
            )
            if start_x >= 0:
                sample = sample[:, start_x:end_x]
        if start_x < 0:
            sample = numpy.pad(
                sample,
                ((0, 0), (-start_x, offset_x + self.crop_width - w_orig), (0, 0)),
                mode="constant",
                constant_values=self.no_value,
            )
            if start_y >= 0:
                sample = sample[start_y:end_y, :]
        if start_x >= 0 and start_y >=0:
            sample = sample[start_y:end_y, start_x:end_x]
        # print("sample shape: {}".format(sample.shape))

        return sample


class VerticalFlip:
    def __call__(self, sample):
        sample = numpy.copy(numpy.flipud(sample))

        return sample


class HorizontalFlip:
    def __call__(self, sample):
        sample = numpy.copy(numpy.fliplr(sample))

        return sample


class Padding:
    def __init__(self, img_height, img_width, no_value):
        self.img_height = img_height
        self.img_width = img_width
        self.no_value = no_value

    def __call__(self, sample):
        ori_height, ori_width = sample.shape[:2]
        bottom_pad = self.img_height - ori_height
        right_pad = self.img_width - ori_width
        if bottom_pad < 0:
            sample = sample[:bottom_pad, :]
        else:
            sample = numpy.pad(
                sample,
                ((0, bottom_pad), (0, 0), (0, 0)),
                mode="constant",
                constant_values=self.no_value,
            )
        if right_pad < 0:
            sample = sample[:, :right_pad]
        else:
            sample = numpy.pad(
                sample,
                ((0, 0), (0, right_pad), (0, 0)),
                mode="constant",
                constant_values=self.no_value,
            )

        return sample


class Resize:
    def __init__(self, downsample_ratio):
        self.downsample_ratio = downsample_ratio
    
    def __call__(self, sample):
        new_height = int(sample.shape[0] / self.downsample_ratio)
        new_width = int(sample.shape[1] / self.downsample_ratio)
        sample = (sample.squeeze() * 255).astype("uint8")
        try:
            sample = cv2.resize(sample, (new_width, new_height), interpolation=cv2.INTER_AREA)
        except:
            print("sample: {}".format(sample.dtype))
            print("sample max: {}".format(sample.max()))
            print("sample min: {}".format(sample.min()))
            print("sample size: {}".format(sample.shape))
            print("new_height: {}, new_width: {}".format(new_height, new_width))
            raise
        sample = sample[:, :, None].astype("float32") / 255.0
        return sample
