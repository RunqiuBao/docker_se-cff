import numpy as np


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample


class ToTensor:
    def __init__(self, imagedata_module, objdet_module):
        self.imagedata_module = imagedata_module.transforms.ToTensor()
        self.objdet_transform = objdet_module.transforms.ToTensor()

    def __call__(self, sample):
        if "imagedata" in sample.keys():
            sample["imagedata"] = self.imagedata_module(sample["imagedata"])

        if "objdet" in sample.keys():
            sample["objdet"] = self.objdet_transform(sample["objdet"])
        return sample


class RandomCrop:
    def __init__(self, imagedata_module, objdet_module, crop_height, crop_width, no_value):
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.imagedata_transform = imagedata_module.transforms.Crop(crop_height, crop_width, no_value)
        self.objdet_transform = objdet_module.transforms.Crop()

    def __call__(self, sample):
        ori_height, ori_width = sample["imagedata"].shape[:2]

        offset_x = np.random.randint(ori_width - self.crop_width + 1) if (ori_width - self.crop_width) >= 0 else np.random.randint(ori_width - self.crop_width, 0)
        offset_y = np.random.randint(ori_height - self.crop_height + 1) if (ori_height - self.crop_height) >= 0 else np.random.randint(ori_width - self.crop_width, 0)

        sample["imagedata"] = self.imagedata_transform(sample["imagedata"], offset_x, offset_y)
        sample["objdet"] = self.objdet_transform(sample["objdet"], offset_x, offset_y)

        return sample


class Padding:
    def __init__(
        self,
        imagedata_module,
        img_height,
        img_width,
        no_value=0,
    ):
        self.img_height = img_height
        self.img_width = img_width
        self.imagedata_transform = imagedata_module.transforms.Padding(
            img_height,
            img_width,
            no_value
        )

    def __call__(self, sample):
        sample["imagedata"] = self.imagedata_transform(sample["imagedata"])
        return sample


class RandomVerticalFlip:
    def __init__(
        self,
        imagedata_module,
        objdet_module,
        img_height,
        img_width,
    ):
        self.imagedata_transform = imagedata_module.transforms.VerticalFlip()
        self.objdet_transform = objdet_module.transforms.VerticalFlip(img_height, img_width)

    def __call__(self, sample):
        if np.random.random() < 0.5:
            sample["imagedata"] = self.imagedata_transform(sample["imagedata"])
            sample["objdet"] = self.objdet_transform(sample["objdet"])

        return sample


class RandomHorizontalFlip:
    def __init__(
        self,
        imagedata_module,
        objdet_module,
        img_height,
        img_width,
    ):
        self.imagedata_transform = imagedata_module.transforms.HorizontalFlip()
        self.objdet_transform = objdet_module.transforms.HorizontalFlip(img_height, img_width)

    def __call__(self, sample):
        if np.random.random() < 0.5:
            sample["imagedata"] = self.imagedata_transform(sample["imagedata"])
            sample["objdet"] = self.objdet_transform(sample["objdet"])

        return sample


class Resize:
    def __init__(
        self,
        imagedata_module,
        objdet_module,
        downsample_ratio
    ):
        self.imagedata_transform = imagedata_module.transforms.Resize(downsample_ratio)
        self.objdet_transform = objdet_module.transforms.Resize(downsample_ratio)
    
    def __call__(self, sample):
        sample["imagedata"] = self.imagedata_transform(sample["imagedata"])
        sample["objdet"] = self.objdet_transform(sample["objdet"])
        return sample
