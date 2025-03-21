import numpy as np


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample


class ToTensor:
    def __init__(self, event_module, disparity_module=None, objdet_module=None):
        self.event_transform = event_module.transforms.ToTensor()
        if disparity_module is not None:
            self.disparity_transform = disparity_module.transforms.ToTensor()
        if objdet_module is not None:
            self.objdet_transform = objdet_module.transforms.ToTensor()

    def __call__(self, sample):
        if "event" in sample.keys():
            sample["event"] = self.event_transform(sample["event"])

        if "disparity" in sample.keys():
            sample["disparity"] = self.disparity_transform(sample["disparity"])

        if "objdet" in sample.keys():
            sample["objdet"] = self.objdet_transform(sample["objdet"])

        return sample


class RandomCrop:
    def __init__(self, event_module, crop_height, crop_width, disparity_module=None):
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.event_transform = event_module.transforms.Crop(crop_height, crop_width)
        if disparity_module is not None:
            self.disparity_transform = disparity_module.transforms.Crop(
                crop_height, crop_width
            )

    def __call__(self, sample):
        if "event" in sample:
            ori_height, ori_width = sample["event"]["left"].shape[:2]
        else:
            raise NotImplementedError

        assert self.crop_height <= ori_height and self.crop_width <= ori_width

        offset_x = np.random.randint(ori_width - self.crop_width + 1)
        offset_y = np.random.randint(ori_height - self.crop_height + 1)

        if "event" in sample.keys():
            sample["event"] = self.event_transform(sample["event"], offset_x, offset_y)

        if "disparity" in sample.keys():
            sample["disparity"] = self.disparity_transform(
                sample["disparity"], offset_x, offset_y
            )

        return sample


class Padding:
    def __init__(
        self,
        img_height,
        img_width,
        event_module,
        no_event_value=0,
        no_disparity_value=0,
        disparity_module=None,
        no_objdet_value=None,
        objdet_module=None
    ):
        self.img_height = img_height
        self.img_width = img_width
        self.event_transform = event_module.transforms.Padding(
            img_height, img_width, no_event_value
        )
        if disparity_module is not None:
            self.disparity_transform = disparity_module.transforms.Padding(
                img_height, img_width, no_disparity_value
            )
        if objdet_module is not None:
            self.objdet_transform = objdet_module.transforms.Padding(img_height, img_width, no_objdet_value)

    def __call__(self, sample):
        if "event" in sample.keys():
            sample["event"] = self.event_transform(sample["event"])

        if "disparity" in sample.keys():
            sample["disparity"] = self.disparity_transform(sample["disparity"])

        if "objdet" in sample.keys():
            sample["objdet"] = self.objdet_transform(sample["objdet"])

        return sample


class RandomVerticalFlip:
    def __init__(
        self,
        event_module,
        disparity_module=None,
        objdet_module=None,
        img_height=None,
        img_width=None,
    ):
        self.event_transform = event_module.transforms.VerticalFlip()
        if disparity_module is not None:
            self.disparity_transform = disparity_module.transforms.VerticalFlip()
        if objdet_module is not None:
            self.objdet_transform = objdet_module.transforms.VerticalFlip(
                img_height, img_width
            )

    def __call__(self, sample):
        if np.random.random() < 0.5:
            if "event" in sample.keys():
                sample["event"] = self.event_transform(sample["event"])

            if "disparity" in sample.keys():
                sample["disparity"] = self.disparity_transform(sample["disparity"])

            if "objdet" in sample.keys():
                sample["objdet"] = self.objdet_transform(sample["objdet"])

        return sample


class RandomHorizontalFlip:
    def __init__(
        self,
        event_module,
        disparity_module=None,
        objdet_module=None,
        img_height=None,
        img_width=None,
    ):
        self.event_transform = event_module.transforms.HorizontalFlip()
        if disparity_module is not None:
            self.disparity_transform = disparity_module.transforms.HorizontalFlip()
        if objdet_module is not None:
            self.objdet_transform = objdet_module.transforms.HorizontalFlip(
                img_height, img_width
            )

    def __call__(self, sample):
        if np.random.random() < 0.5:
            if "event" in sample.keys():
                sample["event"] = self.event_transform(sample["event"])

            if "disparity" in sample.keys():
                sample["disparity"] = self.disparity_transform(sample["disparity"])

            if "objdet" in sample.keys():
                sample["objdet"] = self.objdet_transform(sample["objdet"])

        return sample


class ConvertBboxes:
    def __init__(
        self,
        img_height,
        img_width,
        objdet_module=None
    ):
        self.bboxes_transform = objdet_module.transforms.ConvertBboxes(img_height, img_width)

    def __call__(self, sample):
        sample["objdet"] = self.bboxes_transform(sample["objdet"])
        return sample
