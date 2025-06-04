import numpy
import torch.utils.data
import os
import pickle


class ImageDataset(torch.utils.data.Dataset):
    NO_VALUE = 0
    original_h = None
    original_w = None
    _num_data_files = None

    def __init__(self, root: str, num_repeat: int):
        self._num_repeat = num_repeat  # for data augmentation. Repeating the data sequence.
        path_to_datalist = os.path.join(root, "../", "datalist.txt")
        self._root_path = root
        with open(path_to_datalist, "r") as datalist_file:
            self._data_files_list = datalist_file.readlines()
            self._data_files_list = [data_file.strip() for data_file in self._data_files_list]
            self._data_files_list.sort()
        assert len(self._data_files_list) > 0
        self._num_data_files = len(self._data_files_list)
        # get original image size
        sample_image = self.__getitem__(0)
        self.original_h, self.original_w = sample_image.shape[:2]

    def __len__(self):
        return self._num_data_files * self._num_repeat  # Note: data augmentation by random crop
    
    @property
    def original_h(self):
        return self._original_h
    
    @original_h.setter
    def original_h(self, value):
        self._original_h = value
    
    @property
    def original_w(self):
        return self._original_w
    
    @original_w.setter
    def original_w(self, value):
        self._original_w = value

    def __getitem__(self, indexData: int):
        indexData = indexData % self._num_data_files
        file_path = os.path.join(self._root_path, self._data_files_list[indexData] + ".pkl")
        try:
            with open(file_path, 'rb') as file:
                data = pickle.load(file)
        except:
            raise FileNotFoundError(f"The data file {file_path} does not exist.")
        data['normals'][numpy.isnan(data['normals'])] = self.NO_VALUE
        # imagedata = numpy.concatenate([data['image'].astype('float32') / 255.0, data['normals'][..., -1][..., None].astype('float32')], axis=-1)  # Shape (H, W, 2)
        imagedata = data['image'].astype('float32') / 255.0
        return imagedata

    def collate_fn(self, batch: list):
        """
        batch is a list of dict.
        """
        batch = [one_image.permute(2, 0, 1).unsqueeze(0) for one_image in batch]  # Shape (C, H, W)
        return torch.concat(batch, dim=0)
