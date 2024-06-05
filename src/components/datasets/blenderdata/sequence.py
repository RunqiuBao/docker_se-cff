import os
import csv
import copy

import numpy as np

import torch.utils.data

from . import disparity
from . import event
from . import transforms


class SequenceDataset(torch.utils.data.Dataset):
    timestamps = None
    event_dataset = None
    disparity_dataset = None
    _PATH_DICT = {
        'event': 'events',
        'disparity': 'disparity',
        'timestamps': 'timestamps.txt',
    }

    def __init__(self, root, split, sampling_ratio, event_cfg, 
                 crop_height, crop_width, num_workers=0, **kwargs):
        self.root = root
        self.split = split
        self.sampling_ratio = sampling_ratio
        self.event_cfg = event_cfg
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.num_workers = num_workers

        self.sequence_name = root.split('/')[-1]

        # Event Dataset
        event_module = getattr(event, event_cfg.NAME)
        event_root = os.path.join(root, self._PATH_DICT['event'])
        self.event_dataset = event_module.EventDataset(root=event_root, **event_cfg.PARAMS)        
        
        # Timestamps
        if split in ['train', 'validation', 'trainval']:
            self.timestamps = np.loadtxt(os.path.join(root, self._PATH_DICT['timestamps']), dtype='int64')

            if event_cfg.NAME != 'none':
                minimum_timestamp = max(self.event_dataset.event_slicer['left'].t_offset,
                                        self.event_dataset.event_slicer['right'].t_offset)
                maximum_timestamp = None
                if event_cfg.NAME == 'base':
                    minimum_timestamp += self.event_dataset.time_interval
                elif event_cfg.NAME == 'sbn':
                    minimum_timestamp = max(self.event_dataset.event_slicer['left'].min_time,
                                            self.event_dataset.event_slicer['right'].min_time,)
                    maximum_timestamp = min(self.event_dataset.event_slicer['left'].max_time,
                                            self.event_dataset.event_slicer['right'].max_time)
                self.timestamps = self.timestamps[self.timestamps >= minimum_timestamp]
                if maximum_timestamp is not None:
                    self.timestamps = self.timestamps[self.timestamps <= maximum_timestamp]
            self.timestamp_to_index = {
                timestamp: idx
                for idx, timestamp in enumerate(self.timestamps)
            }
        elif split in ['test']:
            self.timestamps, self.timestamp_to_index = read_csv(os.path.join(root, self.sequence_name + '.csv'))
        else:
            raise NotImplementedError

        self.timestamps = self.timestamps[[idx for idx in range(0, len(self.timestamps), sampling_ratio)]]

        # Transforms
        if split in ['train', 'trainval']:
            transformsList = [
                transforms.RandomCrop(event_module=event_module,
                                      crop_height=crop_height, crop_width=crop_width),
                transforms.ToTensor(event_module=event_module),
            ]
            if kwargs.get('randomverticalflip', False):
                transformsList.append(transforms.RandomVerticalFlip(event_module=event_module))
            self.transforms = transforms.Compose(transformsList)
        elif split in ['validation', 'test']:
            self.transforms = transforms.Compose([
                transforms.Padding(event_module=event_module,
                                   img_height=crop_height, img_width=crop_width,
                                   no_event_value=self.event_dataset.NO_VALUE,
                                   no_disparity_value=self.disparity_dataset.NO_VALUE),
                transforms.ToTensor(event_module=event_module),
            ])
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.timestamps)

    def __getitem__(self, idx):
        data = self.load_data(idx)

        data = self.transforms(data)

        return data

    def collate_fn(self, batch):
        output = {}
        # Event
        domain = 'event'
        if domain in batch[0].keys():
            output[domain] = self.event_dataset.collate_fn([sample[domain] for sample in batch])

        # Others
        for key in batch[0].keys():
            if key not in ['event']:
                output[key] = torch.utils.data._utils.collate.default_collate([sample[key] for sample in batch])

        return output

    def load_data(self, idx):
        timestamp = self.timestamps[idx]
        data = {}
        event_data = self.event_dataset[timestamp]

        data['file_index'] = self.timestamp_to_index[timestamp]
        data['end_timestamp'] = timestamp
        if event_data is not None:
            data['event'] = event_data

        return data


def read_csv(csv_file):
    timestamps = []
    timestamp_to_index = {}
    with open(csv_file) as csvfile:
        data_reader = csv.reader(csvfile)
        for row in data_reader:
            assert row[0] not in timestamps
            if row[0].isnumeric():
                timestamps.append(int(row[0]))
                timestamp_to_index[int(row[0])] = int(row[1])

    return np.asarray(timestamps), timestamp_to_index
