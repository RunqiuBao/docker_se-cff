import os
import numpy as np
import torch.utils.data

from .slice import EventSlicer
from . import stack, constant


class EventDataset(torch.utils.data.Dataset):
    _PATH_DICT = {
        'timestamp': 'timestamps.txt',
        'left': 'left',
        'right': 'right'
    }
    _LOCATION = ['left', 'right']
    NO_VALUE = None
    lmdb_txn = None
    lmdb_env = None
    sequence_name = None
    sequence_length = None

    # image meta data
    event_h = None
    event_w = None
    event_channels = None

    def __init__(
            self,
            root,
            num_of_event,
            stack_method,
            stack_size,
            num_of_future_event=0,
            use_preprocessed_image=False,
            sequence_name='0',
            sequence_length=0,
            lmdb_txn=None,
            **kwargs):
        self.root = root
        self.num_of_event = num_of_event
        self.stack_method = stack_method
        self.stack_size = stack_size
        self.num_of_future_event = num_of_future_event
        self.use_preprocessed_image = use_preprocessed_image
        self.event_h = constant.EVENT_HEIGHT
        self.event_w = constant.EVENT_WIDTH
        self.event_channels = constant.EVENT_CHANNELS
        self.sequence_length = sequence_length

        # moving events into lmdb
        if lmdb_txn is not None:
            self.lmdb_txn = lmdb_txn
            self.sequence_name = sequence_name

        self.event_slicer = {}
        for location in self._LOCATION:
            event_path = os.path.join(root, location, 'events.h5')
            rectify_map_path = os.path.join(root, location, 'rectify_map.h5')
            self.event_slicer[location] = EventSlicer(event_path, rectify_map_path, num_of_event, num_of_future_event)

        self.stack_function = getattr(stack, stack_method)(stack_size, num_of_event,
                                                           constant.EVENT_HEIGHT, constant.EVENT_WIDTH, **kwargs)
        self.NO_VALUE = self.stack_function.NO_VALUE

    def __len__(self):
        return self.sequence_length

    def __getitem__(self, x):
        idx, timestamp = x
        if self.lmdb_txn is not None:
            code = '%03d_%06d_l' % (int(self.sequence_name.split('seq')[-1]), idx)
            code = code.encode()
            left_events = self.lmdb_txn.get(code)
            left_events = np.frombuffer(left_events, dtype='int8')
            left_events = left_events.reshape(constant.EVENT_HEIGHT, constant.EVENT_WIDTH, constant.EVENT_CHANNELS).transpose(2, 0, 1)
            code = '%03d_%06d_r' % (int(self.sequence_name.split('seq')[-1]), idx)
            code = code.encode()
            right_events = self.lmdb_txn.get(code)
            right_events = np.frombuffer(right_events, dtype='int8')
            right_events = right_events.reshape(constant.EVENT_HEIGHT, constant.EVENT_WIDTH, constant.EVENT_CHANNELS).transpose(2, 0, 1)
            event_data = {
                'left': left_events,
                'right': right_events
            }
        else:
            event_data = self._pre_load_event_data(timestamp=timestamp)
            event_data = self._post_load_event_data(event_data)
            for key, value in event_data.items():
                event_data[key] = value.squeeze().transpose(2, 0, 1)
        return event_data

    def _pre_load_event_data(self, timestamp):
        event_data = {}

        minimum_time, maximum_time = -float('inf'), float('inf')
        for location in self._LOCATION:
            event_data[location] = self.event_slicer[location][timestamp]
            minimum_time = max(minimum_time, event_data[location]['t'].min())
            maximum_time = min(maximum_time, event_data[location]['t'].max())

        for location in self._LOCATION:
            mask = np.logical_and(minimum_time <= event_data[location]['t'], event_data[location]['t'] <= maximum_time)
            for data_type in ['x', 'y', 't', 'p']:
                event_data[location][data_type] = event_data[location][data_type][mask]

        for location in self._LOCATION:
            event_data[location] = self.stack_function.pre_stack(event_data[location], timestamp)

        return event_data

    def _post_load_event_data(self, event_data):
        for location in self._LOCATION:
            event_data[location] = self.stack_function.post_stack(event_data[location])

        return event_data

    @staticmethod
    def collate_fn(batch):
        batch = torch.utils.data._utils.collate.default_collate(batch)

        return batch

    # def collate_fn(self, batch):
    #     batch = self.stack_function.collate_fn(batch)

    #     return batch
