import os
import lmdb

import torch.utils.data
import torch.utils.data._utils

from torch.utils.data.distributed import DistributedSampler
from utils.dataloader import MultiEpochsDataLoader
from .sequence import SequenceDataset
from .constant import DATA_SPLIT


class BinpickingDataset(torch.utils.data.Dataset):
    lmdb_txn = None
    lmdb_env = None

    def __init__(
        self,
        root,
        split,
        sampling_ratio,
        event_cfg,
        crop_height,
        crop_width,
        num_workers=0,
        defineSeqIdx=None,
        isDisableLmdbRead=False,
        **kwargs,
    ):
        self.root = root
        self.split = split
        self.sampling_ratio = sampling_ratio
        self.event_cfg = event_cfg
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.num_workers = num_workers
        assert split in DATA_SPLIT.keys()

        # moving events into lmdb
        if not isDisableLmdbRead and os.path.isdir(os.path.join(root, split, "lmdb")) and os.path.getsize(os.path.join(root, split, "lmdb", "data.mdb")) > 1024**2:  # Note: the dataset need to be larger than 1MB
            self.lmdb_env = lmdb.open(
                os.path.join(root, split, "lmdb"), map_size=1099511627776
            )
            self.lmdb_txn = self.lmdb_env.begin()
        else:
            print("baodebug: ==========================: skip lmdb due to isDisableLmdbRead {} and isDir", str(isDisableLmdbRead), str(os.path.isdir(os.path.join(root, split, "lmdb"))))

        sequence_list = DATA_SPLIT[split]
        if defineSeqIdx is not None:
            sequence_list = [sequence_list[defineSeqIdx]]
            
        self.sequence_data_list = []
        for sequence in sequence_list:
            sequence_root = os.path.join(root, sequence)            
            self.sequence_data_list.append(
                SequenceDataset(
                    root=sequence_root,
                    split=split,
                    sampling_ratio=sampling_ratio,
                    event_cfg=event_cfg,
                    crop_height=crop_height,
                    crop_width=crop_width,
                    num_workers=num_workers,
                    lmdb_txn=self.lmdb_txn,
                    **kwargs,
                )
            )

        if len(self.sequence_data_list) == 0:
            self.dataset = []
        else:
            self.dataset = torch.utils.data.ConcatDataset(self.sequence_data_list)            

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]

        return data

    def collate_fn(self, batch):
        return self.dataset.datasets[0].collate_fn(batch)


def event_collate_fn(batch):
    def _up_size(tensor_data, size, fill_value=0):
        output_data = torch.ones([size], dtype=tensor_data.dtype) * fill_value
        output_data[: tensor_data.size(0)] = tensor_data
        return output_data

    max_event_length = max(
        [
            max(sample["event"]["left"]["length"], sample["event"]["right"]["length"])
            for sample in batch
        ]
    )
    for sample in batch:
        for location in ["left", "right"]:
            for key in ["x", "y", "p", "t_q", "t_s", "count_index"]:
                sample["event"][location][key] = _up_size(
                    sample["event"][location][key], max_event_length, -1
                )

    batch = torch.utils.data._utils.collate.default_collate(batch)

    return batch


def get_multi_epochs_dataloader(
    dataset, dataloader_cfg, num_workers, is_distributed, world_size
):
    if len(dataset) == 0:
        return torch.utils.data.DataLoader(dataset)

    if is_distributed:
        batch_size = dataloader_cfg.PARAMS.batch_size // world_size  # Note: keep the effective batch_size the same, so that to keep the same learning rate.
        shuffle = dataloader_cfg.PARAMS.get("shuffle", False)
        drop_last = dataloader_cfg.PARAMS.get("drop_last", False)
        sampler = DistributedSampler(dataset, shuffle=shuffle, drop_last=drop_last)
        multi_epochs_dataloader = MultiEpochsDataLoader(
            dataset=dataset,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=dataset.collate_fn,
            batch_size=batch_size,
            drop_last=drop_last,
            sampler=sampler,
        )
    else:
        multi_epochs_dataloader = MultiEpochsDataLoader(
            dataset=dataset,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=dataset.collate_fn,
            **dataloader_cfg.PARAMS,
        )

    return multi_epochs_dataloader


def get_sequence_dataloader(
    dataset, dataloader_cfg, num_workers, is_distributed, world_size
):
    if len(dataset) == 0:
        return torch.utils.data.DataLoader(dataset)
    if is_distributed:
        batch_size = dataloader_cfg.PARAMS.batch_size // world_size
        shuffle = dataloader_cfg.PARAMS.get("shuffle", False)
        drop_last = dataloader_cfg.PARAMS.get("drop_last", False)
        sampler = DistributedSampler(dataset, shuffle=shuffle, drop_last=drop_last)
        sequence_dataloader = [
            torch.utils.data.DataLoader(
                dataset=sequence_dataset,
                num_workers=num_workers,
                pin_memory=True,
                collate_fn=dataset.collate_fn,
                batch_size=batch_size,
                drop_last=drop_last,
                sampler=sampler,
            )
            for sequence_dataset in dataset.sequence_data_list
        ]
    else:
        sequence_dataloader = [
            torch.utils.data.DataLoader(
                dataset=sequence_dataset,
                num_workers=num_workers,
                pin_memory=False,
                collate_fn=dataset.collate_fn,
                **dataloader_cfg.PARAMS,
            )
            for sequence_dataset in dataset.sequence_data_list
        ]

    return sequence_dataloader


def get_dataloader(
    args, dataset_cfg, dataloader_cfg, is_distributed=False, defineSeqIdx=None, isDisableLmdbRead=False
):
    """
    Args:
        ...
        defineSeqIdx: int, only use this sequence if defined.
    """
    dataset = BinpickingDataset(
        root=args.data_root,
        num_workers=args.num_workers,
        defineSeqIdx=defineSeqIdx,
        isDisableLmdbRead=isDisableLmdbRead,
        **dataset_cfg.PARAMS,
    )

    dataloader = globals()[dataloader_cfg.NAME](
        dataset=dataset,
        dataloader_cfg=dataloader_cfg,
        num_workers=args.num_workers,
        is_distributed=is_distributed,
        world_size=args.world_size if is_distributed else None,
    )

    return dataloader
