import os
import numpy
import torch.utils.data
import torch.utils.data._utils
from torch.utils.data.distributed import DistributedSampler

from utils.dataloader import MultiEpochsDataLoader
from .sequence import SequenceDataset
from .constant import DATA_SPLIT
from .imagedata.base.dataset import read_pkl_data, NO_VALUE


class MujinExtendedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        split,
        **kwargs,
    ):
        self.root = root
        self.split = split
        assert split in DATA_SPLIT.keys()
        sequence_list = DATA_SPLIT[split]

        self.sequence_data_list = []
        for sequence in sequence_list:
            sequence_root = os.path.join(root, sequence)            
            self.sequence_data_list.append(
                SequenceDataset(
                    root=sequence_root,
                    split=split,
                    **kwargs,
                )
            )

        if len(self.sequence_data_list) == 0:
            self.dataset = []
        else:
            self.dataset = torch.utils.data.ConcatDataset(self.sequence_data_list)
        
        # for random multi-scale training
        self._rfdetr_resolution = kwargs["resolution"]
        self._patch_size = kwargs["patch_size"]
        self._num_windows = kwargs["num_windows"]

    @property
    def rfdetr_resolution(self):
        return self._rfdetr_resolution

    @rfdetr_resolution.setter
    def rfdetr_resolution(self, value: float):
        self._rfdetr_resolution = value

    @property
    def rfdetr_patch_size(self):
        return self._patch_size

    @rfdetr_patch_size.setter
    def rfdetr_patch_size(self, value: float):
        self._patch_size = value

    @property
    def rfdetr_num_windows(self):
        return self._num_windows

    @rfdetr_num_windows.setter
    def rfdetr_num_windows(self, value: float):
        self._num_windows = value

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        return data

    def collate_fn(self, batch):
        return self.dataset.datasets[0].collate_fn(batch)


def get_multi_epochs_dataloader(
    dataset,
    dataloader_cfg,
    num_workers,
    is_distributed,
    world_size
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
    dataset,
    dataloader_cfg,
    num_workers,
    is_distributed,
    world_size
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
    args,
    dataset_cfg,
    dataloader_cfg,
    is_distributed=False,
    defineSeqIdx=None,
    isDisableLmdbRead=False
):
    """
    Args:
        ...
        defineSeqIdx: int, only use this sequence if defined.
    """
    dataset = MujinExtendedDataset(
        root=args.data_root,
        num_workers=args.num_workers,
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
