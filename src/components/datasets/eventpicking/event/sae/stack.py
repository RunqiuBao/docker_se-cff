import torch
import numpy as np


class SurfaceOfActiveEvents:
    NO_VALUE = 0.0
    STACK_LIST = ["stacked_polarity", "index"]

    def __init__(self, stack_size, num_of_event, height, width, tau):
        self.stack_size = stack_size
        self.num_of_event = num_of_event
        self.height = height
        self.width = width
        self.tau = tau

    def do_stack(self, event_sequence):
        t_ref = event_sequence['t'][-1]

        sae = np.zeros((self.height, self.width), np.float32)
        # calculate timesurface using expotential decay
        psign = np.where(event_sequence['p'] <= 0, -1, 1)
        np.add.at(sae, (event_sequence['y'], event_sequence['x']), np.sign(psign) * np.exp(-(t_ref - event_sequence['t']).astype('float') / self.tau))

        if self.stack_size == 1:
            return np.expand_dims(sae, axis=-1)
        else:
            return np.repeat(sae[:, :, np.newaxis], repeats=self.stack_size, axis=-1)

    @staticmethod
    def collate_fn(batch):
        batch = torch.utils.data._utils.collate.default_collate(batch)

        return batch
