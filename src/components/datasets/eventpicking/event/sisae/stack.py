import torch
import numpy as np
import time


class SpeedInvariantSurfaceOfActiveEvents:
    NO_VALUE = 0.0
    STACK_LIST = ["stacked_polarity", "index"]

    def __init__(self, stack_size, num_of_event, height, width):
        self.stack_size = stack_size
        self.num_of_event = num_of_event
        self.height = height
        self.width = width

    def do_stack(self, event_sequence):
        rr = 6

        sae = np.zeros((self.height, self.width, 2), np.float32)
        # calculate timesurface using expotential decay
        x, y, ts, p = event_sequence['x'], event_sequence['y'], event_sequence['t'], event_sequence['p']
        starttime = time.time()
        for i in range(len(event_sequence['t'])):
            if y[i] <= rr or y[i] >= self.height - rr or x[i] <= rr or x[i] >= self.width - rr:
                continue
            dx = np.arange(-rr, rr + 1)
            dy = np.arange(-rr, rr + 1)
            dx, dy = np.meshgrid(dx, dy)
            if p[i] > 0:
                np.add.at(sae[:, :, 0], (y[i] + dy, x[i] + dx), -1)
                sae[y[i], x[i], 0] += (2 * rr + 1)**2
            else:
                np.add.at(sae[:, :, 1], (y[i] + dy, x[i] + dx), -1)            
                sae[y[i], x[i], 1] += (2 * rr + 1)**2
        np.clip(sae, 0, 255.0, out=sae)
        sae = sae[..., 0] - sae[..., 1]
        print("one sisae time: {}".format(time.time() - starttime))

        if self.stack_size == 1:
            return np.expand_dims(sae, axis=-1)
        else:
            return np.repeat(sae[:, :, np.newaxis], repeats=self.stack_size, axis=-1)

    @staticmethod
    def collate_fn(batch):
        batch = torch.utils.data._utils.collate.default_collate(batch)

        return batch
