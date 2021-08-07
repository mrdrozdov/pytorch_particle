import torch
from torch_particle.data.sequence import Sequence, SequenceBatch


class Collater(object):
    def collate(self, batch):
        elem = batch[0]
        if isinstance(elem, Sequence):
            return SequenceBatch.from_sequence_list(batch)

        raise TypeError('DataLoader found invalid type: {}'.format(type(elem)))

    def __call__(self, batch):
        return self.collate(batch)


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, **kwargs):

        if "collate_fn" in kwargs:
            del kwargs["collate_fn"]

        super().__init__(dataset, batch_size, shuffle,
                         collate_fn=Collater(), **kwargs)
