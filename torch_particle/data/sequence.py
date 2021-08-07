import torch


class Sequence(object):
    r"""Container for sequences.

    Args:
        x (list or Tensor): Tokens.

    """
    def __init__(self, x):
        super().__init__()

        if isinstance(x, list):
            x = torch.tensor(x, dtype=torch.long)
        assert isinstance(x, torch.Tensor)
        assert len(x.shape) == 1 or len(x.shape) == 2
        self.x = x


class SequenceBatch(object):
    r"""Container for batch of sequences.

    """

    @classmethod
    def from_sequence_list(cls, sequence_list):
        r"""Create SequenceBatch from list of sequences.

        """
        batch = cls()
        batch.x = torch.cat([el.x for el in sequence_list], 0)
        batch.batch_size = len(sequence_list)
        batch.lengths = torch.tensor([el.x.shape[0] for el in sequence_list], dtype=torch.long)
        batch.offsets = torch.tensor([0] + batch.lengths[:-1].cumsum(0).tolist(), dtype=torch.long)
        return batch

    def to_tensor_list(self):
        return [self.x[o:o+n] for o, n in zip(self.offsets, self.lengths)]

    def to_padded_tensor(self):
        batch_size = self.batch_size
        max_length = max(self.lengths)
        output = torch.zeros(batch_size, max_length, dtype=self.x.dtype)

        for i_b, seq in enumerate(self.to_tensor_list()):
            output[i_b, :seq.shape[0]] = seq

        return output

    def new_from_padded_tensor(self, padded_tensor):
        batch_size, padded_length, size = padded_tensor.shape

        assert batch_size == self.batch_size
        assert padded_length == self.lengths.max()

        new_x = torch.zeros(self.lengths.sum(), size)

        for i_b, (o, n) in enumerate(zip(self.offsets, self.lengths)):
            new_x[o:o+n] = padded_tensor[i_b, :n]

        batch = SequenceBatch()
        batch.x = new_x
        batch.batch_size = self.batch_size
        batch.lengths = self.lengths
        batch.offsets = self.offsets

        return batch
