import torch
from torch_particle.utils.tree_utils import tree_to_leaves
from torch_particle.data.sequence import Sequence


class Dataset(torch.utils.data.Dataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def add_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, ix):
        leaves = tree_to_leaves(self.corpus[ix])
        return Sequence(self.tokenizer.indexify(leaves))
