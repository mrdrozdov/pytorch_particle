import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from torch_geometric.data import Batch, Data
import torch_geometric.nn as gnn

from tqdm import tqdm

from torch_particle.datasets.craft import CRAFT
from torch_particle.data.dataloader import DataLoader
from torch_particle.data.tokenizer import Tokenizer
from torch_particle.utils.chart_utils import num_dense_nodes
from torch_particle.utils.tree_utils import tree_to_leaves


def get_sentences(craft_dataset):
    for tree in craft_dataset.corpus:
        yield tree_to_leaves(tree)


class GeometricChart(object):
    def __init__(self, x):
        super().__init__()


def chart_offset(chart_size, level_size):
    return num_dense_nodes(chart_size) - num_dense_nodes(level_size)


class Net(nn.Module):
    def __init__(self, embed):
        super().__init__()
        self.embed = embed

    def to_data(self, seq):
        length, size = seq.shape
        chart_size = num_dense_nodes(length)

        # Inside edge index.
        inside_edge_index = []
        for level in range(1, length):
            L = length - level
            N = level
            x_size = level + 1
            for x_pos in range(L):
                x_idx = chart_offset(chart_size, x_size - 1) + x_pos
                for n_idx in range(N):
                    l_pos = x_pos
                    l_size = n_idx + 1
                    l_idx = chart_offset(chart_size, l_size - 1) + l_pos
                    inside_edge_index.append([l_idx, x_idx])

                    r_pos = l_pos + l_size
                    r_size = x_size - l_size
                    r_idx = chart_offset(chart_size, r_size - 1) + r_pos
                    inside_edge_index.append([r_idx, x_idx])

        inside_edge_index = torch.tensor(inside_edge_index, dtype=torch.long)
        data = Data(edge_index=inside_edge_index.t().contiguous(), num_nodes=chart_size)
        import ipdb; ipdb.set_trace()
        pass

    def forward(self, batch):
        padded_batch = batch.to_padded_tensor()
        embed_batch = self.embed(padded_batch)

        data_list = [self.to_data(seq) for seq in batch.new_from_padded_tensor(embed_batch).to_tensor_list()]

        return embed_batch.sum()


def main(args):

    dataset = CRAFT(split='test')

    tokenizer = Tokenizer()
    tokenizer.add_sequence_list(get_sentences(dataset))
    tokenizer.finalize()

    dataset.tokenizer = tokenizer

    loader = DataLoader(dataset, batch_size=4)

    # Initialize model.
    num_embeddings = len(tokenizer.word_to_idx)
    embed_dim = 10
    embed = torch.nn.Embedding(num_embeddings, embed_dim, padding_idx=0)
    net = Net(embed)

    # Initiaze optimization.

    opt = optim.Adam(net.parameters())

    # Training and eval loop.
    for epoch in range(10):

        # Train.
        net.train()
        for batch in tqdm(loader, desc='train'):
            loss = net(batch)

            opt.zero_grad()
            loss.backward()
            opt.step()

        # Eval.
        net.eval()
        with torch.no_grad():
            for batch in tqdm(loader, desc='eval'):
                loss = net(batch)


def argument_parser():
    parser = argparse.ArgumentParser()
    return parser


if __name__ == '__main__':
    parser = argument_parser()

    args = parser.parse_args()

    main(args)
