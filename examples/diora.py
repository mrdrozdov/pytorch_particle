import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from torch_particle.datasets.craft import CRAFT
from torch_particle.data.chart import DenseChart, ChartBatch
from torch_particle.data.dataloader import DataLoader
from torch_particle.data.tokenizer import Tokenizer
from torch_particle.utils.tree_utils import tree_to_leaves


def get_sentences(craft_dataset):
    for tree in craft_dataset.corpus:
        yield tree_to_leaves(tree)


class Net(nn.Module):
    def __init__(self, embed):
        super().__init__()
        self.embed = embed

    def forward(self, batch):
        padded_batch = batch.to_padded_tensor()
        embed_batch = self.embed(padded_batch)
        chart_list = [DenseChart(seq) for seq in batch.new_from_padded_tensor(embed_batch).to_tensor_list()]
        chart_batch = ChartBatch.from_chart_list(chart_list)

        return chart_batch.node_feat.sum()


def main():
    # Initialize data.
    train_dataset = CRAFT(split='train')
    test_dataset = CRAFT(split='test')

    tokenizer = Tokenizer()
    tokenizer.add_sequence_list(get_sentences(train_dataset))
    tokenizer.add_sequence_list(get_sentences(test_dataset))
    tokenizer.finalize()

    train_dataset.tokenizer = tokenizer
    test_dataset.tokenizer = tokenizer

    train_loader = DataLoader(train_dataset, batch_size=4)
    test_loader = DataLoader(test_dataset, batch_size=4)

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
        for batch in tqdm(train_loader, desc='train'):
            loss = net(batch)

            opt.zero_grad()
            loss.backward()
            opt.step()

        # Eval.
        net.eval()
        with torch.no_grad():
            for batch in tqdm(test_loader, desc='eval'):
                loss = net(batch)


if __name__ == '__main__':
    main()
