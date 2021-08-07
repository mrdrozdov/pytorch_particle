import torch
from torch_particle.utils.chart_utils import num_dense_nodes
from torch_particle.data.chart import DenseChart, ChartBatch


def test_chart():
    lengths = [3, 7, 4]
    size = 10
    chart_list = [DenseChart(torch.randn(n, size)) for n in lengths]
    batch = ChartBatch.from_chart_list(chart_list)
    assert all([na == nb for na, nb in zip(lengths, batch.lengths.tolist())])
    assert batch.node_feat.shape == (sum([num_dense_nodes(n) for n in lengths]), size)
