import torch
from torch_particle.utils.chart_utils import num_dense_nodes


class DenseChart(object):
    r"""Container for chart data structure. Assumes dense connectivity.

    Args:
        x (Tensor): Leaf features.

    """
    def __init__(self, x):
        super().__init__()

        self.node_feat = self.init_chart(x)
        self.length = x.shape[0]

    def init_chart(self, x):
        num_leaves, size = x.shape
        num_nodes = num_dense_nodes(num_leaves)
        node_feat = torch.zeros(num_nodes, size)
        node_feat[:num_leaves] = x
        return node_feat


class ChartBatch(object):
    r"""Container for batch of charts.

    """

    @classmethod
    def from_chart_list(cls, chart_list):
        r"""Create ChartBatch from list of charts.

        """
        batch = cls()
        batch.chart_list = chart_list
        batch.node_feat = torch.cat([el.node_feat for el in chart_list], 0)
        batch.batch_size = len(chart_list)
        batch.lengths = torch.tensor([el.length for el in chart_list], dtype=torch.long)
        batch.chart_sizes = torch.tensor([el.node_feat.shape for el in chart_list], dtype=torch.long)
        batch.offsets = batch.chart_sizes.cumsum(0) - batch.chart_sizes[0]
        return batch
