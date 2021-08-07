import torch
from torch_particle.utils.chart_utils import num_dense_nodes


def helper_compute_expected_dense_nodes(n):
    out = 0
    for i in range(1, n + 1):
        out += i
    return out


def test_chart_utils():
    for i in range(1, 10):
        expected = helper_compute_expected_dense_nodes(i)
        actual = num_dense_nodes(i)
        assert expected == actual, (i, expected, actual)

