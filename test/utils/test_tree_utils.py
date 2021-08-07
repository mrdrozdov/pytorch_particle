from torch_particle.utils.tree_utils import tree_from_string, tree_to_leaves


def test_tree_utils():
    tree = tree_from_string('(S (A B) (C D))', add_space=True)
    assert tree == ('S', (('A', 'B'), ('C', 'D')))

    leaves = tree_to_leaves(tree)
    assert tuple(leaves) == ('B', 'D')
