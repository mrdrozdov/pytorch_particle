def tree_from_string(s, tokens=None, add_space=False):
    if tokens is not None:
        assert add_space is False
        return recursive_parse(tokens)[0]
    if add_space:
        s = s.replace('(', '( ').replace(')', ' )')
    return recursive_parse(s.split())[0]


def recursive_parse(tokens, pos=0):
    if tokens[pos + 2] != '(':
        label = tokens[pos + 1]
        leaf = tokens[pos + 2]
        size = 4
        node = (label, leaf)
        return node, size

    size = 2
    nodes = []

    while tokens[pos + size] != ')':
        xnode, xsize = recursive_parse(tokens, pos + size)
        size += xsize
        nodes.append(xnode)
    size += 1

    label = tokens[pos + 1]
    children = tuple(nodes)
    node = (label, children)
    return node, size


def print_tree(tree):
    def helper(tr):
        label, tr = tr
        if isinstance(tr, str):
            return '({} {})'.format(label, tr)
        nodes = [helper(x) for x in tr]
        return '({} {})'.format(label, ' '.join(nodes))
    return helper(tree)


def tree_to_leaves(tree):
    def helper(tr):
        label, tr = tr
        if isinstance(tr, str):
            return [tr]
        nodes = []
        for x in tr:
            nodes += helper(x)
        return nodes
    return helper(tree)