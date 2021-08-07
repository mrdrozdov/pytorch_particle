import os
from torch_particle.data.dataset import Dataset
from torch_particle.utils.tree_utils import print_tree, tree_from_string
from tqdm import tqdm


def remove_none(tree):
    def helper(tr):
        label, tr = tr
        if isinstance(tr, str):
            if label == '-NONE-':
                return None, False
            return '({} {})'.format(label, tr), True

        assert label != '-NONE-', tr

        nodes = []
        for x in tr:
            x_node, x_is_okay = helper(x)
            if not x_is_okay:
                continue
            nodes.append(x_node)

        if len(nodes) == 0:
            return None, False

        return '({} {})'.format(label, ' '.join(nodes)), True
    assert tree[0] != '-NONE-'
    new_line, is_okay = helper(tree)
    assert is_okay == True
    return new_line


class CRAFT(Dataset):
    r"""The Colorado Richly Annotated Full-Text (CRAFT) Corpus from
    ``"The Colorado richly annotated full text (CRAFT) corpus: multi-model
    annotation in the biomedical domain. In Handbook of Linguistic Annotation"
    <https://github.com/UCDenver-ccp/CRAFT>``.

    This dataset is useful for constituency parsing with biomedical text, and is
    one of larger readily available constituency parsing datasets outside of
    LDC.

    """

    def __init__(self, split='train', download=True):
        super().__init__()

        self.data_directory = data_directory = './particle-data-cache'
        self.tmp_directory = os.path.join(data_directory, 'tmp')
        self.target = os.path.join(data_directory, 'CRAFT-master')

        if download:
            self.download()

        self.preprocess(split)

        split_target = os.path.join(self.target, '{}.clean.txt'.format(split))

        self.corpus = self.read_corpus(split_target)

    def read_corpus(self, path):
        corpus = []
        with open(path) as f:
            for line in f:
                corpus.append(tree_from_string(line, add_space=True))
        return corpus

    def download(self):
        data_directory, tmp_directory, target = \
            self.data_directory, self.tmp_directory, self.target

        if os.path.exists(target):
            print('Found data. Skipping download.')
            return

        os.system("mkdir -p {}".format(data_directory))
        os.system("mkdir -p {}".format(tmp_directory))
        os.system("curl -L https://github.com/UCDenver-ccp/CRAFT/archive/master.zip --output {}/craft-data.zip".format(tmp_directory))
        os.system("unzip {}/craft-data.zip -d {}".format(tmp_directory, data_directory))

        assert os.path.exists(target), "File not found after download."

    def preprocess(self, split):
        def read_file_ids(path):
            output = []
            with open(path) as f:
                for line in f:
                    output.append(line.strip())
            return output

        def read_file_for_trees(path):
            with open(path) as f:
                for line in f:
                    line = line.strip()[1:-1].strip()
                    if not line:
                        continue
                    line = line.replace('[', '-LSB-').replace(']', '-RSB-')
                    tree = tree_from_string(line, add_space=True)

                    # Hacky way to remove -NONE- labels.
                    cleaned_line = remove_none(tree)
                    tree = tree_from_string(cleaned_line, add_space=True)
                    yield tree

        train_ids = os.path.join(self.target, 'articles/ids/craft-ids-train.txt')
        test_ids = os.path.join(self.target, 'articles/ids/craft-ids-test.txt')
        corpus_path = os.path.join(self.target, 'structural-annotation/treebank/penn')

        configs = {}
        configs['train'] = (train_ids,)
        configs['test'] = (test_ids,)

        (file_ids,) = configs[split]
        split_target = os.path.join(self.target, '{}.clean.txt'.format(split))
        print('writing to {} ...'.format(split_target))
        with open(split_target, 'w') as f:
            file_ids = read_file_ids(file_ids)
            for fid in tqdm(file_ids, desc=f'read[{split}]'):
                section_path = os.path.join(corpus_path, fid + '.tree')
                for tree in read_file_for_trees(section_path):
                    f.write(print_tree(tree) + '\n')
