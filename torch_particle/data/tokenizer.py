class Tokenizer(object):
    def __init__(self):
        super().__init__()
        self.vocab = set()
        self.frozen = False

    def add_sequence_list(self, sequence_list):
        for seq in sequence_list:
            self.vocab.update(seq)

    def finalize(self):
        word_to_idx = {}

        word_to_idx['<PAD>'] = len(word_to_idx)

        for x in sorted(self.vocab):
            assert x not in word_to_idx
            word_to_idx[x] = len(word_to_idx)

        self.word_to_idx = word_to_idx
        self.frozen = True

    def indexify(self, seq):
        return [self.word_to_idx[x] for x in seq]
