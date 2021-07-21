import itertools
import torch

from config import Config


class Batcher:
    def __init__(self):
        self.config = Config()

    def indexes_from_sentence(self, voc, sentence):
        return [voc.word2index[word] for word in sentence.split(' ')] + [self.config.EOS_token]

    # Returns all items for a given batch of pairs
    def batch_to_train_data(self, voc, pair_batch):
        pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
        input_batch, output_batch = [], []
        for pair in pair_batch:
            input_batch.append(pair[0])
            output_batch.append(pair[1])
        inp, lengths = self._input_var(input_batch, voc)
        output, mask, max_target_len = self._output_var(output_batch, voc)
        return inp, lengths, output, mask, max_target_len

    def _zero_padding(self, indexes_batch):
        return list(itertools.zip_longest(*indexes_batch, fillvalue=self.config.PAD_token))

    def _binary_matrix(self, pad_list):
        m = []
        for i, seq in enumerate(pad_list):
            m.append([])
            for token in seq:
                if token == self.config.PAD_token:
                    m[i].append(0)
                else:
                    m[i].append(1)
        return m

    # Returns padded input sequence tensor and lengths
    def _input_var(self, l, voc):
        indexes_batch = [self.indexes_from_sentence(voc, sentence) for sentence in l]
        lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
        pad_list = self._zero_padding(indexes_batch)
        pad_var = torch.LongTensor(pad_list)
        return pad_var, lengths

    # Returns padded target sequence tensor, padding mask, and max target length
    def _output_var(self, l, voc):
        indexes_batch = [self.indexes_from_sentence(voc, sentence) for sentence in l]
        max_target_len = max([len(indexes) for indexes in indexes_batch])
        pad_list = self._zero_padding(indexes_batch)
        mask = self._binary_matrix(pad_list)
        mask = torch.BoolTensor(mask)
        pad_var = torch.LongTensor(pad_list)
        return pad_var, mask, max_target_len
