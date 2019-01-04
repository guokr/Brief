#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import torch

def load_embedding(embedding_file, dim, vocab_size, index2word):
    """
    :param embedding_file: path of embedding file
    :type embedding_file: str
    :param dim: dimension of vector
    :type dim: int
    :param vocab_size: size of vocabulary
    :type vocab_size: int
    :param index2word: index => word
    :type index2word: dict

    Load pre-trained embedding file.

    First line of file should be the number of words and dimension of vector.
    Then each line is combined of word and vectors separated by space.

    ::

        1024, 64 # 1024 words and 64-d
        a 0.223 0.566 ......
        b 0.754 0.231 ......
        ......

    """
    word2vec = {}
    with open(embedding_file, 'r', encoding='utf-8') as f:
        print('Embedding file header: {}'.format(f.readline())) # ignore header
        for line in f.readlines():
            items = line.strip().split(' ')
            word2vec[items[0]] = [float(vec) for vec in items[1:]]

    embedding = [[]] * vocab_size
    bound = np.sqrt(6.0) / np.sqrt(vocab_size)
    count_exist, count_not_exist = 0, 0
    for i in range(vocab_size):
        word = index2word[i]
        try:
            embedding[i] = word2vec[word]
            count_exist += 1
        except:
            embedding[i] = np.random.uniform(-bound, bound, dim)
            count_not_exist += 1

    print('word exists embedding:', count_exist, '\tword not exists:', count_not_exist)
    embedding = np.array(embedding)
    return embedding


class MiniBatchWrapper(object):
    """
    wrap the simple torchtext iter
    """
    def __init__(self, dl, source_var, target_var):
        self.dl, self.source_var, self.target_var = dl, source_var, target_var

    def __iter__(self):
        for batch in self.dl:
            source_seq = getattr(batch, self.source_var) # we assume only one input in this wrapper
            target_seq = getattr(batch, self.target_var) # we assume only one input in this wrapper
            # if self.y_vars is  not None:
            # temp = [getattr(batch, feat).unsqueeze(1) for feat in self.y_vars]
            # y = torch.cat(temp, dim=1).float()
            # else:
                # y = torch.zeros((1))
            # yield (x, y)
            yield (source_seq, target_seq)

    def __len__(self):
        return len(self.dl)



