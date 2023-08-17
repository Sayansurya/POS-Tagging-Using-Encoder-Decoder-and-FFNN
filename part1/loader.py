from torch.utils.data import Dataset
import numpy as np
import torch
import pickle


class POSTaggedDataset(Dataset):
    # context_size = 3  [_, _, _, word, _, _, _]
    def __init__(self, corpus_index, context_size, word_vec_dim):
        super().__init__()
        self.context_size = context_size
        self.corpus_index = corpus_index
        self.word_vec_dim = word_vec_dim

    def __len__(self):
        return len(self.corpus_index)

    def __getitem__(self, index):
        file_index, sent_index, word_index = self.corpus_index[index]
        words_vec, tags_vec = pickle.load(
            open(f'../data/encoded_corpus/encoded_{file_index}.pickle', 'rb'))[sent_index]
        len_sentence = words_vec.shape[0]

        # index of context window w.r.t to index of word in sentence
        index = torch.cat((word_index - (self.context_size - torch.arange(
            self.context_size)), word_index + (torch.arange(self.context_size+1))))

        # pad with zero if no words before word to be tagged
        # pad with zero if no words after the word to be tagged
        context_vector = torch.cat((
            torch.zeros((sum(index < 0), self.word_vec_dim)),
            words_vec.index_select(
                dim=0, index=index[(index >= 0) & (index < len_sentence)]),
            torch.zeros((sum(index >= len_sentence), self.word_vec_dim)),
        )).flatten()
        tag_vector = tags_vec[word_index, :]
        return context_vector, tag_vector
