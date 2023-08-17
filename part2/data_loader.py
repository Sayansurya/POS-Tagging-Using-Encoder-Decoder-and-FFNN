import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from nltk.corpus import brown
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pickle


class EncoderDataset(Dataset):
    def __init__(self, corpus_index, word_vec_dim, truncate_seq, seq_len):
        self.corpus_index = corpus_index
        self.word_vec_dim = word_vec_dim
        self.seq_len = seq_len
        self.truncate_seq = truncate_seq

    def __getitem__(self, index):
        batch_file_index, sentence_index, word_index = self.corpus_index[index]
        word_vectors, tag_vectors = pickle.load(open(
            f'../data/encoded_corpus/encoded_{batch_file_index}.pickle', 'rb'))[sentence_index]
        mask = torch.ones_like(tag_vectors[:, 0])
        if self.truncate_seq:
            return word_vectors[:self.seq_len, :], tag_vectors[:self.seq_len, :], mask[:self.seq_len]
        else:
            return word_vectors, tag_vectors, mask

    def __len__(self):
        return len(self.corpus_index)


def collate_fn(batch):
    # tuple of word_vectors [sent_len, 300]
    word_vectors, tag_vectors, mask = zip(*batch)
    X = nn.utils.rnn.pad_sequence(word_vectors, batch_first=True)
    Y = nn.utils.rnn.pad_sequence(tag_vectors, batch_first=True)
    mask = nn.utils.rnn.pad_sequence(
        mask, batch_first=True).type(torch.bool)
    return X, Y, mask


# def main():
#     with open('../data/corpus_index.pickle', 'rb') as file:
#         corpus_index = pickle.load(file)
#     dataset = EncoderDataset(corpus_index, 300)
#     dataloader = DataLoader(dataset, batch_size=32,
#                             shuffle=True, collate_fn=collate_fn)
#     for index, data in enumerate(dataloader):
#         print(f"Batch {index} Data {data}")


# if __name__ == '__main__':
#     main()
