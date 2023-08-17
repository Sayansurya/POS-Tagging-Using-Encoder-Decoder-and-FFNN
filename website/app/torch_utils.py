import torch 
import re
from ffnn_model import FFNN
from enc_model import Encoder
from enc_dec_model import Seq2SeqPOSTagger
from gensim.models import KeyedVectors
import pickle

torch.manual_seed(3137)
index2tag = pickle.load(open('../../data/index_to_tag.pickle', 'rb'))
word2vec = KeyedVectors.load_word2vec_format('../../word_vectors/GoogleNews-vectors-negative300.bin', binary=True)
unknown_words = pickle.load(open('../../word_vectors/unknown_words.pickle', 'rb'))

# load model
model = FFNN(900, 12, [1024, 512, 512])
model.load_state_dict(torch.load('../../part1/runs/FINAL_epochs=8,batch_size=1024,hidden_dim=[1024, 512, 512],timestamp=2023-03-10_01-25-08/final_model.pt'))
enc_model = Encoder(300, 128, 1, 12)
enc_model.load_state_dict(torch.load('../../part2/runs/enc_only,epochs=5,batch_size=128,hidden_dim=128,timestamp=2023-03-10_01-27-26/final_model.pt'))
encoder_decoder = Seq2SeqPOSTagger(
    encoder_input_dim=300,
    decoder_input_dim=268,
    output_dim=12,
    hidden_dim=128,
    num_layers=1
)
encoder_decoder.load_state_dict(torch.load('../../part2/runs/epochs=5,batch_size=128,hidden_dim=128,timestamp=2023-03-09_21-31-24/final_model.pt'))

def gen_wv(word, rand=True): # generate word vectors 
    global word2vec, unknown_words
    word = word.lower()
    try:
        word_vec = torch.tensor(word2vec[word]).reshape(1,-1) 
        return word_vec
    except: # not in word2vec
        try:
            if word in unknown_words.keys(): # if in unknown words
                word_vec = unknown_words[word].clone().detach().reshape(1,-1)
                return word_vec
            else:
                if re.search("'", word):
                    word = re.split("'", word)[0] # words with apostrophe are queried by removing apostrophe 
                    word_vec = torch.tensor(word2vec[word]).reshape(1,-1) 
                    return word_vec
                if re.search('-', word): # for compound words, word vector of each word is averaged 
                    word_vec = torch.randn((1,300)) if rand else torch.zeros((1,300))
                    words = re.split('-', word)
                    for w in words:
                        try:
                            word_vec += word2vec[w]
                        except:
                            if w in unknown_words.keys():
                                word_vec += unknown_words[w]                            
                    word_vec = word_vec/len(words)
                    return word_vec
                else:
                    if word not in unknown_words.keys():
                        word_vec = torch.randn((1,300)) if rand else torch.zeros((1,300))
                    else:
                        word_vec = unknown_words[word]
                    return word_vec
        except:
            if word not in unknown_words.keys():
                word_vec = torch.randn((1,300)) if rand else torch.zeros((1,300))
            else:
                word_vec = unknown_words[word]
            return word_vec

# preprocess sentence
def ffnn_prediction(sentence):
    sentence_vector = []
    for word in sentence.split():
        print(word)
        word_vec = gen_wv(word)
        sentence_vector.append(word_vec)
    sentence_vector = torch.cat(sentence_vector)
    word_vec_dim=300
    context_size=1
    len_sentence = sentence_vector.shape[0]

    ffnn_input = []

    for word_index in range(len_sentence):
        # index of context window w.r.t to index of word in sentence
        index = torch.cat((word_index - (context_size - torch.arange(context_size)), word_index + (torch.arange(context_size+1))))
        # pad with zero if no words before word to be tagged
        # pad with zero if no words after the word to be tagged
        context_vector = torch.cat((
            torch.zeros((sum(index < 0), word_vec_dim)),
            sentence_vector.index_select(
                dim=0, index=index[(index >= 0) & (index < len_sentence)]),
            torch.zeros((sum(index >= len_sentence), word_vec_dim)),
        )).flatten()
        ffnn_input.append(context_vector.reshape(1,-1))
    ffnn_input = torch.cat(ffnn_input)    
    with torch.no_grad():
        output = model.predict(ffnn_input).dim
    tags = []
    for i in torch.argmax(output, dim=1).tolist():
        tags.append(index2tag[i])
    words = sentence.split()
    result = list(map(lambda i,j : (i,j) , words, tags))
    return result

def enc_ffnn_predictions(sentence):
    sentence_vector = []
    for word in sentence.split():
        print(word)
        word_vec = gen_wv(word)
        sentence_vector.append(word_vec)
    sentence_vector = torch.cat(sentence_vector).unsqueeze(0)
    softmax = torch.nn.Softmax(dim=1)
    with torch.no_grad():
        enc_output = enc_model(sentence_vector.to('cuda'))
        enc_output = softmax(enc_output.squeeze(0))
    tags = []
    for i in torch.argmax(enc_output, dim=1).tolist():
        tags.append(index2tag[i])
    words = sentence.split()
    result = list(map(lambda i,j : (i,j) , words, tags))
    return result

def enc_dec_predictions(sentence):
    sentence_vector = []
    for word in sentence.split():
        print(word)
        word_vec = gen_wv(word)
        sentence_vector.append(word_vec)
    sentence_vector = torch.cat(sentence_vector).unsqueeze(0)
    softmax = torch.nn.Softmax(dim = 1)
    with torch.no_grad():
        enc_dec_output = encoder_decoder.predict(sentence_vector.to('cuda'))
        enc_dec_output = softmax(enc_dec_output.squeeze(0))
    tags = []
    for i in torch.argmax(enc_dec_output, dim=1).tolist():
        tags.append(index2tag[i])
    words = sentence.split()
    result = list(map(lambda i,j : (i,j) , words, tags))
    print(result)
    return result
