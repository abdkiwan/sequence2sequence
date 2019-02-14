

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from dataset_loader import prepareData
from model import EncoderRNN, DecoderRNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1

dataset_dir = 'webnlg-dataset/webnlg_challenge_2017'

print('Getting vocabulary from train dataset')
vocab, _ = prepareData(dataset_dir, mode='train', n_triples=1)
print('Reading test dataset.')
_, pairs = prepareData(dataset_dir, mode='dev', n_triples=1)

for pair in pairs:
    vocab.addSentence(pair[0])
    vocab.addSentence(pair[1])

def indexesFromSentence(vocab, sentence):
    return [vocab.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(vocab, sentence):
    indexes = indexesFromSentence(vocab, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(vocab, pair):
    input_tensor = tensorFromSentence(vocab, pair[0])
    target_tensor = tensorFromSentence(vocab, pair[1])
    return (input_tensor, target_tensor)


def evaluate(encoder, decoder, sentence):
    with torch.no_grad():
        input_tensor = tensorFromSentence(vocab, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(input_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        
        di = 0
        while True:
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(vocab.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()
            
            di += 1
            
        return decoded_words

hidden_size = 512
encoder = EncoderRNN(vocab.n_words, hidden_size).to(device)
decoder = DecoderRNN(hidden_size, vocab.n_words).to(device)

encoder.load_state_dict(torch.load('encoder.pt'))
decoder.load_state_dict(torch.load('decoder.pt'))

f = open("output.txt", "a")
 
for pair in pairs:
    
    print(pair)
    triple = pair[0]
    target_sentence = pair[1]
    predicted_sentence = evaluate(encoder, decoder, triple)
    predicted_sentence = ' '.join(predicted_sentence[:-1])
    f.write(triple)
    f.write('\n')
    f.write(target_sentence)
    f.write('\n')
    f.write(predicted_sentence)
    f.write('\n')
    f.write('#####################################################################################')
    f.write('\n')
    
    
 
