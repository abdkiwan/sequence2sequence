

from __future__ import unicode_literals, print_function, division
import io
from io import open
import unicodedata
import string
import re
import random
import urllib
import requests 
import numpy as np

SOS_token = 0
EOS_token = 1

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from dataset_loader import prepareData
from model import EncoderRNN, DecoderRNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def indexesFromSentence(vocab, sentence):
    return [vocab.word2index[word] for word in sentence]

def tensorFromSentence(vocab, sentence):
    indexes = indexesFromSentence(vocab, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorsFromPair(vocab, pair):
    input_tensor = tensorFromSentence(vocab, pair[0])
    target_tensor = tensorFromSentence(vocab, pair[1])
    return (input_tensor, target_tensor)


def evaluate_sentence(encoder, decoder, sentence, vocab):
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
                decoded_words.append('.')
                break
            else:
                decoded_words.append(vocab.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()
            
            di += 1
            
        return decoded_words

def load_embeddings(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        word = tokens[0]
        vector = tokens[1:]
        vector = [float(v) for v in vector]
        data[word] = vector
    return data

def prepare_embeddings_array(vocab, embeddings):
    embeddings_array = np.random.uniform(-0.25, 0.25, (vocab.n_words, len(embeddings['a'])))
    for idx in range(vocab.n_words):
        word = vocab.index2word[idx]
        if word in embeddings:
            embeddings_array[idx] = embeddings[word]
    return torch.FloatTensor(embeddings_array)

import json
if __name__ == "__main__":
    
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

    print("Reading word embeddings.")
    embeddings = load_embeddings('cc.en.300.vec')
    embeddings_array = prepare_embeddings_array(vocab, embeddings)
    del embeddings
    
    hidden_size = 300
    encoder = EncoderRNN(vocab.n_words, hidden_size, embeddings_array).to(device)
    decoder = DecoderRNN(hidden_size, vocab.n_words).to(device)

    encoder.load_state_dict(torch.load('encoder.pt'))
    decoder.load_state_dict(torch.load('decoder.pt'))

    outputs_f = open("outputs.txt", "w")
    hypothesis_f = open("hypothesis.txt", "w")
    reference_f = open("reference.txt", "w")
    for pair in pairs:

        triple = pair[0]
        reference = pair[1]
        reference = ' '.join(reference)
        hypothesis = evaluate_sentence(encoder, decoder, triple, vocab)
        hypothesis = ' '.join(hypothesis)
        triple = ' '.join(triple)
        outputs_f.write(triple)
        outputs_f.write('\n')
        outputs_f.write(reference)
        outputs_f.write('\n')
        outputs_f.write(hypothesis)
        outputs_f.write('\n')
        outputs_f.write('#####################################################################################')
        outputs_f.write('\n')
        
        hypothesis_f.write(hypothesis)
        hypothesis_f.write('\n')
        
        reference_f.write(reference)
        reference_f.write('\n')
