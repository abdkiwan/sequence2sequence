

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

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('cuda:0')

from dataset_loader import prepareData
from model import EncoderRNN, DecoderRNN

SOS_token = 0
EOS_token = 1

dataset_dir = 'webnlg-dataset/webnlg_challenge_2017'

print('Reading train dataset')
vocab, train_pairs = prepareData(dataset_dir, mode='train', n_triples=1)
print('Reading test dataset.')
_, test_pairs = prepareData(dataset_dir, mode='dev', n_triples=1)

for pair in test_pairs:
    vocab.addSentence(pair[0])
    vocab.addSentence(pair[1])

print("Writing vocabs into a text file.")
f= open("vocab.txt","w+")
for i in range(vocab.n_words):
    f.write(vocab.index2word[i])
    f.write('\n')

import io

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

print("Reading word embeddings.")
embeddings = load_embeddings('cc.en.300.vec')
embeddings_array = prepare_embeddings_array(vocab, embeddings)

print("Writing vocabs that don't have an embedding vector into a text file.")
f= open("vocab_no_embedding.txt","w+")
for i in range(vocab.n_words):
    if vocab.index2word[i] not in embeddings:
        f.write(vocab.index2word[i])
        f.write('\n')
del embeddings


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.savefig("loss.png")
 
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

import time
import math

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


teacher_forcing_ratio = 0.5

def cal_accuracy(encoder, decoder, pairs):
    acc = 0

    for pair in pairs:

        triple = pair[0]
        target_sentence = pair[1]
        target_sentence = ' '.join(target_sentence)
        predicted_sentence = evaluate_sentence(encoder, decoder, triple, vocab)
        predicted_sentence = ' '.join(predicted_sentence)
       
        bleu_score = bleu.corpus_bleu([predicted_sentence], [[target_sentence]])[0][0]*100
        acc += bleu_score
            
    return (acc / float(len(pairs)))


def train_pair(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(input_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing
    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def train(encoder, decoder, n_epochs, learning_rate=0.01):
    start = time.time()
    loss_list = [] 

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
   
    criterion = nn.NLLLoss()

    for epo in range(1, n_epochs+1):
        
        total_loss = 0
        for pair in train_pairs:

            pair = tensorsFromPair(vocab, pair)
            input_tensor = pair[0]
            target_tensor = pair[1]

            loss = train_pair(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
            total_loss += loss
            loss_list.append(loss)
            
        avg_loss = total_loss / len(train_pairs)
        
        print('%s (%d %d%%) Loss : %.4f' % (timeSince(start, epo / n_epochs),
                     epo, epo / n_epochs * 100, avg_loss))

    torch.save(encoder.state_dict(), 'encoder.pt')
    torch.save(decoder.state_dict(), 'decoder.pt')
    showPlot(loss_list)

hidden_size = 300
encoder = EncoderRNN(vocab.n_words, hidden_size, embeddings_array).to(device)
decoder = DecoderRNN(hidden_size, vocab.n_words).to(device)

train(encoder, decoder, n_epochs=25)
