# -*- coding: utf-8 -*-

import os
import re
from bs4 import BeautifulSoup
import nltk

SOS_token = 0
EOS_token = 1

def preprocess_triple(triple):
    # Removing camel casing
    new_triple = []
    for entity in triple:
        entity = entity.lower()
        entity = entity.strip()
        entity = entity.replace('á', 'a')
        entity = entity.replace('ä', 'a')
        entity = entity.replace('ò', 'o')
        entity = entity.replace('ó', 'o')
        entity = entity.replace('ö', 'o')
        entity = entity.replace('ü', 'u') 
        # Removing non-letter and non-digit characters
        entity = re.sub(r"[^a-zA-Z1234567890.!?]+", r" ", entity)
        # Removing camel casing
        entity = re.sub("([a-z])([A-Z])","\g<1> \g<2>",entity)
        entity = entity.split(' ')
        for e in entity:
            if not e.isspace() and e: new_triple.append(e)

    return new_triple

def preprocess_sentence(sentence):
    sentence = [s.strip() for s in sentence]
    sentence = [s.lower() for s in sentence]

    return sentence

def parse_xml(file_dir):

    pairs = []
    handler = open(file_dir).read()
    soup = BeautifulSoup(handler, 'lxml')
    entries = soup.findAll('entry')
    for entry in entries:
        triple = entry.find('mtriple').text.split('|')
        triple = preprocess_triple(triple)
       
        for sentence in entry.findAll('lex'):
            sentence = nltk.word_tokenize(sentence.text.lower())
            sentence = preprocess_sentence(sentence)
            pairs.append((triple, sentence))
    
    return pairs

class Vocab:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence:
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def prepareData(dataset_dir, mode='train', n_triples=1):
    
    xml_file_dir = dataset_dir+'/'+mode+'/'+str(n_triples)+'triples/'
    all_pairs = []
    for xml_file_name in os.listdir(xml_file_dir):
        pairs = parse_xml(xml_file_dir+'/'+xml_file_name)
        all_pairs += pairs
        
    print("Read %s sentence pairs" % len(all_pairs))    
    
    vocab = Vocab()
    for pair in all_pairs:
        vocab.addSentence(pair[0])
        vocab.addSentence(pair[1])
    
    print("Counted words: ", vocab.n_words)

    return vocab, all_pairs
    



