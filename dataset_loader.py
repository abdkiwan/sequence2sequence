

import os
import re
from bs4 import BeautifulSoup

SOS_token = 0
EOS_token = 1

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def parse_xml(file_dir):

    pairs = []
    handler = open(file_dir).read()
    soup = BeautifulSoup(handler, 'lxml')
    entries = soup.findAll('entry')
    for entry in entries:
        triple = ''.join(entry.find('mtriple').text.split('|'))
        
        for sentence in entry.findAll('lex'):
            sentence = sentence.text
            pairs.append((triple, sentence))
    
    return pairs

class Vocab:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
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
    



