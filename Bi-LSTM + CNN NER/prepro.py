import numpy as np
import random
from keras.preprocessing.sequence import pad_sequences
import json
import re
import glob
import unidecode
import pandas as pd


def readfile(filename):
    '''
    read file
    return format :
    [ ['EU', 'B-ORG'], ['rejects', 'O'], ['German', 'B-MISC'], ['call', 'O'], ['to', 'O'], ['boycott', 'O'], ['British', 'B-MISC'], ['lamb', 'O'], ['.', 'O'] ]
    '''
    glob_data = []
    final_ents = []
    for file in glob.glob(filename+'/*.json'):
        #print(file)
        name = file.split(".ann")[0]
        #print(name)
        text_file = name + ".txt"
        if(text_file != ""):
            t_file = open(text_file,"r",encoding = "utf8")
    #         print("json filename: " + text_file)
    #         print("raw text of review: " + t_file.read())

        #look up table to see tagged words
            annotations = {}

    #         file = open('review1.txt','r')
            review = t_file.read()

    #         with open('ann.txt') as json_file:
            with open(name+'.ann.json') as json_file:
                data = json.load(json_file)
    #             print(data)
                for r in data['entities']:
            #         print('Full',r['offsets'])
            #         print('Index',r['offsets'][0]['start'])
            #         print('Food',r['offsets'][0]['text'])
                    food = r['offsets'][0]['text'].split()
                    #print(food)
                    i = int(r['offsets'][0]['start']) #used to keep track of position
                    #print(i)
                    c = 0
                    if len(food) == 1:
                        ner = 'U\n'
                        annotations[i] = [ner,food[0]]
                    else:
                        for j in range(len(food)):
                            if c==0:
                                #tag as the beginning
                                ner = 'B\n'
                                annotations[i] = [ner,food[j]]
                                i = i + len(food[j])+1
                            elif c == len(food)-1:
                                #tag as the end
                                ner = 'L\n'
                                annotations[i] = [ner,food[j]]
                            else:
                                #tag as in between
                                ner = 'I\n'
                                annotations[i] = [ner,food[j]]
                                i = i + len(food[j])+1
                            c+=1
            #         reviews[r['offsets'][0]['start']] = [r['offsets'][0]['text']]
            #        break
            dataFinal = {}
            #will need to change 
            dataFinal['id'] = name
            dataFinal['paragraphs'] = []
            sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s',review)
            s = []
            i = -1
            index = 0
            for sent in sentences:
                t = sent.split()
                t = [x for x in t if x not in ['.','+']]
                tokens = []
                for word in t:
                    b = review.find(word,i+1)
                    i = b
                    if b in annotations:
                        tags = annotations[b]
                        orth = word
                        orth = unidecode.unidecode(orth)
                        ner = tags[0]
                        index += 1
                        tokens.append([orth,ner])
                    else:
                        orth = word
                        orth = unidecode.unidecode(orth)
                        ner = 'O\n'
                        index += 1
                        tokens.append([orth,ner])
                        
                final_ents.append(tokens)

    return final_ents

def readfileAll(filename):
    '''
    read file
    return format :
    [ ['EU', 'B-ORG'], ['rejects', 'O'], ['German', 'B-MISC'], ['call', 'O'], ['to', 'O'], ['boycott', 'O'], ['British', 'B-MISC'], ['lamb', 'O'], ['.', 'O'] ]
    '''
    final_ents = []

    df = pd.read_csv(filename)
    df = df['text'].apply(lambda x: x + '.' if x[len(x)-1] not in [".","!","?"] else x)

    review = df.str.cat(sep=" ")

    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s',review)
    index = 0
    for sent in sentences:
        t = sent.split()
        t = [x for x in t if x not in ['.','+']]
        tokens = []
        for word in t:
            orth = word
            orth = unidecode.unidecode(orth)
            ner = 'O\n'
            tokens.append([orth,ner])
                
        final_ents.append(tokens)
                
    return final_ents


# define casing s.t. NN can use case information to learn patterns
def getCasing(word, caseLookup):
    casing = 'other'

    numDigits = 0
    for char in word:
        if char.isdigit():
            numDigits += 1

    digitFraction = numDigits / float(len(word))

    if word.isdigit():  # Is a digit
        casing = 'numeric'
    elif digitFraction > 0.5:
        casing = 'mainly_numeric'
    elif word.islower():  # All lower case
        casing = 'allLower'
    elif word.isupper():  # All upper case
        casing = 'allUpper'
    elif word[0].isupper():  # is a title, initial char upper, then all lower
        casing = 'initialUpper'
    elif numDigits > 0:
        casing = 'contains_digit'

    return caseLookup[casing]


# return batches ordered by words in sentence
def createEqualBatches(data):
    
    
    # num_words = []
    # for i in data:
    #     num_words.append(len(i[0]))
    # num_words = set(num_words)
    
    n_batches = 100
    batch_size = len(data) // n_batches
    num_words = [batch_size*(i+1) for i in range(0, n_batches)]
    
    batches = []
    batch_len = []
    z = 0
    start = 0
    for end in num_words:
        # print("start", start)
        for batch in data[start:end]:
            # if len(batch[0]) == i:  # if sentence has i words
            batches.append(batch)
            z += 1
        batch_len.append(z)
        start = end

    return batches, batch_len

def createBatches(data):
    l = []
    for i in data:
        l.append(len(i[0]))
    l = set(l)
    if 0 in l:
        l.remove(0)
    batches = []
    batch_len = []
    z = 0
    for i in l:
        for batch in data:
            if len(batch[0]) == i:
                batches.append(batch)
                z += 1
        batch_len.append(z)

    return batches,batch_len


# returns matrix with 1 entry = list of 4 elements:
# word indices, case indices, character indices, label indices
def createMatrices(sentences, word2Idx, label2Idx, case2Idx, char2Idx):
    unknownIdx = word2Idx['UNKNOWN_TOKEN']
    paddingIdx = word2Idx['PADDING_TOKEN']

    dataset = []

    wordCount = 0
    unknownWordCount = 0

    for sentence in sentences:
        wordIndices = []
        caseIndices = []
        charIndices = []
        labelIndices = []

        for word, char, label in sentence:
            wordCount += 1
            if word in word2Idx:
                wordIdx = word2Idx[word]
            elif word.lower() in word2Idx:
                wordIdx = word2Idx[word.lower()]
            else:
                wordIdx = unknownIdx
                unknownWordCount += 1
            charIdx = []
            for x in char:
                charIdx.append(char2Idx[x])
            # Get the label and map to int
            wordIndices.append(wordIdx)
            caseIndices.append(getCasing(word, case2Idx))
            charIndices.append(charIdx)
            labelIndices.append(label2Idx[label])

        dataset.append([wordIndices, caseIndices, charIndices, labelIndices])

    return dataset


def iterate_minibatches(dataset, batch_len):
    start = 0
    for i in batch_len:
        tokens = []
        caseing = []
        char = []
        labels = []
        data = dataset[start:i]
        start = i
        for dt in data:
            t, c, ch, l = dt
            l = np.expand_dims(l, -1)
            tokens.append(t)
            caseing.append(c)
            char.append(ch)
            labels.append(l)
        
        yield np.asarray(labels), np.asarray(tokens), np.asarray(caseing), np.asarray(char)        


# returns data with character information in format
# [['EU', ['E', 'U'], 'B-ORG\n'], ...]
def addCharInformation(Sentences):
    temp = Sentences.copy()
    for i, sentence in enumerate(Sentences):
        for j, data in enumerate(sentence):
            if type(data[1]) != list:
                chars = [c for c in data[0]]
                temp[i][j] = [data[0], chars, data[1]]
    return temp


# 0-pads all words
def padding(Sentences):
    maxlen = 52
    for sentence in Sentences:
        char = sentence[2]
        for x in char:
            maxlen = max(maxlen, len(x))
    for i, sentence in enumerate(Sentences):
        Sentences[i][2] = pad_sequences(Sentences[i][2], 52, padding='post')
    return Sentences
