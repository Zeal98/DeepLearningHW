import math
import json

filepath = './MLDS_hw2_1_data/'
filename = ['training_label.json', 'testing_label.json']
DIM = 3072  # dim of the word vector
vectorMaxLength = 18 # maximum length of prediction vector

def build_dict():
    input = json.load(open((filepath + filename[0]), 'r'))
#    print(input[0]['caption'], input[0]['id'], sep='\n')

    words = dict()
    words['<BOS>'] = 0
    words['EOS'] = 1

    wordCounter = dict()
    total = 0

    for video in input:
        for sent in video['caption']:
            temp = sent.lower()
            temp = temp.split(' ')
            for word in temp:
                if word.endswith('.'):
                    word = word[:len(word)-1]
                if word not in wordCounter.keys():
                    wordCounter[word] = 0
                wordCounter[word] += 1
                total += 1

    seq = list()
    for word in wordCounter.keys():
        seq.append(wordCounter[word])
    seq.sort(reverse=True)
    print(total, len(seq), seq[DIM])

    for word in wordCounter.keys():
        if wordCounter[word] > seq[DIM]:
            words[word] = len(words)

    json.dump(words, open('temp.json', 'w'))
    return 'temp.json'


def label_2_vector(filename, dictname):
    labels = json.load(open(filename, 'r'))
    words = json.load(open(dictname, 'r'))
    vectors = list()
    maxLength = 0
    for tuple in labels:
        captions = list()
        for sent in tuple['caption']:
            temp = sent.lower()
            if temp.endswith('.'):
                temp = temp[:len(temp)-1]
            temp = temp.split(' ')
            vector = list()
            vector.append(0)
            for word in temp:
                if word in words.keys():
                    vector.append(words[word])
                else:
                    vector.append(3000-1)
            vector.append(1)
            if len(vector) > maxLength:
                maxLength = len(vector)
            while (len(vector) < vectorMaxLength):
                vector.append(3000-2)
                if len(vector) == vectorMaxLength:
                    captions.append(vector)
        tempdict = dict()
        tempdict['caption'] = captions
        tempdict['id'] = tuple['id']
        vectors.append(tempdict)

    print(maxLength)
    json.dump(vectors, open('training_vac.json', 'w'))
    return 'training_vac.json'


def vec2label(idx, words_re):
    if idx == 3000-2:
        return ''
    if idx == 3000-1:
        return ''
    if idx == 0:
        return ''
    if idx == 1:
        return ''
    if idx not in words_re.keys():
        return '<Unknown>'
    else:
        return words_re[idx]
def vector_2_label(filename, dictname):
    words = json.load(open(dictname, 'r'))
    words_re = dict()
    words_re[3000-1] = '<Unknown>'
    words_re[3000-2] = '<Pad>'
    words_re[0] = '<BOS>'
    words_re[1] = '<EOS>'
    for word in words.keys():
        words_re[words[word]] = word

    vectors = json.load(open(filename, 'r'))
    output = dict()
    for my_tuple in vectors:
        sent = str()
        for idx in my_tuple['prediction']:
            word = vec2label(idx, words_re)
            sent = sent + ' ' + word



if __name__ == '__main__':
    build_dict()
    label_2_vector(filepath+filename[0], 'temp.json')
