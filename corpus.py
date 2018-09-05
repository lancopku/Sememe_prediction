import pickle, json
import numpy as np
import random
MAXLEN = 100

class Language:
    def __init__(self, name):
        self.name = name
        self.word2id = {'PADDING':0,'SOS':1,'EOS':2,'OOV':3}
        self.id2word = ['PADDING','SOS','EOS','OOV']
        self.word2count = {}
        self.SOS_token = 1
        self.EOS_token = 2

    def add_sentence(self, sentence):
        if sentence != None and len(sentence) > 0:
            for word in sentence:
                if len(word) > 0:
                    self.add_word(word)
		
    def add_word(self, word):
        if word not in self.word2id:
            self.word2id[word] = len(self.word2id)
            self.id2word.append(word)
            self.word2count[word] = 0
        self.word2count[word] += 1

    def sent2id(self, sent, min_count=0, add_eos=False):
        if sent != None:
            result = []
            for word in sent:
                if len(word) > 0: 
                    if word not in self.word2id:
                        result.append(self.word2id['OOV'])
                    elif self.word2count[word] >= min_count:
                        result.append(self.word2id[word])
                    else:
                        result.append(self.word2id['OOV'])
            if add_eos:
                result.append(self.EOS_token)
            return result
        return [self.EOS_token]

def read_emb(fname):
    vec = {}
    lines = open(fname).readlines()
    for line in lines[1:]:
        tem = line.strip().split()
        word = tem[0]
        v = np.array([float(num) for num in tem[1:]])
        vec[word] = v
    return vec

def match_emb(vec, language):
    emb = [np.zeros(200),np.zeros(200),np.zeros(200),np.zeros(200)]
    for word in language.id2word[4:]:
        emb.append(vec[word])
    return np.array(emb)

def split_data(data, random_seed=255):
    random.seed(random_seed)
    random.shuffle(data)
    size = len(data)
    train_data = data[:int(size*0.8)]
    dev_data = data[int(size*0.8):int(size*0.9)]
    test_data = data[int(size*0.9):]
    return data, train_data, dev_data, test_data

if __name__ == '__main__':
    word_set = json.load(open('../dataset/word_sense_description.json'))
    sememe_voc = Language('sememe')
    chinese = Language('chinese')
    data = []
    combine_data = []
    description_data = []
    definition_data = []
    for line in open('../dataset/word_sememes.txt'):
        tem = line.strip().split('\t')
        if len(tem) < 2: continue
        sememes = tem[1].split(' ')
        sememe_voc.add_sentence(sememes)
    for word in word_set:
        word_object = word_set[word]
        chinese.add_sentence(word)
        chinese.add_sentence(word_object['description'])
        chinese.add_sentence(word_object['definition'])
        #sememe_voc.add_sentence(word_object['sense'])
        data_item = {}
        data_item['word'] = chinese.sent2id(word_object['word'])
        data_item['word_raw'] = word_object['word']
        data_item['description'] = chinese.sent2id(word_object['description'], min_count=2, add_eos=True)
        data_item['definition'] = chinese.sent2id(word_object['definition'], min_count=2, add_eos=True)
        data_item['sense'] = sememe_voc.sent2id(word_object['sense'],min_count=5, add_eos=True)
        if len(data_item['sense']) < 2 and data_item['sense'][0] == sememe_voc.EOS_token:
            continue
        if len(word) > 0 and ((word_object['description']!=None and len(word_object['description']) > 0) or (word_object['definition']!=None and len(word_object['definition']) > 0)) and len(word_object['sense']) > 0:
            data.append(data_item)
        if len(word) > 0 and (word_object['description']!=None and len(word_object['description']) > 0):
            description_data.append(data_item)
        if len(word) > 0 and (word_object['definition']!=None and len(word_object['definition']) > 0):
            definition_data.append(data_item)
        if len(word) > 0 and ((word_object['description']!=None and len(word_object['description']) > 0) and (word_object['definition']!=None and len(word_object['definition']) > 0)) and len(word_object['sense']) > 0:
            combine_data.append(data_item)
    print('Chinese',len(chinese.word2id),'Sememe',len(sememe_voc.word2id))
    print('any',len(data), 'description',len(description_data), 'definition', len(definition_data), 'combine', len(combine_data))
    vec = read_emb('../dataset/sememe_vec.txt')
    sememe_emb = match_emb(vec, sememe_voc)
    combine_data, train_data, dev_data, test_data = split_data(combine_data)
    random.seed(255)
    random.shuffle(data)
    random.shuffle(description_data)
    random.shuffle(definition_data)
    pickle.dump((chinese, sememe_voc, sememe_emb), open('chinese_sememe_voc.pkl','wb'))
    pickle.dump((data, description_data, definition_data, (train_data, dev_data, test_data)), open('sememe_prediction_data.pkl','wb'))
