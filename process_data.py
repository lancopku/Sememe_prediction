#encoding=utf-8
import pickle, json
import xlrd
import re


class Word:
    """
    Technically, a word has multiple senses in Hownet. However, it makes the problem much too complicated, and many of the senses are just trivial. Therefore, we only return the first sense, but we still maintain all the senses in the Word object.
    For the description, we only keep the first one to simplify the problem
    """
    def __init__(self, word):
        self.word = word
        self.senses = []
        self.description = None
        self.definition = None

    def add_sense(self, sense):
        for s in self.senses:
            if s.equals(sense):
                return
        self.senses.append(sense)

    def add_description(self, description):
        self.description = description

    def add_definition(self, definition):
        self.definition = definition

    def to_json(self):
        return {'word':self.word, 'sense':self.senses[0].sememes, 'description':self.description,'definition':self.definition}


class Sense:
    def __init__(self, word, sememes=None):
        self.word = word
        self.sentiment = None
        if sememes == None:
            self.sememes = []
        else:
            self.sememes = [s for s in sememes if len(s)>0]

    def add_sememe(self, sememe):
        if len(sememe)>0 and sememe not in set(self.sememes):
            self.sememes.append(sememe)

    def add_sentiment(self, sentiment):
        self.sentiment = sentiment

    def equals(self, other):
        if type(other) == type(self) and self.word == other.word and set(self.sememes) == set(other.sememes):
            return True
        return False

    def to_json(self):
        return {'word':self.word, 'sememes':self.sememes, 'sentiment':self.sentiment}


def parse_sememe(DEF, bound):
    sememes = []
    sememe = ''
    begin = False
    for x in DEF:
        if x == '|':
            begin = True
            continue
        if x in bound:
            begin = False
            sememes.append(sememe)
            sememe = ''
            continue
        if begin:
            sememe += x
    if len(sememe) > 0:
        sememes.append(sememe)
    return sememes

def read_excel(fname, word_set):
    workbook = xlrd.open_workbook(fname)
    table = workbook.sheets()[0]
    for i in range(1,table.nrows):
        row_values = table.row_values(i)
        word = row_values[0]
        sememes = parse_sememe(row_values[-1], bound=[u',',u')'])
        if len(sememes) < 1:
            continue
        if word not in word_set:
            word_set[word] = Word(word)
        word_set[word].add_sense(Sense(word, sememes))
    return word_set

def read_text_file(fname, word_set):
    lines = open(fname).readlines()
    for i in range(0,len(lines),12):
        if len(lines[i+1].strip()) == 4:
            continue
        word = lines[i+1].split('=')[1].strip()
        pos = lines[i+2].split('=')[1].split(' ')[0].strip()
        sentiment = lines[i+3].split('|')[1].strip() if '|' in lines[i+3] else None
        sememes = parse_sememe(lines[i+9].strip(), bound=[':', '}',u':'])
        if len(sememes) < 1:
            continue
        RMK = lines[i+10].split('=')[1].strip()
        if RMK == u'非常用汉字':
            continue
        if word not in word_set:
            word_set[word] = Word(word)
        sense = Sense(word, sememes)
        if sentiment != None:
            sense.add_sentiment(sentiment)
        word_set[word].add_sense(sense)
    return word_set

def read_description(fname, word_set, is_def=False):
    for line in open(fname):
        tem = line.split(u':')
        if len(tem) < 2:
            tem = line.split(u'：')
        if len(tem) < 2:
            continue
        word = tem[0]
        description = ''
        if len(tem) == 2:
            description = tem[1]
        else:
            description = ':'.join(tem[1:])
        english_pattern = re.compile('[a-zA-Z]+')
        description = re.sub(english_pattern, '', description)
        if all(ord(c) < 128 for c in description):
            continue
        if word in word_set:
            if is_def:
                word_set[word].add_definition(description)
            else:
                word_set[word].add_description(description)
        else:
            word_set[word] = Word(word)
            word_set[word].add_description(description)
    return word_set


def convert2json(word_set):
    words = {}
    for word in word_set:
        '''
        if len(word_set[word].senses) == 1:
            words[word] = word_set[word].to_json()
    json.dump(words, open('word_sense_description_single.json','w'),ensure_ascii=False)
        '''
        if len(word_set[word].senses) > 0:
            words[word] = word_set[word].to_json()
    json.dump(words, open('word_sense_description.json','w'),ensure_ascii=False, indent=4)

        
def write_sememe(word_set):
    write = open('word_sememes.txt','w')
    write_sememe = open('sememes.txt','w')
    for word in word_set:
        senses = word_set[word].senses
        for sense in senses:
            write.write(word+'\t'+' '.join(sense.sememes)+'\n')
            write_sememe.write(' '.join(sense.sememes)+'\n')
    write.close()
    write_sememe.close()
        
if __name__ == '__main__':
    word_set = {}
    # Here we first read in the excel file, because we assume the quality of this file is better than the other one
    #word_set = read_excel('HYSem.xlsx',word_set)
    word_set = read_text_file('HowNet.txt', word_set)
    write_sememe(word_set)
    # baike.txt and cidian represent the baike descriptions and dictionary definitions repectively, the definitions in the dictionary seems to have better quality
    word_set = read_description('baike.txt', word_set)
    word_set = read_description('cidian.txt', word_set, is_def=True)
    convert2json(word_set)
    #pickle.dump(word_set, open('word_sense_description.pkl','wb'))
