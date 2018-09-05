import torch 
import codecs, os, sys
import time, random
import pickle
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim
import argparse
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import json
from corpus import Language
import numpy as np
from average_precision import mapk

use_cuda = torch.cuda.is_available()
MAX_LENGTH = 8
logsoftmax = torch.nn.LogSoftmax(dim=1)

def one_hot(ids, nclass):
    result = Variable(torch.zeros((ids.size()[0],nclass)))
    if use_cuda: result = result.cuda()
    result.scatter_(1,torch.unsqueeze(ids,-1),1.)
    return result

def multi_hot(ids, nclass):
    '''
        ids :: bs, sl
    '''
    result = Variable(torch.zeros((ids.size()[0],nclass)))    # bs, nv
    padding_index = Variable(torch.zeros((ids.size()[0],1)).long())      # bs, 1
    if use_cuda: 
        result = result.cuda()
        padding_index = padding_index.cuda()
    result.scatter_(1,ids,1.)
    result.scatter_(1,padding_index,0.)
    assert result.data[0][0] == 0.
    return result

def eval_map(reference, candidate):
    mean_ap = mapk(reference, candidate, k=5)
    return mean_ap

def eval_F(reference, candidate, log_path, descriptions, definitions, words):
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    ref_file = log_path+'/reference.txt'
    cand_file = log_path+'/candidate.txt'
    with codecs.open(ref_file, 'w', 'utf-8') as f:
        for d1,d2,s,w in zip(descriptions, definitions, reference, words):
            f.write('word: '+''.join(w)+'\n')
            f.write('description: '+("".join(d1)).replace('\n','')+'\ndefinition: '+("".join(d2)).replace('\n','')+'\n')
            f.write('sememes: '+" ".join(s)+'\n')
    with codecs.open(cand_file, 'w', 'utf-8') as f:
        for d1,d2,s,w in zip(descriptions, definitions,candidate, words):
            f.write('word: '+''.join(w)+'\n')
            f.write('description: '+("".join(d1)).replace('\n','')+'\ndefinition: '+("".join(d2)).replace('\n','')+'\n')
            f.write('sememes: '+" ".join(s)+'\n')

    total_right = 0.
    total_ref = 0.
    total_can = 0.
    for r,c in zip(reference, candidate):
        r_set = set(r)-{'EOS','PADDING'}
        c_set = set(c)-{'EOS','PADDING','OOV'}
        right = set.intersection(r_set, c_set)
        total_right += len(right)
        total_ref += len(r)
        total_can += len(c_set)
    total_can = total_can if total_can != 0 else 1
    precision = total_right/float(total_can)*100.
    recall = total_right/float(total_ref)*100.
    if precision == 0 or recall == 0:
        F1 = 0.
    else:
        F1 = precision*recall*2./(precision+recall)
    return precision, recall, F1

class Encoder(nn.Module):
    def __init__(self, voc_size, emb_size, hidden_size, n_layers=1, bidirectional=False):
        super(Encoder, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.emb_size = emb_size
        self.bidirectional = bidirectional
        self.gate_w1 = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.gate_w2 = nn.Linear(self.hidden_size*2, self.hidden_size)

        self.embedding = nn.Embedding(voc_size, emb_size, padding_idx=0)
        self.gru = nn.GRU(emb_size,  hidden_size, num_layers=self.n_layers, batch_first=True, bidirectional=self.bidirectional)

    def load_emb(self, emb):
        emb = torch.from_numpy(np.array(emb, dtype=np.float32))
        self.embedding.weight.data.copy_(emb)

    def combine(self, desc_state, def_state):
        gate_1 = F.sigmoid(self.gate_w1(torch.cat([desc_state, def_state],1)))
        gate_2 = F.sigmoid(self.gate_w2(torch.cat([desc_state, def_state],1)))
        result = desc_state*gate_1+def_state*gate_2
        return result

    def forward(self, args, desc, defi):
        batch_size = desc.size()[0]
        assert batch_size == defi.size()[0]
        desc_init_state = self.initHidden(batch_size)
        desc_output, desc_state = self.encode(desc, desc_init_state)
        def_init_state = self.initHidden(batch_size)
        def_output, def_state = self.encode(defi, def_init_state)
        if args.source == 'description':
            encoder_state = desc_state
        elif args.source == 'definition':
            encoder_state = def_state
        else:
            encoder_state = self.combine(desc_state, def_state)
        return (desc_output, def_output), encoder_state

    def encode(self, input, hidden):
        '''
            input :: bs, sl

            return
                output :: bs, sl, nh*directions
                hidden :: n_layers*directions,bs, nh
        '''
        mask = torch.gt(input.data,0)
        input_length = torch.sum((mask.long()),dim=1)       # batch first = True, (batch, sl)
        lengths, indices = torch.sort(input_length, dim=0, descending=True)
        _, ind = torch.sort(indices, dim=0)
        input_length = torch.unbind(lengths, dim=0)
        embedded = self.embedding(torch.index_select(input,dim=0,index=Variable(indices)))
        output, hidden = self.gru(pack(embedded, input_length, batch_first=True), hidden)
        output = torch.index_select(unpack(output, batch_first=True)[0], dim=0,index=Variable(ind))*Variable(torch.unsqueeze(mask.float(),-1))
        hidden = torch.index_select(hidden[-1], dim=0, index=Variable(ind))
        #hidden = torch.unbind(hidden, dim=0)
        #hidden = torch.cat(hidden, 1)
        direction = 2 if self.bidirectional else 1
        assert hidden.size() == (input.size()[0],self.hidden_size) and output.size() == (input.size()[0], input.size()[1],self.hidden_size*direction)
        return output, hidden

    def initHidden(self, batch_size):
        bid = 2 if self.bidirectional else 1
        result = Variable(torch.zeros(self.n_layers*bid, batch_size, self.hidden_size))
        if use_cuda:
            return result.cuda()
        return result


class Luong_Attn(nn.Module):
    def __init__(self, input_size, hidden_size, direction):
        super(Luong_Attn, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size*direction)
        self.combine = nn.Linear(input_size+hidden_size*direction, hidden_size)

    def forward(self, x, memory):
        '''
            x :: bs, nh
            memory :: bs, sl, nh
        '''
        h1 = self.linear(x)   # bs, nh -> bs, nh
        bil = torch.sum(torch.unsqueeze(h1,1)*memory,-1)    # bs, sl
        score = F.softmax(bil, 1)      # bs, sl
        c = torch.sum(torch.unsqueeze(score, -1)*memory, 1)      # bs, nh
        output = F.tanh(self.combine(torch.cat([x,c],1)))       # bs, nh
        return output, score

class Bah_Attn(nn.Module):
    def __init__(self, input_size, hidden_size, direction):
        super(Bah_Attn, self).__init__()
        self.lin1 = nn.Linear(input_size, hidden_size)
        self.lin2 = nn.Linear(hidden_size*direction, hidden_size)
        self.v = nn.Linear(hidden_size, 1)

    def forward(self, x, memory):
        '''
            x :: bs, nh
            memory :: bs, sl, nh
            v*tanh(W*([x:memory]))
        '''
        h_x = self.lin1(x)      # bs, nh
        h_m = self.lin2(memory)     # bs, sl, nh
        score = self.v(F.tanh(torch.unsqueeze(h_x,1)+h_m))      # bs, sl, nh -> bs, sl, 1
        score = F.softmax(torch.squeeze(score,-1),1)     # bs, sl
        context = torch.sum(torch.unsqueeze(score,-1)*memory,1)     # bs, nh
        return context, score    # bs, sl

class Cover(nn.Module):
    def __init__(self, input_size, hidden_size, direction):
        super(Cover, self).__init__()
        self.lin1 = nn.Linear(input_size, hidden_size)
        self.lin2 = nn.Linear(hidden_size*direction, hidden_size)
        self.lin3 = nn.Linear(1, hidden_size)
        self.v = nn.Linear(hidden_size, 1)
        
    def forward(self, x, memory, cover):
        '''
            x :: bs, nh
            memory :: bs, sl, nh
            cover :: bs, sl
            v*tanh(W*([x:memory:cover*u]))
        '''
        h_x = self.lin1(x)      # bs, nh
        h_m = self.lin2(memory)     # bs, sl, nh
        h_c = self.lin3(torch.unsqueeze(cover, -1))     # bs,sl,1 -> bs, sl, nh
        score = self.v(F.tanh(torch.unsqueeze(h_x,1)+h_m+h_c))      # bs, sl, nh -> bs, sl, 1
        score = F.softmax(torch.squeeze(score,-1),1)     # bs, sl
        context = torch.sum(torch.unsqueeze(score,-1)*memory,1)     # bs, nh
        return context, score    # bs, sl

class MultiLabel(nn.Module):
    def __init__(self, args, voc_size, emb_size, hidden_size):
        super(MultiLabel, self).__init__()
        self.hidden_size = hidden_size
        self.emb_size = emb_size
        self.voc_size = voc_size
        self.embedding = nn.Embedding(self.voc_size, self.emb_size, padding_idx=0)
        self.linear = nn.Linear(self.hidden_size, self.emb_size)
        self.out = nn.Linear(self.hidden_size, self.voc_size)
        self.out_weight = torch.zeros(self.voc_size)
        for i in range(4):
            self.out_weight[i] = 10.**6
        self.out_weight = torch.unsqueeze(self.out_weight,0)
        if use_cuda: self.out_weight = self.out_weight.cuda()

    def forward(self, hidden):
        pred_word = self.out(hidden) - Variable(self.out_weight)
        '''
        h1 = self.linear(hidden)   # bs, nh -> bs, de
        pred_word = torch.sum(torch.unsqueeze(h1,1)*torch.unsqueeze(self.embedding.weight,0),-1) - Variable(self.out_weight)    # bs, nv
        '''
        return pred_word

    def load_emb(self, emb):
        emb = torch.from_numpy(np.array(emb, dtype=np.float32))
        self.embedding.weight.data.copy_(emb)


class Decoder(nn.Module):
    def __init__(self, voc_size, emb_size, hidden_size, bidirectional=False):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.emb_size = emb_size
        self.voc_size = voc_size
        self.bidirectional = bidirectional
        self.direction = 2 if self.bidirectional else 1
        
        self.embedding = nn.Embedding(self.voc_size, self.emb_size, padding_idx=0)
        #self.pretrain_w = nn.Linear(self.emb_size, self.hidden_size)
        self.pretrain_out = nn.Linear(self.hidden_size+self.emb_size, self.voc_size)
        self.gru_cell = nn.GRUCell(self.emb_size, self.hidden_size)
        self.pretrain_encoder = Encoder(self.voc_size, self.emb_size, self.hidden_size)
        #self.attn = Luong_Attn(hidden_size, hidden_size, self.direction)
        self.attn = Bah_Attn(hidden_size+self.emb_size, hidden_size, self.direction)
        self.gate_w1 = nn.Linear(self.hidden_size*2*self.direction, self.hidden_size*self.direction)
        self.gate_w2 = nn.Linear(self.hidden_size*2*self.direction, self.hidden_size*self.direction)
        self.out = nn.Linear(self.hidden_size*(1+self.direction)+self.emb_size, self.voc_size)
        self.out_weight = torch.zeros(self.voc_size)
        self.out_weight[0] = 10.**6
        self.out_weight = torch.unsqueeze(self.out_weight,0)
        if use_cuda: self.out_weight = self.out_weight.cuda()

    def combine(self, desc_output, def_output):
        gate_1 = F.sigmoid(self.gate_w1(torch.cat([desc_output, def_output],1)))
        gate_2 = F.sigmoid(self.gate_w2(torch.cat([desc_output, def_output],1)))
        result = desc_output*gate_1+def_output*gate_2
        return result

    def forward(self, args, word, hidden, encoder_outputs, cover, last_output):
        assert hidden.size() == (word.size()[0], self.hidden_size), (hidden.size(),word.size(),self.hidden_size)
        batch_size = word.size()[0]
        if args.emb == 'word':
            word_vec = self.embedding(word)     #  bs, de
        elif args.emb == 'global':
            word_vec = torch.matmul(last_output, self.embedding.weight) 
        elif args.emb == 'add':
            word_vec = self.embedding(word)+torch.matmul(last_output, self.embedding.weight)     #  bs, de
        hidden = self.gru_cell(word_vec, hidden)
        desc_outputs, def_outputs = encoder_outputs
        c1, s1 = self.attn(torch.cat([hidden, word_vec],1), desc_outputs)     # (bs, nh) (bs, sl, nh) -> (bs, nh)
        c2, s2 = self.attn(torch.cat([hidden, word_vec],1), def_outputs)
        if args.source == 'description':
            context = c1
            score = s1
        elif args.source == 'definition':
            context = c2
            score = s2
        else:
            context = self.combine(c1, c2)
            score = (s1,s2)
        out_hidden = torch.cat([hidden, context, word_vec], 1)
        pred_word = self.out(out_hidden)-Variable(self.out_weight)
        #pred_word = F.softmax(pred_word, 1)
            
        return pred_word, hidden, score

    def load_emb(self, emb):
        emb = torch.from_numpy(np.array(emb, dtype=np.float32))
        self.embedding.weight.data.copy_(emb)

    def pretrain(self, sememes, label_lang):
        batch_size = sememes.size()[0]
        target_length = sememes.size()[1]
        decoder_input = torch.unsqueeze(Variable(torch.LongTensor([label_lang.SOS_token] * batch_size)), -1)
        init_state = self.pretrain_encoder.initHidden(batch_size)
        _, hidden = self.pretrain_encoder.encode(sememes, init_state)
        #hidden = self.pretrain_w(torch.mean(self.embedding(sememes), dim=1))
        loss = 0.
        if use_cuda:
            hidden = hidden.cuda()
            decoder_input = decoder_input.cuda()
        word_vec = self.embedding(torch.cat([decoder_input, sememes], 1))
        all_sememe = F.softmax(multi_hot(sememes, len(label_lang.word2id)), 1)
        for time in range(target_length):
            hidden = self.gru_cell(word_vec[:, time, :], hidden)
            pred_word = self.pretrain_out(torch.cat([hidden, word_vec[:,time,:]],1)) - Variable(self.out_weight)
            if args.soft_loss:
                loss += cross_entropy(pred_word, (multi_hot(torch.unsqueeze(sememes[:,time],1),len(label_lang.word2id))+all_sememe)/2., weight=Variable(self.out_weight))
            else:
                loss += F.cross_entropy(pred_word, sememes[:, time], ignore_index=0)
        return loss


def cross_entropy(prob, targets, weight):
    H = -logsoftmax(prob)*targets
    return torch.sum(H*weight)

def train(args, desc, defi, sememe, encoder, decoder, optimizer, lang):
    optimizer.zero_grad()
    input_length = desc.size()[1]
    target_length = sememe.size()[1]
    batch_size = desc.size()[0]  # type: int
    assert batch_size == sememe.size()[0]
    sememe_mask = (sememe>0).int()
    if use_cuda: sememe_mask = sememe_mask.cuda()
    loss = 0.

    # encoding part
    encoder_output, encoder_state = encoder(args, desc, defi)

    # decoding part
    if args.architecture == 'multi-label':
        output = decoder(encoder_state)
        out_weight = torch.ones(len(lang.word2id))
        for i in range(4):
            out_weight[i] = 0.
        if use_cuda: out_weight = out_weight.cuda()
        loss += F.multilabel_soft_margin_loss(output, multi_hot(sememe, len(lang.word2id)), weight=out_weight)
    elif args.architecture == 'seq2seq':
        out_weight = Variable(torch.ones(len(lang.word2id)))
        out_weight[0] = 0.
        out_weight = torch.unsqueeze(out_weight, 0)
        if use_cuda: out_weight = out_weight.cuda()
        decoder_input = Variable(torch.LongTensor([lang.SOS_token]*batch_size))
        decoder_hidden = torch.squeeze(encoder_state,0) if encoder_state.dim()==3 else encoder_state
        last_output = Variable(torch.zeros((batch_size, len(lang.word2id))))
        if use_cuda:
            decoder_input = decoder_input.cuda()
            last_output = last_output.cuda()
        history_outputs = torch.zeros((batch_size, len(lang.word2id)))
        all_sememe = F.softmax(multi_hot(sememe, len(lang.word2id)), 1)
        if use_cuda: history_outputs = history_outputs.cuda()
        for time in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(args,
                decoder_input, decoder_hidden, encoder_output, Variable(history_outputs/float(time+0.00001)), last_output)
            history_outputs = history_outputs+decoder_output.data
            last_output = F.softmax(decoder_output, 1)
            if args.soft_loss:
                loss += cross_entropy(decoder_output, (multi_hot(torch.unsqueeze(sememe[:,time],1),len(lang.word2id))+all_sememe)/2., weight=out_weight)
            else:
                loss += F.cross_entropy(decoder_output, sememe[:,time], ignore_index=0)
            decoder_input = sememe[:,time]
        loss = loss/torch.sum(sememe_mask.float())
    loss.backward()
    optimizer.step()

    return loss.data

def evaluate(args, encoder, decoder, batch, lang, is_test=False):
    desc, defi, sememe = batch
    input_length = desc.size()[1]
    batch_size = desc.size()[0]
    encoder_output, encoder_state = encoder(args, desc, defi)

    if args.architecture == 'multi-label' or args.architecture == 'hmulti-label':
        if args.architecture == 'multi-label':
            decoded_words = F.softmax(decoder(encoder_state),1)
        elif args.architecture == 'hmulti-label':
            decoded_words, _ = decoder(encoder_state, encoder_output)
            decoded_words = F.softmax(decoded_words, 1)
        pred_sememes = []
        for word in decoded_words:
            if not is_test:
                topv,topi = word.topk(MAX_LENGTH)
            else:
                topv,topi = word.topk(20)
            results = []
            for v,i in zip(topv,topi):
                if is_test:
                    results.append(i.data[0])
                elif v.data[0] > args.threshold:
                    results.append(i.data[0])
            pred_sememes.append(results)
        return pred_sememes
    elif args.architecture == 'seq2seq':
        decoder_input = Variable(torch.LongTensor([lang.SOS_token]*batch_size))
        decoder_hidden = torch.squeeze(encoder_state,0) if encoder_state.dim()==3 else encoder_state
        last_output = Variable(torch.zeros((batch_size, len(lang.word2id))))
        if use_cuda:
            decoder_input = decoder_input.cuda()
            last_output = last_output.cuda()
        decoded_words = []
        history_outputs = torch.zeros((batch_size, len(lang.word2id)))
        if use_cuda: history_outputs = history_outputs.cuda()
        for time in range(MAX_LENGTH):
            decoder_output, decoder_hidden, decoder_attention = decoder(args,
                decoder_input, decoder_hidden, encoder_output, Variable(history_outputs/float(time+0.00001)), last_output)
            decoder_output = F.softmax(decoder_output, 1)
            last_output = decoder_output
            history_outputs = history_outputs + decoder_output.data
            topv, topi = decoder_output.data.topk(1)
            decoder_input = Variable(torch.squeeze(topi))
            if use_cuda: decoder_input = decoder_input.cuda()
            decoded_words.append(torch.squeeze(topi))
        decoded_words = torch.stack(decoded_words,1)
        assert decoded_words.size() == (batch_size, MAX_LENGTH)
        return decoded_words.cpu().numpy()

def padding(sequence, length):
    sequence = sequence[:length]
    while len(sequence) < length:
        sequence.append(0)
    return sequence
         
def make_batches(data, batch_size, test_words, is_test=False):
    batch_num = len(data)//batch_size if len(data)%batch_size==0 else len(data)//batch_size+1
    batches = []
    for batch in range(batch_num):
        mini_batch = data[batch*batch_size:(batch+1)*batch_size] 
        max_desc_len = max([len(item['description']) for item in mini_batch if is_test or item['word_raw'] not in test_words])
        max_def_len = max([len(item['definition']) for item in mini_batch if is_test or item['word_raw'] not in test_words])
        desc_batch = [] 
        def_batch = []
        sememe_batch = []
        word_batch = []
        for item in mini_batch:
            if not is_test and item['word_raw'] in test_words:
                continue
            description = item['description']
            definition = item['definition']
            desc = padding(description, length=max_desc_len)
            defi = padding(definition, length=max_def_len)
            sememe = padding(item['sense'], length=MAX_LENGTH)
            desc_batch.append(np.array(desc, dtype=np.long))
            def_batch.append(np.array(defi, dtype=np.long))
            sememe_batch.append(np.array(sememe, dtype=np.long))
            word_batch.append(item['word'])
        batches.append((np.array(desc_batch, dtype=np.long), np.array(def_batch, dtype=np.long), np.array(sememe_batch, dtype=np.long),word_batch))
    return batches


def convert_result(args, result, lang):
    l = []
    for sense in result:
        result = []
        for i in range(len(sense)):
            if sense[i] == lang.EOS_token:
                break
            elif sense[i] < 4:
                continue
            result.append(lang.id2word[sense[i]])
        l.append(result)
    return l

def convert_to_json(words, descriptions, definitions, reference, candidate, log_path):
    data = []
    for w,des,df, r, c in zip(words, descriptions, definitions, reference, candidate):
        data.append({'word':w, 'description':(''.join(des)).replace('\n',''), 'definition':(''.join(df)).replace('\n',''), 'reference':r, 'candidate':c})
    json.dump(data, open(log_path+'/result.json','w'))

def evaluate_all(args, data, encoder, decoder, label_lang, src_lang, is_test=False):
    hypothsis = []
    reference = []
    descriptions = []
    definitions = []
    words = []
    for batch in tqdm(data,disable=not args.verbose):
        desc, defi, sememe, word = batch
        desc_variable = Variable(torch.from_numpy(desc))
        def_variable = Variable(torch.from_numpy(defi))
        sememe_variable = Variable(torch.from_numpy(sememe))
        if use_cuda:
            desc_variable = desc_variable.cuda()
            sememe_variable = sememe_variable.cuda()
            def_variable = def_variable.cuda()
        result = evaluate(args, encoder, decoder, (desc_variable, def_variable, sememe_variable),  lang=label_lang, is_test=is_test)
        mask = [[word>0 for word in sent] for sent in sememe]
        hypothsis.extend(convert_result(args, result, label_lang))
        reference.extend(convert_result(args, sememe, label_lang))
        descriptions.extend(convert_result(args, desc, src_lang))
        definitions.extend(convert_result(args, defi, src_lang))
        words.extend(convert_result(args, word, src_lang))

    #convert_to_json(words, descriptions, definitions, reference, hypothsis, log_path='./log/'+args.log_dir)
    precision, recall, F1 = eval_F(reference, hypothsis, log_path='./log/'+args.log_dir, descriptions=descriptions, definitions=definitions, words=words)
    mean_ap = eval_map(reference, hypothsis)
    return precision, recall, F1, mean_ap


def pretrain(args, decoder, label_lang, test_words):
    print('pre-training', flush=True)
    sememe_data = []
    for line in open('../dataset/word_sememes.txt'):
        tem = line.strip().split('\t')
        if len(tem) < 2: continue
        word, sememes = tem
        if word in test_words:
            continue
        sememe_data.append(sememes.split(' '))
    batches = []
    batch_size = args.batch_size
    for i in range(0, len(sememe_data), batch_size):
        mini_batch = sememe_data[i:i + batch_size]
        m_len = max([len(s) for s in mini_batch])
        batch_data = []
        for s in mini_batch:
            sememes = [label_lang.word2id[sememe] for sememe in s]
            while len(sememes) < m_len:
                sememes.append(label_lang.word2id['PADDING'])
            batch_data.append(sememes)
        batches.append(np.array(batch_data, dtype=np.long))
    parameters = decoder.parameters()
    optimizer = optim.Adam(parameters)
    for iter in range(1):
        for batch in tqdm(batches, disable=not args.verbose):
            sememe_variable = Variable(torch.from_numpy(batch))
            if use_cuda: sememe_variable = sememe_variable.cuda()
            loss = decoder.pretrain(sememe_variable, label_lang)
            loss.backward()
            optimizer.step()

def self_learn_data(data, chinese, sememe_voc):
    infer_data = []
    for item in data:
        data_item = {}
        data_item['word'] = chinese.sent2id(item[0])
        data_item['word_raw'] = item[0]
        data_item['description'] = chinese.sent2id(item[1], min_count=2, add_eos=True)
        data_item['definition'] = chinese.sent2id('', min_count=2, add_eos=True)
        data_item['sense'] = sememe_voc.sent2id(item[2], min_count=5, add_eos=True)
        infer_data.append(data_item)
    return infer_data

def convert_pad(sentences, lang):
    m_len = max([len(s) for s in sentences])
    batch = []
    for s in sentences:
        sentence = []
        for word in s:
            if word in lang.word2id:
                sentence.append(lang.word2id[word])
            else:
                sentence.append(lang.word2id['OOV'])
        while len(sentence) < m_len:
            sentence.append(lang.word2id['PADDING'])
        batch.append(sentence)
    return batch

def self_learn(args, encoder, decoder, label_lang, threshold):
    word_set = json.load(open('../dataset/duoyi_test.json'))
    result = []
    if args.debug:
        word_set = word_set[:100]
    for word_object in tqdm(word_set, disable=not args.verbose):
        word = word_object[0]
        descriptions = word_object[2]
        senses = [sense.split(':')[1].split(';') for sense in word_object[1]]
        description_vec = np.array(convert_pad(descriptions, chinese), dtype=np.long)
        desc_variable = Variable(torch.from_numpy(description_vec))
        if use_cuda: desc_variable = desc_variable.cuda()
        defi_variable = torch.ones_like(desc_variable)
        input_length = desc_variable.size()[1]
        batch_size = desc_variable.size()[0]
        encoder_output, encoder_state = encoder(args, desc_variable, defi_variable)
        decoder_input = Variable(torch.LongTensor([label_lang.SOS_token]*batch_size))
        decoder_hidden = torch.squeeze(encoder_state,0) if encoder_state.dim()==3 else encoder_state
        last_output = Variable(torch.zeros((batch_size, len(sememe_voc.word2id))))
        if use_cuda:
            decoder_input = decoder_input.cuda()
            last_output = last_output.cuda()
        decoded_words = []
        history_outputs = torch.zeros((batch_size, len(sememe_voc.word2id)))
        if use_cuda: history_outputs = history_outputs.cuda()
        for time in range(MAX_LENGTH):
            decoder_output, decoder_hidden, decoder_attention = decoder(args,
                decoder_input, decoder_hidden, encoder_output, Variable(history_outputs/float(time+0.00001)), last_output)
            decoder_output = F.softmax(decoder_output, 1)
            last_output = decoder_output
            history_outputs = history_outputs + decoder_output.data
            topv, topi = decoder_output.data.topk(1)
            decoder_input = Variable(torch.squeeze(topi))
            if use_cuda: decoder_input = decoder_input.cuda()
            decoded_words.append(torch.squeeze(topi))
        decoded_words = torch.stack(decoded_words,1)
        decoded_words = decoded_words.cpu().numpy().tolist()
        for sense in senses:
            m_sim = (-1,-1)
            for i,d,p in zip(range(len(descriptions)), descriptions, decoded_words):
                similarity = len(set(sense).intersection({sememe_voc.id2word[sememe] for sememe in p}))
                if similarity > m_sim[0]:
                    m_sim = (similarity, i)
            if m_sim[0]/float(len(sense)) >= threshold:
                result.append((word, descriptions[m_sim[1]], sense))
    return result
    
def train_iters(args, encoder, decoder, batches, dev_data, test_data, n_iters, label_lang, src_lang, best_F = 0.):
    start_time = time.time()
    parameters = [p for p in encoder.parameters()] + [p for p in decoder.parameters()]
    optimizer = optim.Adam(parameters)

    best_epoch = -1
    for iter in range(1, n_iters+1):
        total_loss = 0.
        if args.debug:
            batches = batches[:2]
        for batch in tqdm(batches, disable=not args.verbose):
            desc, defi, sememe, word = batch
            desc_variable = Variable(torch.from_numpy(desc))
            def_variable = Variable(torch.from_numpy(defi))
            sememe_variable = Variable(torch.from_numpy(sememe))
            if use_cuda:
                desc_variable = desc_variable.cuda()
                sememe_variable = sememe_variable.cuda()
                def_variable = def_variable.cuda()
            loss = train(args, desc_variable, def_variable, sememe_variable, encoder, decoder, optimizer, lang=label_lang)
            total_loss += loss
        print('epoch %d loss %.2f'%(iter,total_loss[0]))
        precision, recall, F1, mean_ap = evaluate_all(args, dev_data, encoder, decoder, label_lang=label_lang, src_lang=src_lang)
        print('precision %.2f, recall %.2f, F value %.2f, MAP %.4f'%(precision, recall, F1, mean_ap), flush=True)
        if F1 > best_F:
            best_epoch = iter
            best_F = F1
            torch.save(encoder.state_dict(),'./models/'+args.log_dir+'_encoder.pt',pickle_protocol=3)
            torch.save(decoder.state_dict(),'./models/'+args.log_dir+'_decoder.pt',pickle_protocol=3)
    encoder.load_state_dict(torch.load('./models/'+args.log_dir+'_encoder.pt'))
    decoder.load_state_dict(torch.load('./models/'+args.log_dir+'_decoder.pt'))
    print('best epoch %d, best F value %.2f'%(best_epoch, best_F), flush=True)
    test_precision, test_recall, test_F, test_map = evaluate_all(args, test_data, encoder, decoder, label_lang=label_lang,src_lang=src_lang, is_test=False)
    print('test precision %.2f, recall %.2f, F value %.2f, MAP %.4f'%(test_precision, test_recall, test_F, test_map), flush=True)
    return best_F


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb_size',default=200,type=int)
    parser.add_argument('--hidden_size',default=300,type=int)
    parser.add_argument('--batch_size',default=20, type=int)
    parser.add_argument('-e','--epoch_num',default=10, type=int)
    parser.add_argument('--threshold',default=0.5,type=float)
    parser.add_argument('--bidirectional',default=True,action='store_true')
    parser.add_argument('--gpu',default=0,type=int)
    parser.add_argument('-v','--verbose',default=False, action='store_true')
    parser.add_argument('-d','--debug',default=False, action='store_true')
    parser.add_argument('-c','--cover',default=False, action='store_true')
    parser.add_argument('-s','--share_outlayer',default=False, action='store_true')
    parser.add_argument('--probsum',default=False, action='store_true')
    parser.add_argument('--soft_align',default=False, action='store_true')
    parser.add_argument('--multi_loss',default=False, action='store_true')
    parser.add_argument('--layer_num',default=1,type=int)
    ##################################################################################
    parser.add_argument('-l','--log_dir',default='seq2seq',type=str)
    parser.add_argument('--soft_loss',default=False, action='store_true')
    parser.add_argument('--emb',default='word',choices=['word','global','add'])
    parser.add_argument('--source',default='any',choices=['any','combine','description','definition'])
    parser.add_argument('--architecture',default='seq2seq',choices=['hmulti-label','multi-label', 'seq2seq'])
    parser.add_argument('--pretrain', default=False, action='store_true')
    parser.add_argument('--self_learn', default=False, action='store_true')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    if use_cuda: torch.cuda.set_device(args.gpu)
    chinese, sememe_voc, sememe_emb = pickle.load(open('chinese_sememe_voc.pkl','rb'))
    data, description_data, definition_data, combine_data = pickle.load(open('sememe_prediction_data.pkl','rb'))
    train_data, dev_data, test_data = combine_data
    test_words = {word_object['word_raw'] for word_object in test_data+dev_data}
    encoder = Encoder(voc_size=len(chinese.word2id), emb_size=args.emb_size, hidden_size=args.hidden_size, n_layers=args.layer_num, bidirectional=args.bidirectional)
    if args.architecture == 'multi-label':
        decoder = MultiLabel(args, voc_size=len(sememe_voc.word2id), emb_size=args.emb_size, hidden_size=args.hidden_size)
        decoder.load_emb(sememe_emb)
    elif args.architecture == 'seq2seq':
        decoder = Decoder(voc_size=len(sememe_voc.word2id), hidden_size=args.hidden_size, emb_size=args.emb_size, bidirectional=args.bidirectional)
        decoder.load_emb(sememe_emb)
    if use_cuda:
        encoder.cuda()
        decoder.cuda()
    any_batches = make_batches(data, args.batch_size, test_words, is_test=False)
    description_batches = make_batches(description_data, args.batch_size, test_words)
    definition_batches = make_batches(definition_data, args.batch_size, test_words)
    train_batches = make_batches(train_data, args.batch_size, test_words, is_test=False)
    dev_batches = make_batches(dev_data, args.batch_size, test_words, is_test=True)
    test_batches = make_batches(test_data, args.batch_size, test_words, is_test=True)
    best_F = 0.
    if args.pretrain:
        pretrain(args, decoder, sememe_voc, test_words)
    if args.source == 'combine':
        best_F = train_iters(args, encoder, decoder, train_batches, dev_batches, test_batches, args.epoch_num, label_lang=sememe_voc, src_lang=chinese)
    elif args.source == 'description':
        best_F = train_iters(args, encoder, decoder, description_batches, dev_batches, test_batches, args.epoch_num, label_lang=sememe_voc, src_lang=chinese)
    elif args.source == 'definition':
        best_F = train_iters(args, encoder, decoder, definition_batches, dev_batches, test_batches, args.epoch_num, label_lang=sememe_voc, src_lang=chinese)
    elif args.source == 'any':
        best_F = train_iters(args, encoder, decoder, any_batches, dev_batches, test_batches, args.epoch_num, label_lang=sememe_voc, src_lang=chinese)
    if args.self_learn:
        print('self learning stage')
        for i in range(5):
            infer_data = self_learn(args, encoder, decoder, sememe_voc, threshold=0.5)
            infer_data = self_learn_data(infer_data, chinese, sememe_voc)
            infer_batches = make_batches(infer_data, args.batch_size, test_words, is_test=True)
            train_batches = any_batches+infer_batches
            random.shuffle(train_batches)
            improved = False
            for j in range(5):
                F1 = train_iters(args, encoder, decoder, train_batches, dev_batches, test_batches, n_iters=1, label_lang=sememe_voc, src_lang=chinese, best_F=best_F)
                if F1 > best_F:
                    best_F = F1
                    improved = True
                    break
            if improved:
                break

        