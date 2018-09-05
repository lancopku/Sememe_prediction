from seq2seq import *
from corpus import *

use_cuda = torch.cuda.is_available()

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

if __name__ == '__main__':
    args = parse_args()
    chinese, sememe_voc, sememe_emb = pickle.load(open('chinese_sememe_voc.pkl','rb'))
    encoder = Encoder(voc_size=len(chinese.word2id), emb_size=args.emb_size, hidden_size=args.hidden_size, n_layers=args.layer_num, bidirectional=args.bidirectional)
    decoder = Decoder(voc_size=len(sememe_voc.word2id), hidden_size=args.hidden_size, emb_size=args.emb_size, bidirectional=args.bidirectional)
    if use_cuda:
        encoder.cuda()
        decoder.cuda()
    encoder.load_state_dict(torch.load('./models/'+args.log_dir+'_encoder.pt'))
    decoder.load_state_dict(torch.load('./models/'+args.log_dir+'_decoder.pt'))
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
        decoder_input = Variable(torch.LongTensor([sememe_voc.SOS_token]*batch_size))
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
        '''
        print(' '.join([sememe_voc.id2word[word] for word in decoded_words[0] if word > 3]))
        print(' ')
        '''
        for sense in senses:
            m_sim = (-1,-1)
            for i,d,p in zip(range(len(descriptions)), descriptions, decoded_words):
                similarity = len(set(sense).intersection({sememe_voc.id2word[sememe] for sememe in p}))
                if similarity > m_sim[0]:
                    m_sim = (similarity, i)
            if m_sim[0]/float(len(sense)) >= args.threshold:
                result.append((word, descriptions[m_sim[1]], sense))
    json.dump(result, open('duoyi_alignment.json','w'), indent=4, ensure_ascii=False)

