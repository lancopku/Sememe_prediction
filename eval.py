import sys

def eval_f(pred_has_answer,has_answer,pred_answers,answers,threshold=0.3):
    assert len(pred_has_answer)==len(has_answer) and len(has_answer)==len(pred_answers) and len(pred_answers)==len(answers)
    right = 0
    pre_total = 0
    rec_total = 0
    for i in range(len(has_answer)):
        if pred_has_answer[i]>threshold:
            pre_total += 1
        if has_answer[i] == True:
            rec_total += 1
        if pred_has_answer[i]>threshold and has_answer[i] == True and answers[i][pred_answers[i]]==1:
            right += 1
    if pre_total == 0:
        pre_total = 1
    precision = right/float(pre_total)
    recall = right/float(rec_total)
    if precision+recall == 0:
        return 0,0,0
    F1 = (2*precision*recall)/(precision+recall)
    return precision,recall,F1

def rank(probs, sn, qid, did):
    result = []
    for p,n,q,d in zip(probs,sn,qid,did):
        index_p = zip(p[:n],d[:n])
        assert n >= len(d) or d[n]=='None'
        new_index = sorted(index_p,key=lambda x:x[0],reverse=True)
        for ind,i in zip(new_index, range(1,n+1)):
            result.append((q,ind[1],i))
    return result

def eval_map(answer_list, gold_file, total_num):
    dic = {}

    fin = open(gold_file)
    for line in fin:
        line = line.strip()
        if not line:
            continue
        cols = line.split('\t')
        if cols[0] == 'QuestionID':
            continue

        q_id = cols[0]
        a_id = cols[4]
        
        if not q_id in dic:
            dic[q_id] = {}
        dic[q_id][a_id] = [cols[6],-1]
    fin.close()

    '''
    fin = open(answer_file)
    for line in fin:
        line = line.strip()
        if not line:
            continue
        cols = line.split('\t')
        q_id = cols[0]
        a_id = cols[1]
        rank = int(cols[2])
        dic[q_id][a_id][1] = rank
    fin.close()
    '''
    for item in answer_list:
        q_id = item[0]
        a_id = item[1]
        rank = item[2]
        dic[q_id][a_id][1] = rank


    MAP = 0.0
    MRR = 0.0
    for q_id in dic:
        sort_rank = sorted(dic[q_id].items(), key = lambda asd:asd[1][1], reverse = False)
        correct = 0
        total = 0
        AP = 0.0
        mrr_mark = False
        for i in range(len(sort_rank)):
            #compute MRR
            if sort_rank[i][1][0] == '1' and mrr_mark == False:
                MRR += 1.0 / float(i + 1)
                mrr_mark = True

            #compute MAP
            total += 1
            if sort_rank[i][1][0] == '1':
                correct += 1
                AP += float(correct) / float(total)
        if correct != 0:
            AP /= float(correct)
        MAP += AP

    #MAP /= float(len(dic))
    #MRR /= float(len(dic))
    
    return MAP/float(total_num), MRR/float(total_num)

if __name__ == "__main__":
    
    if len(sys.argv) != 3:
        print("Usage: python eval.py <your_predict_file> <origin_data_file>")
        exit(0)
    
    answer_file = sys.argv[1]
    gold_file = sys.argv[2]
    MAP,MRR = eval(answer_file, gold_file)
    print("Final Evaluation Score:")
    print("MAP:", MAP)
    print("MRR", MRR)

    exit(0)
    
