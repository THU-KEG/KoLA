from tqdm import tqdm
from Levenshtein import distance
from scipy.optimize import linear_sum_assignment
from transformers import GPT2Tokenizer
import numpy as np
import fire
special_tokens = ["<spt>","<ans>",]

label_names = ['input','output']

class Metric():
    def __init__(self,tokenizer=None) -> None:
        if tokenizer is None:
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer = tokenizer
    
    @staticmethod
    def match(pred_list:list, tgt_list:list):
        distance_mat = np.zeros((len(pred_list),len(tgt_list)))
        for i,pred in enumerate(pred_list):
            for j,tgt in enumerate(tgt_list):
                distance_mat[i,j] = distance(pred,tgt)

        row_ind, col_ind = linear_sum_assignment(distance_mat)

        # not allow extra num predict
        penalty1 = len(tgt_list) / len(pred_list) if len(pred_list) > len(tgt_list) else len(pred_list) / len(tgt_list)
        # not allow multiple prediction to point same tgt
        penalty2 = 1.0 # drop

        return [(tgt_list[tgt],pred_list[pred]) for tgt,pred in zip(col_ind,row_ind)], penalty1, penalty2

    @staticmethod
    def seq_f1(y_pred,y_tgt):
        """
        :param y_pred: [n_samples]
        :param y_tgt: [n_samples]
        :return: 严格F1(exact match)和松弛F1(字符匹配率)
        """
        exact_match_cnt = 0
        token_match_cnt = 0
        token_pred_sum = token_tgt_sum = 0
        for pred,tgt in zip(y_pred,y_tgt):
            if pred == tgt:
                exact_match_cnt += 1
            
            pred_input_ids = Metric().tokenizer(pred)['input_ids']
            tgt_input_ids = Metric().tokenizer(tgt)['input_ids']

            token_pred_sum += len(pred_input_ids)
            token_tgt_sum += len(tgt_input_ids)
            
            for pred_idx in pred_input_ids:
                if pred_idx in tgt_input_ids:
                    token_match_cnt += 1
        em_acc = exact_match_cnt / (len(y_tgt)+0.001)
        token_acc = token_match_cnt / (token_pred_sum+0.001)
        token_recall = token_match_cnt / (token_tgt_sum+0.001)
        token_f1 = 0
        if token_acc + token_recall != 0:
            token_f1 = 2 * token_acc * token_recall / (token_acc + token_recall)

        return em_acc,token_f1
    
    def metric(self,eval_predict):
        predictions = eval_predict.predictions
        label_ids = eval_predict.label_ids

        em_cnt = 0
        pred_str_list = []
        tgt_str_list = []
        for pred,label in tqdm(zip(predictions,label_ids),total=len(predictions)):
            pred_str = self.tokenizer.decode(pred[pred>0],skip_special_tokens=True)
            label_str = self.tokenizer.decode(label[label>0],skip_special_tokens=True)
            if pred_str == label_str:
                em_cnt+=1
            
            pred_str_list.append(pred_str.split(' <ans> '))
            tgt_str_list.append(label_str.split(' <ans> '))
        
        metric_dict = self.str_metric(pred_str_list,tgt_str_list)
        metric_dict['acc'] = float(em_cnt/len(predictions))
        
        print(pred_str,label_str)
        print(metric_dict)
        return metric_dict
    
    @staticmethod
    def str_metric(pred_str_list,tgt_str_list):    
        em_acc_total = 0
        token_f1_total = 0
        for pred, tgt in zip(pred_str_list, tgt_str_list):
            if len(pred) == 0:
                if (len(tgt) == 1 and len(tgt[0]) == 0) or len(tgt) == 0:
                    em_acc_total += 1
                    token_f1_total += 1
                else:
                    em_acc_total += 0
                    token_f1_total += 0
                continue
            pair,p1,p2 = Metric.match(pred,tgt)
            # print(pair)
            # print(p1)
            # print(p2)
            # print()
            em_acc,token_f1 = Metric.seq_f1([unit[1] for unit in pair],[unit[0] for unit in pair])
            em_acc_total += (em_acc * p1 * p2)
            token_f1_total += (token_f1 * p1 * p2)

        return {
            'em_acc_with_penalty':float(em_acc_total/len(pred_str_list)),
            'token_f1_with_penalty':float(token_f1_total/len(pred_str_list))
        }




def find_and_split(ipt_str):
    if len(ipt_str) > 1 and ipt_str.find("<stop>") < 0:
        ipt_str = ipt_str + "<stop>"
    elif len(ipt_str) < 1:
        return []

    if ipt_str.find("<stop>") > 0:
        position = ipt_str.find("<stop>")
        pst_str = ipt_str[:position]
        pst_str = [x for x in pst_str.split("\", \"")]
        pst_str[-1] = pst_str[-1].strip()
        if pst_str[-1][-1] == "\"":
            pst_str[-1] = pst_str[-1][:-1]
        if pst_str[0][0] == "\"":
            pst_str[0] = pst_str[0][1:]
        return pst_str
    
    elif ipt_str.find("<stop>") == 0:
        return []


import json


def run_eval(file_name, output_file):
    f = open(file_name)
    data = json.load(f)

    pred = [find_and_split(x["request"]['result']["completions"][0]["text"].strip()) for x in data["request_states"]]
    gold = [find_and_split(x["instance"]["references"][0]["output"]["text"].strip()) for x in data["request_states"]]

    metric = Metric()
    result = metric.str_metric(pred, gold)
    print(result)
    json.dump(result, open(output_file, "w"), indent = 2)

fire.Fire(run_eval)