# coding=utf-8
import json
import sys
import copy
# coding=utf-8
import warnings
import numpy as np
from typing import List
from collections import Counter
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from copy import deepcopy
# from bert_score import score
import torch
from rouge import Rouge


class Ngrams(object):
    """
        Ngrams datastructure based on `set` or `list`
        depending in `exclusive`
    """

    def __init__(self, ngrams={}, exclusive=True):
        if exclusive:
            self._ngrams = set(ngrams)
        else:
            self._ngrams = list(ngrams)
        self.exclusive = exclusive

    def add(self, o):
        if self.exclusive:
            self._ngrams.add(o)
        else:
            self._ngrams.append(o)

    def __len__(self):
        return len(self._ngrams)

    def intersection(self, o):
        if self.exclusive:
            inter_set = self._ngrams.intersection(o._ngrams)
            return Ngrams(inter_set, exclusive=True)
        else:
            other_list = deepcopy(o._ngrams)
            inter_list = []

            for e in self._ngrams:
                try:
                    i = other_list.index(e)
                except ValueError:
                    continue
                other_list.pop(i)
                inter_list.append(e)
            return Ngrams(inter_list, exclusive=False)

    def union(self, *ngrams):
        if self.exclusive:
            union_set = self._ngrams
            for o in ngrams:
                union_set = union_set.union(o._ngrams)
            return Ngrams(union_set, exclusive=True)
        else:
            union_list = deepcopy(self._ngrams)
            for o in ngrams:
                union_list.extend(o._ngrams)
            return Ngrams(union_list, exclusive=False)

def my_lcs(string, sub):
    """
    Calculates longest common subsequence for a pair of tokenized strings
    :param string : list of str : tokens from a string split using whitespace
    :param sub : list of str : shorter string, also split using whitespace
    :returns: length (list of int): length of the longest common subsequence between the two strings

    Note: my_lcs only gives length of the longest common subsequence, not the actual LCS
    """
    if len(string) < len(sub):
        sub, string = string, sub

    lengths = [[0 for _ in range(0,len(sub)+1)] for _ in range(0,len(string)+1)]

    for j in range(1,len(sub)+1):
        for i in range(1, len(string) + 1):
            if string[i - 1] == sub[j - 1]:
                lengths[i][j] = lengths[i-1][j-1] + 1
            else:
                lengths[i][j] = max(lengths[i-1][j] , lengths[i][j-1])

    return lengths[len(string)][len(sub)]


class Metric(object):
    def __init__(self, toker):
        self.refs = []
        self.hyps = []
        self.toker = toker

    def forword(self, refs: List[List[str]], hyp: List[str]): # TODO: only applicable to token ids
        self.refs.append(refs)
        self.hyps.append(hyp)

    def forstr(self, refs: List[str], hyp: str): 
        self.refs.append(refs)  # 人的回答
        self.hyps.append(hyp)  # ai的回答
        # print("append refs = ", refs)
        # print("append hyps = ", hyp)

    def calc_bleu_k(self, k):  # k-gram
        weights = [1. / k] * k + (4 - k) * [0.]
        try:
            bleu = corpus_bleu(self.refs, self.hyps, weights=weights,
                               smoothing_function=SmoothingFunction().method3)
        except ZeroDivisionError as _:
            warnings.warn('the bleu is invalid')
            bleu = 0.
        return bleu

    def calc_distinct_k(self, k):
        d = {}
        tot = 0
        for sen in self.hyps:
            for i in range(0, len(sen)-k):
                key = tuple(sen[i:i+k])
                d[key] = 1
                tot += 1
        if tot > 0:
            dist = len(d) / tot
        else:
            warnings.warn('the distinct is invalid')
            dist = 0.
        return dist

    def calc_unigram_f1(self):
        f1_scores = []
        for hyp, refs in zip(self.hyps, self.refs):
            scores = []
            for ref in refs:
                if(len(ref) > 0):
                    cross = Counter(hyp) & Counter(ref)
                    cross = sum(cross.values())
                    p = cross / max(len(hyp), 1e-10)  # precision
                    # print("ref = ", ref)
                    r = cross / len(ref)  # recall
                    f1 = 2 * p * r / max(p + r, 1e-10)
                    scores.append(f1)
            f1_scores.append(max(scores))
        return np.mean(f1_scores), f1_scores

    # def calc_f1_with_bleu_rouge(self):
    #     f1_scores = []
    #     for hyp, refs in zip(self.hyps, self.refs):
    #         scores = []
    #         for ref in refs:
    #             if(len(ref) > 0):
    #                 cross = Counter(hyp) & Counter(ref)
    #                 cross = sum(cross.values())
    #                 p = cross / max(len(hyp), 1e-10)  # precision
    #                 # print("ref = ", ref)
    #                 r = cross / len(ref)  # recall
    #                 f1 = 2 * p * r / max(p + r, 1e-10)
    #                 scores.append(f1)
    #         f1_scores.append(max(scores))
    #     return np.mean(f1_scores), f1_scores

    def calc_rouge_l(self, beta=1.2):
        scores = []
        for hyp, refs in zip(self.hyps, self.refs):
            prec = []
            rec = []
            for ref in refs:
                if(len(ref) > 0):
                    lcs = my_lcs(ref, hyp)
                    prec.append(lcs / max(len(hyp), 1e-10))
                    rec.append(lcs / len(ref))
            prec_max = max(prec)
            rec_max = max(rec)
            if prec_max != 0 and rec_max !=0:
                score = ((1 + beta**2)*prec_max*rec_max)/float(rec_max + beta**2*prec_max)
            else:
                score = 0.0
            scores.append(score)
        return np.mean(scores), scores

    def _get_ngrams(self, n, text, exclusive=True):
        """Calcualtes n-grams.
        Args:
          n: which n-grams to calculate
          text: An array of tokens
        Returns:
          A set of n-grams
        """
        ngram_set = Ngrams(exclusive=exclusive)
        text_length = len(text)
        max_index_ngram_start = text_length - n
        for i in range(max_index_ngram_start + 1):
            ngram_set.add(tuple(text[i:i + n]))
        return ngram_set

    def _get_word_ngrams(self, n, sentences, exclusive=True):
        """Calculates word n-grams for multiple sentences.
        """
        assert len(sentences) > 0
        assert n > 0
        words = [x for y in sentences for x in y] # flatten the sentences
        return self._get_ngrams(n, words, exclusive=exclusive)

    def f_r_p_rouge_n(self, evaluated_count, reference_count, overlapping_count):
        # Handle edge case. This isn't mathematically correct, but it's good enough
        if reference_count == 0:
            recall = 0.0
        else:
            recall = overlapping_count / reference_count

        return recall

    def calc_rouge_n(self, n=2, exclusive=True):
        """
        Computes ROUGE-N of two text collections of sentences.
        Sourece: http://research.microsoft.com/en-us/um/people/cyl/download/
        papers/rouge-working-note-v1.3.1.pdf
        Args:
          evaluated_sentences: The sentences that have been picked by the
                               summarizer
          reference_sentences: The sentences from the referene set
          n: Size of ngram.  Defaults to 2.
        Returns:
          A tuple (f1, precision, recall) for ROUGE-N
        Raises:
          ValueError: raises exception if a param has len <= 0
        """
        if len(self.hyps) <= 0:
            raise ValueError("Hypothesis is empty.")
        if len(self.refs) <= 0:
            raise ValueError("Reference is empty.")
        
        # evaluated_ngrams = self._get_word_ngrams(n, self.hyps, exclusive=exclusive)
        # refs = [x[0] for x in self.refs]
        # reference_ngrams = self._get_word_ngrams(n, refs, exclusive=exclusive)
        # reference_count = len(reference_ngrams)
        # evaluated_count = len(evaluated_ngrams)

        # # Gets the overlapping ngrams between evaluated and reference
        # overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
        # overlapping_count = len(overlapping_ngrams)

        # return self.f_r_p_rouge_n(evaluated_count, reference_count, overlapping_count)
    def calc_mauve_score(self):
        label = [i[0] for i in self.refs]
        generated = self.hyps
        import mauve
        return mauve.compute_mauve(p_text=label, q_text=generated, device_id=0, max_text_length=2048, verbose=False).mauve
    
    def calc_bert_score(self):
        try:
            P, R, F1 = score(self.hyps, self.refs, model_type='/data2/cookie_huggingface_models/bert-base-uncased', verbose=True)
        except Exception:
            F1 = torch.randn(3)
        # print("P = ", torch.mean(P), " R = ", torch.mean(R), " F1 = ", torch.mean(F1).item())
        bert_score = torch.mean(F1).item()
        return bert_score

    def close(self):
        result = {
            **{f"dist-{k}": 100 * self.calc_distinct_k(k) for k in range(3, 5)},
            **{f"bleu-{k}": 100 * self.calc_bleu_k(k) for k in range(4, 5)}
        }

        f1, scores = self.calc_unigram_f1()
        # f1, scores = self.calc_f1_with_bleu_rouge()
        result['f1'] = 100 * f1
        result_list = {
            'f1': scores
        }

        # rouge
        print("Calculating rouge...")
        print("hyps = ", self.hyps)
        rouge = Rouge()
        # hyps, refs = map(list, zip(*[[d[0], d[1]] for d in list(zip(self.hyps, self.refs))]))
        # scores = rouge.get_scores(hyps, refs, avg=True)
        # for hyp, ref in zip(self.hyps, self.refs):
            # print("hyp = ", hyp, " ref = ", ref)
            # scores = rouge.get_scores(hyp, ref)
            # print("hyp = ", hyp, " ref = ", ref, " scores = ", scores)

        for rouge_type in ['rouge-1', 'rouge-2', 'rouge-l']:
            for sp in ['f', 'p', 'r']:
                result[f'{rouge_type}_{sp}'] = 0
        refs = [x[0] for x in self.refs]
        hyps = []
        for hyp, ref in zip(self.hyps, refs):
            if not hyp:
                print("hyp = ", hyp, " ref = ", ref)
                hyp = "Null"
                hyps.append("Null")
            else:
                hyps.append(hyp)
            scores = rouge.get_scores(hyp, ref)[0]
            print("hyp = ", hyp, " ref = ", ref, " scores = ", scores)
            for rouge_type in ['rouge-1', 'rouge-2', 'rouge-l']:
                for sp in ['f', 'p', 'r']:
                    result[f'{rouge_type}_{sp}'] += 100 * scores[rouge_type][sp]
        print("len hyps = ", len(hyps), " len refs = ", len(refs))
        
        
        for rouge_type in ['rouge-1', 'rouge-2', 'rouge-l']:
            for sp in ['f', 'p', 'r']:
                result[f'{rouge_type}_{sp}'] = result[f'{rouge_type}_{sp}'] / 95
                # result_list[rouge_type] = [score[rouge_type]['f'] for score in scores]
        # rl, scores = self.calc_rouge_l()
        # result['rouge-l'] = 100 * rl
        # result_list.update({
        #     'rouge-l': scores
        # })

        # result["rouge-1"] = 100 * self.calc_rouge_n(n=1)
        # result["rouge-2"] = 100 * self.calc_rouge_n(n=2)
        # result["bert-score"] = self.calc_bert_score()

        return result, result_list



def evaluate(results):
    metric = Metric(None)
    metric_res = {}
    for lab, gen in results:  # reference, generation
        metric.forstr([lab], gen)
    metric_res, *_ = metric.close()
    return metric_res 

def save_metrics(metrics, save_path):
    log_string = "Eval case: "
    for key, value in metrics.items():
        log_string += " {}: {:.5} | ".format(key, value)
    with open(save_path, "w", encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
        

if __name__ == "__main__":
    in_file1, in_file2, out_file1, out_file2 = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    with open(in_file1, 'r', encoding="utf-8") as f:
        results1 = json.load(f)["request_states"]
    f.close()
    with open(in_file2, 'r', encoding="utf-8") as f:
        results2 = json.load(f)["request_states"]
    f.close()
    
    tmp = []
    for result in results1:
        lab = result["instance"]["references"][0]["output"]["text"]
        gen = result["request"]["result"]["completions"][0]["text"]
        tmp.append([lab, gen])
    metrics1 = evaluate(tmp)
        
    tmp = []
    for result in results2:
        lab = result["instance"]["references"][0]["output"]["text"]
        gen = result["request"]["result"]["completions"][0]["text"]
        tmp.append([lab, gen])
    metrics2 = evaluate(tmp)
    stage1_metrics = copy.deepcopy(metrics1)
    keys = list(stage1_metrics.keys())
    for key in keys:
        stage1_metrics[f"min_{key}"] = min(metrics1[key], metrics2[key])
        stage1_metrics[f"avg_{key}"] = (metrics1[key] + metrics2[key]) / 2
        stage1_metrics[f"max_{key}"] = max(metrics1[key], metrics2[key])
        stage1_metrics[f"m1_{key}"] = metrics1[key]
        stage1_metrics[f"m2_{key}"] = metrics2[key]
    save_metrics(stage1_metrics, out_file1)
    
    compare_results = []
    for i in range(len(results1)):
        gen1 = results1[i]["request"]["result"]["completions"][0]["text"]
        gen2 = results2[i]["request"]["result"]["completions"][0]["text"]
        if gen1 == "" or gen2 == "":
            continue
        compare_results.append([
            gen1,
            gen2,
        ])
    stage2_metrics = evaluate(compare_results)
    save_metrics(stage2_metrics, out_file2)
