import os
import sys
import json
from datetime import date
from collections import defaultdict, Counter
from tqdm import tqdm
def whether_equal(answer, pred):
    def truncate_float(x):
        # convert answer from '100.0 meters' to '100 meters'
        try:
            v, *u = x.split()
            v = float(v)
            if v - int(v) < 1e-5:
                v = int(v)
            if len(u) == 0:
                x = str(v)
            else:
                x = '{} {}'.format(str(v), ' '.join(u))
        except:
            pass
        return x

    def equal_as_date(x, y):
        # check whether x and y are equal as type of date or year
        try:
            x_split = x.split('-')
            y_split = y.split('-')
            if len(x_split) == 3:
                x = date(int(x_split[0]), int(x_split[1]), int(x_split[2]))
            else:
                x = int(x)
            if len(y_split) == 3:
                y = date(int(y_split[0]), int(y_split[1]), int(y_split[2]))
            else:
                y = int(y)
            if isinstance(x, date) and isinstance(y, date):
                return x == y
            else:
                x = x.year if isinstance(x, date) else x
                y = y.year if isinstance(y, date) else y
                return x == y
        except:
            return False

    answer = truncate_float(answer)
    pred = truncate_float(pred)
    if equal_as_date(answer, pred):
        return True
    else:
        return answer == pred


def evaluate(raw_data):
    raw_data = raw_data["request_states"]
    labels = ['overall', 'multihop', 'qualifier', 'comparison', 'logical', 'count', 'verify']
    total = {k:0 for k in labels}
    correct = {k:0 for k in labels}
    for item in tqdm(raw_data):
        cot = item['request']['result']['completions'][0]['text']
        # predicted_answer = cot.split('So the answer is: ')[-1][:-1]
        predicted_answer = cot.split('So the answer is: ')[-1].split('Q: ')[0].strip()[:-1]
        # print(predicted_answer)
        ground_truth_answer = item['instance']['references'][0]['output']['text'].strip()
        # print(ground_truth_answer)
        program = item['instance']['program']
        functions = [f['function'] for f in program]
        cur_labels = ['overall']
        for f in functions:
            if f in {'Relate'} or f.startswith('Filter'):
                cur_labels.append('multihop')
                break
        for f in functions:
            if f in {'QFilterStr', 'QFilterNum', 'QFilterYear', 'QFilterDate', 'QueryAttrUnderCondition', 'QueryAttrQualifier', 'QueryRelationQualifier'}:
                cur_labels.append('qualifier')
                break
        for f in functions:
            if f in {'SelectBetween','SelectAmong'}:
                cur_labels.append('comparison')
                break
        for f in functions:
            if f in {'And', 'Or'}:
                cur_labels.append('logical')
                break
        for f in functions:
            if f in {'Count'}:
                cur_labels.append('count')
                break
        for f in functions:
            if f in {'VerifyStr','VerifyNum','VerifyYear','VerifyDate'}:
                cur_labels.append('verify')
                break
        if whether_equal(ground_truth_answer, predicted_answer):
            for k in cur_labels:
                correct[k] += 1
        
        for k in cur_labels:
            total[k] += 1
    accuracies = {}
    for k in labels:
        if total[k] != 0:
            accuracies[k] = round(correct[k]/total[k], 3)
        else:
            accuracies[k] = 0
    return accuracies


def main():
    in_file, out_file = sys.argv[1], sys.argv[2]
    raw_data = json.load(open(in_file))
    accuracies = evaluate(raw_data)
    json.dump(accuracies, open(out_file, 'w'), indent = 2)

if __name__ == '__main__':
    main()
