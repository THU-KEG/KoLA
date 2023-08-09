import json
import copy
import sys
import pprint
from typing import Tuple, Dict, List, Optional, Union
from sklearn.metrics import f1_score, precision_score, recall_score


def find_triggers(tokens):
    events = []
    start_pos = []
    end_pos = []
    for i, t in enumerate(tokens):
        if '<mark>' in t:
            start_pos.append(i)
        if '</mark>' in t:
            end_pos.append(i)
    for i, (start, end) in enumerate(zip(start_pos, end_pos)):
        events.append({'trigger': ' '.join(tokens[start+1: end]), 'offset': [start-2*i, end-1-2*i]})
    return events


def get_labels(text):
    text = ' '.join(text.split())
    try:
        text = text.split('Events:')
        output = text[0]
        event = text[1]
    except:
        output = 'Output:'
        event = ''
    try:
        tokens = ' </mark> '.join(' <mark> '.join(output.split('Output:')[1].split('<mark>')).split('</mark>')).split()
    except:
        tokens = []
    # print(tokens)
    types = event.split(';')
    pred = find_triggers(tokens)
    for i, p in enumerate(pred):
        if i < len(types):
            t = types[i].split(':')
            if p['trigger'] == t[0]:
                p['type'] = t[1]
            else:
                p['type'] = 'NA'
        else:
            p['type'] = 'NA'
    return pred


def get_predictions(request_states):
    preds, labels = [], []
    for r in request_states:
        if r.get('request', False) and r['request']['result']['success']:
            label = r['instance']['references'][0]['output']['text']
            pred = r['request']['result']['completions'][0]['text']
            sent_id = r['instance']['id']
            label = get_labels(label)
            pred = get_labels(pred)
            # print(label)
            # print(pred)
            for l in label:
                l['sent_id'] = sent_id
            for p in pred:
                p['sent_id'] = sent_id
            preds.append(pred)
            labels.append(label)
    return preds, labels


def f1_score_overall(preds: Union[List[str], List[tuple]],
                     labels: Union[List[str], List[tuple]]) -> Tuple[float, float, float]:
    """Computes the overall F1 score of the predictions.
    Computes the overall F1 score of the predictions based on the calculation of the overall precision and recall after
    counting the true predictions, in which both the prediction of mention and type are correct.
    Args:
        preds (`Union[List[str], List[tuple]]`):
            A list of strings indicating the prediction of labels from the model.
        labels (`Union[List[str], List[tuple]]`):
            A list of strings indicating the actual labels obtained from the annotated dataset.
    Returns:
        precision (`float`), recall (`float`), and f1 (`float`):
            Three float variables representing the computation results of precision, recall, and F1 score, respectively.
    """

    true_pos = 0
    tot_preds = 0
    tot_labels = 0
    for i in range(len(preds)):
        tot_preds += len(preds[i])
        tot_labels += len(labels[i])
        label_stack = copy.deepcopy(labels[i])
        for pred in preds[i]:
            if pred in label_stack:
                true_pos += 1
                # one prediction can only be matched to one ground truth.
                label_stack.remove(pred)
    precision = true_pos / (tot_preds+1e-10)
    recall = true_pos / (tot_labels+1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    return precision, recall, f1


def event_detection_score(preds, labels):
    ps = [[(t['trigger'], t['sent_id'], t['offset']) for t in p] for p in preds]
    ls = [[(t['trigger'], t['sent_id'], t['offset']) for t in l] for l in labels]
    return f1_score_overall(ps, ls)


def event_classification_score(preds, labels):
    ps = [[(t['trigger'], t['sent_id'], t['offset'], t['type']) for t in p] for p in preds]
    ls = [[(t['trigger'], t['sent_id'], t['offset'], t['type']) for t in l] for l in labels]
    return f1_score_overall(ps, ls)


if __name__ == '__main__':

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--result_file', default='./data/test.json')
    # parser.add_argument('--output_file', default='./data/result.json')
    # args = parser.parse_args()
    in_file, out_file = sys.argv[1], sys.argv[2]

    with open(in_file, 'r') as f:
        result = json.load(f)
        preds, labels = get_predictions(result['request_states'])
        ed_score = event_detection_score(preds, labels)
        ec_score = event_classification_score(preds, labels)
        score = {'identification': {'precision': ed_score[0], 'recall': ed_score[1], 'f1': ed_score[2]}, 
                 'classification': {'precision': ec_score[0], 'recall': ec_score[1], 'f1': ec_score[2]}}
    with open(out_file, 'w') as fp:
        json.dump(score, fp, indent=4)
    pprint.pprint(score)