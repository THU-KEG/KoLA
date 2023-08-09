import os
import sys
import json
import argparse

def get_entity_and_type(text):
    entity_types = []
    items = text.split("]")
    for item in items:
        value = item.split("[")[-1]
        if "|" not in value:
            continue
        entity, type = value.split("|")[:2]
        entity_types.append((entity.strip().lower(), type.strip().lower()))
    return entity_types


def evaluate_F1(predictions, labels):
    TP = 0
    all_entities = 0
    all_preds = 0
    for prediction, label in zip(predictions, labels):
        pred_entity_types = get_entity_and_type(prediction)
        gold_entity_types = get_entity_and_type(label)
        for pred in pred_entity_types:
            if pred in gold_entity_types:
                TP += 1
        all_entities += len(gold_entity_types)
        all_preds += len(pred_entity_types)
    P = TP / (all_preds + 1e-10)
    R = TP / (all_entities + 1e-10)
    F1 = 2 * P * R / (P + R + 1e-10)
    return {
        "precision": P,
        "recall": R,
        "F1": F1
    }

def compute_acc(file_path, output_path):
    data = json.load(open(file_path))
    predictions, labels = [], []
    for instance in data["request_states"]:
        gold_answer = instance["instance"]["references"][0]["output"]["text"].lower()
        labels.append(gold_answer)
        pred_answer = instance["request"]["result"]["completions"][0]["text"]
        predictions.append(pred_answer)

    metric = evaluate_F1(predictions, labels)
    with open(output_path, "w") as f:
        json.dump(metric, f)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    in_file, out_file = sys.argv[1], sys.argv[2]
    compute_acc(in_file, out_file)
    # parser.add_argument("--input_path", type=str, default=None)
    # parser.add_argument("--output_path", type=str, default=None)
    # args = parser.parse_args()

    # compute_acc(args.input_path, args.output_path)
