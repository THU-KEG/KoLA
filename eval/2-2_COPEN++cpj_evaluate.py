import json
import re
import sys
import argparse


def compute_acc(file_path, output_path):
    data = json.load(open(file_path))
    correct = 0
    for instance in data["request_states"]:
        gold_answer = instance["instance"]["references"][0]["output"]["text"].lower()
        pred_answer = instance["request"]["result"]["completions"][0]["text"]
        pred_answer = pred_answer.split(".")[0].strip().lower()
        if gold_answer == pred_answer:
            correct += 1
    accuracy = correct / len(data["request_states"])
    metric = {"accuracy": accuracy}
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
