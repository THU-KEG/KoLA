import sys
import json
import numpy as np
import re
import random


def compute_acc(input_file, output_file):
    # 对于4类关系，num_ans为label的数量、num_out为回答的数量、num_cor为答对的数量
    num_ans, num_out, num_cor = [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]
    with open(input_file) as file:
        input_json = json.load(file)
    rel_class = {"before": 0, "overlap": 0, "contains": 0, "simultaneous": 0, "begins-on": 0,
                "ends-on": 0, "cause": 1, "precondition": 1, "subevent": 2, "coreference": 3}

    for result in input_json["request_states"]:
        label = result["instance"]["references"][0]["output"]["text"]
        output = result["request"]["result"]["completions"][0]["text"]
        for rel_name in label:
            num_ans[rel_class[rel_name]] += 1
        try:
            search_obj = re.search(r"[aA]nswer(.*?): \[(.*)\]", output)
            output_label = re.findall(r"([a-zA-Z-]+)", search_obj.group(2))
        except:    # 如果识别不了就当做没输出
            continue
        for la in output_label:
            la = la.lower()
            if la in rel_class.keys():
                num_out[rel_class[la]] += 1
                if la in label:
                    num_cor[rel_class[la]] += 1
            else:  # 如果输出了在关系集合之外的关系，则随机指定一类关系作为false positive
                num_out[random.randint(0, 3)] += 1

    num_ans = np.array(num_ans)
    num_out = np.array(num_out)
    num_cor = np.array(num_cor)
    p = num_cor / (num_out+1e-10)
    r = num_cor / (num_ans+1e-10)
    # zero_place = (p==0) | (r==0)
    # p[zero_place] = 1e-10
    # r[zero_place] = 1e-10
    f1 = 2 * p * r / (p + r + 1e-10)
    # f1[zero_place] = 0
    with open(output_file, "w") as f:
        data = dict()
        for i, name in enumerate(["temporal", "causal", "subevent", "coreference"]):
            data[name] = {
                "precision": p[i],
                "recall": r[i],
                "f1-score": f1[i]
            }
        json.dump(data, f, indent=4)
        # f.write("            Temporal | Causal | Subevent | Coreference\n")
        # f.write(f"precision:  {p[0]:.4f}   | {p[1]:.4f} | {p[2]:.4f}   | {p[3]:.4f}\n")
        # f.write(f"recall:     {r[0]:.4f}   | {r[1]:.4f} | {r[2]:.4f}   | {r[3]:.4f}\n")
        # f.write(f"f1-score:   {f1[0]:.4f}   | {f1[1]:.4f} | {f1[2]:.4f}   | {f1[3]:.4f}\n")


if __name__ == "__main__":
    in_file, out_file = sys.argv[1], sys.argv[2]
    compute_acc(in_file, out_file)