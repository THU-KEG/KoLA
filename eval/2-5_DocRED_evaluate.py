import sys
import re
import json
import os

def compare(correct, res):
    correct = correct.split('\n')
    #cor存储正确答案
    cor = []
    for s in correct:
        eid = re.findall(r'(?<=entity)\d+\.?\d*', s)
        if len(eid) < 2:
            continue
        h_id = eid[0]
        t_id = eid[1]
        r = s.split(',')[1]
        cor.append([h_id, t_id, r])
    co_r = 0
    #searchobj存储所有满足格式的答案
    searchobj = re.findall(r'entity\d+,[a-zA-Z\s]+,entity\d+', res, re.I)
    for s in searchobj:
        eid = re.findall(r'(?<=entity)\d+\.?\d*', s)
        #import pdb; pdb.set_trace()
        if len(eid) < 2:
            continue
        h_id = eid[0]
        t_id = eid[1]
        r_id = s.split(',')[1]
        for node in cor:
            h = node[0]
            t = node[1]
            r = node[2]
            if h == h_id and t == t_id and r == r_id:
                co_r += 1
    return [co_r, len(searchobj), len(cor)]
    # print("Correct_Re",correct_re)
    # print("Len_submission_answer",Len_submission_answer)
    # print("tot_relations",tot_relations)
    # print("P",re_p)
    # print("R",re_r)
    # print ('F1:', re_f1)

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='evaluate')
    # parser.add_argument('in_path', type=str, default="../../unified_data/docred2/docred_tu/testforevaluate.json")
    # parser.add_argument('out_path', type=str, default="../../unified_data/docred2/docred_tu/testforevaluate_evaluate.json")
    # parser.add_argument('in_path')
    # parser.add_argument('out_path')
    # args = parser.parse_args()
    in_file, out_file = sys.argv[1], sys.argv[2]


    with open(in_file, 'r', encoding='utf-8') as f:
        res = json.load(f)
        f.close()

    print("Comparing...")
    co = 0
    sub = 0
    tot = 0
    length = len(res["request_states"])
    for vert in res["request_states"]:
        if vert.__contains__('request') == False:
            print("no 'request' key in id:",vert["instance"]["id"])
        else:
            temp = compare(vert["instance"]['references'][0]['output']['text'],
                           vert["request"]["result"]["completions"][0]["text"])
            co += temp[0]
            sub += temp[1]
            tot += temp[2]
    print("-------")
    text=str(length) + " samples in total compared.\n"
    text+="Correct_Re: "+str(co)+"\n"
    text+="Len_submission_answer: "+str(sub)+"\n"
    text+="Tot_relations: "+str(tot)+"\n"
    re_p = 1.0 * co / (sub + 1e-10)
    re_r = 1.0 * co / (tot + 1e-10)
    if re_p + re_r == 0:
        re_f1 = 0
    else:
        re_f1 = 2.0 * re_p * re_r / (re_p + re_r + 1e-10)
    text += "P:" + str(re_p) + "\n"
    text += "R:" + str(re_r) + "\n"
    text += "F1:" + str(re_f1)
    print(text)
    # path = args.path[0:-6]
    # path += '_result.txt'
    # file = open(args.out_path, 'w')
    # file.write(text)
    # file.close()
    res = {
        "P":re_p,
        "R":re_r,
        "F1":re_f1
    }
    json.dump(res, open(os.path.join(out_file), "w"), indent=4)
    print("-------")
    print("Results save at: " + out_file)
