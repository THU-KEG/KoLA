from argparse import ArgumentParser
import regex as re
from typing import Any, Tuple
import numpy as np
import pandas as pd
import json
from copy import deepcopy
from utils import std_model_names, id2dataset, level_1_2, dataset_level_1_2, level_3_4, dataset_level_3_4

threshold = 25
notfound_dict = {}


level2color = {
    '1-1': 'deep1',
    '1-2': 'shallow1',
    '1-3': 'deep1',
    '2-1': 'deep2',
    '2-2': 'shallow2',
    '2-3': 'deep2',
    '2-4': 'shallow2',
    '2-5': 'deep2',
    '2-6': 'shallow2',
    '2-7': 'deep2',
    '2-8': 'shallow2',
    '3-1': 'deep3',
    '3-2': 'shallow3',
    '3-3': 'deep3',
    '3-4': 'shallow3',
    '3-5': 'deep3',
    '3-6': 'shallow3',
    '4-1': 'deep4',
    '4-2': 'shallow4',
}

def parse_model_conf(model_name:str)->Tuple[str,str]:
    """
    return (model,parameter)
    """
    model_conf = {
        'flan-t5-xxl':{"model":"FLAN-T5","parameter":"11B"},
        'ul2':{ "model": "UL2","parameter": "20B"},
        'flan-ul2':{ "model": "FLAN-UL2","parameter": "20B"},
        'GPT-JT-6B-v1':{"model": "GPT-JT","parameter": "6B"},
        'gpt-j-6b':{   "model": "GPT-J","parameter": "6B"},
        'gpt-neox-20b':{ "model": "GPT-NeoX","parameter": "20B"},
        'bloom-7b1':{"model": "BLOOM","parameter": "7B"},
        'T0pp':{"model": "T0++","parameter": "11B"},
        'llama-65b-hf':{"model": "LLaMa","parameter": "65B"},
        'alpaca-lora-7b':{ "model": "Alpaca","parameter": "7B"},
        'J2-Jumbo':{"model": "J2-Jumbo-Instruct","parameter": "178B"},
        'Cohere-xlarge':{"model": "Cohere-command","parameter": "52.4B"},
        'GPT-3.5-turbo':{"model": "GPT-3.5-turbo","parameter": "Unknown"},
        'GPT-3_curie_v1':{"model": "GPT-3 curie v1","parameter": "6.7B"},
        'GPT3_davinci_v1':{ "model": "GPT-3 davinci v1","parameter": "175B"},
        'instructGPT_curie_v1':{    "model": "InstructGPT curie v1","parameter": "6.7B"},
        'InstructGPT_davinci_v2':{"model": "InstructGPT davinci v2","parameter": "175B"},
        'GLM-130B':{"model": "GLM","parameter": "130B"},
        'GPT-4':{"model": "GPT-4","parameter": "Unknown"},
        'ChatGLM':{"model":"ChatGLM","parameter":"130B"},
        'chatglm-6b':{ "model": "ChatGLM","parameter": "6B"},
        # 'ChatGLM-2048'
    }

    res = model_conf.get(model_name,None)
    if res  is not  None:
        return res["model"],res["parameter"]
    raise Exception("获取模型信息失败")




def parse_args():
    default_models = [
        'flan-t5-xxl',
        'ul2',
        'flan-ul2',
        'GPT-JT-6B-v1',
        'gpt-j-6b',
        'gpt-neox-20b',
        'bloom-7b1',
        'T0pp',
        'llama-65b-hf',
        'alpaca-lora-7b',
        'J2-Jumbo',
        'Cohere-xlarge',
        'GPT-3.5-turbo',
        'GPT-3_curie_v1',
        'GPT3_davinci_v1',
        'instructGPT_curie_v1',
        'InstructGPT_davinci_v2',
        'GLM-130B',
        'GPT-4',
        'ChatGLM',
        'chatglm-6b',
        # 'ChatGLM-2048'
    ]
    parser = ArgumentParser(
        description='Run inference on a dataset with a given model.')
    parser.add_argument('--models', dest='model_names',
                        default=default_models, nargs='+')
    parser.add_argument('--datasets', dest='dataset_names',
                        default=dataset_level_3_4, nargs='+')

    args = parser.parse_args()
    return args.model_names, args.dataset_names


def get_one_metric(model_name, dataset_name):
    with open(f'./analyse/results/{model_name}/{dataset_name}_evaluation.json') as f:
        json_obj = json.load(f)
    one_metric = 0
    if dataset_name in ['hotpotqa', 'musique',  '2wikimultihopqa']:
        one_metric = json_obj['answer_f1']
    elif dataset_name == 'kqapro':
        one_metric = json_obj['overall']
    elif dataset_name in ['KoRC++ood', 'r_KoRC++ood', 'r_1_simple_sample', '1_low_freq_ent', '2_high_freq_ent']:
        one_metric = json_obj['token_f1_with_penalty']
    elif dataset_name in ['COPEN++cic', 'COPEN++cpj', 'COPEN++csj']:
        one_metric = json_obj['accuracy']
    elif dataset_name in ['FewNERD++inter', 'FewNERD++intra', 'FewNERD++supervised', 'r_DocRED', 'DocRED']:
        one_metric = json_obj['F1']
    elif dataset_name == 'MAVEN':
        one_metric = json_obj['classification']['f1']
    elif dataset_name == 'MAVEN-ERE':
        one_metric = 0
        for k, v in json_obj.items():
            one_metric += v['f1-score']
        one_metric = one_metric/len(json_obj)
    elif dataset_name.startswith('c1_'):
        # creating, rolling
        # one_metric = json_obj['min_rouge-l_f']
        one_metric = json_obj['avg_rouge-l_f']  # baodi
        # one_metric = json_obj['m2_rouge-l_f']
    else:
        # creating, rolling stage2 score
        # one_metric = 100 - json_obj['rouge-1']
        one_metric = json_obj['rouge-l_f']
    return one_metric


def get_final_score(model_names, dataset_name):
    final_scores = []
    c1_scores = []
    for model_name in model_names:
        final_score = 0
        # get one metric for representation
        try:
            if dataset_name == 'FewNERD':
                f_score = 0
                for f_name in ['FewNERD++inter', 'FewNERD++intra', 'FewNERD++supervised']:
                    f_score += get_one_metric(model_name, f_name)
                final_score = f_score / 3
            elif dataset_name == 'Creating' or dataset_name == 'Rolling':
                final_score = get_one_metric(model_name, f'c2_{dataset_name}')
                c1_scores.append(get_one_metric(
                    model_name, f'c1_{dataset_name}'))
            else:
                final_score = get_one_metric(model_name, dataset_name)
        except FileNotFoundError:
            print(f'{dataset_name}: {model_name} not found')
            if dataset_name in notfound_dict.keys():
                notfound_dict[dataset_name].append(model_name)
            else:
                notfound_dict[dataset_name] = [model_name]
            if dataset_name == 'Creating' or dataset_name == 'Rolling':
                c1_scores.append(0)
        # x100 for f1 and accuracy
        if dataset_name == 'Creating' or dataset_name == 'Rolling':
            final = final_score
        else:
            final = final_score*100
        # print('%.2f'%final)
        final_scores.append(final)
    # avergae and std
    avg = sum(final_scores)/len(final_scores)
    # print('average: %.2f'%avg)
    std = np.std(final_scores)
    # print('std: %.2f'%std)
    # standarderize
    final_score_nps = []
    for final_score in final_scores:
        final_score_np = (final_score - avg) / std
        final_score_nps.append(final_score_np)
    return final_scores, final_score_nps, c1_scores


def re_strandardize(dataset2raw_scores, dataset2final_scores, dataset_pair):
    # d1 = dataset_pair[0]
    # d2 = dataset_pair[1]
    # raw_scores1 = dataset2raw_scores[d1]
    # raw_scores2 = dataset2raw_scores[d2]
    # combined avg and std
    raw_scores_lst = []
    combined = []
    for dataset in dataset_pair:
        combined += dataset2raw_scores[dataset]
        raw_scores_lst.append(dataset2raw_scores[dataset])
    avg = sum(combined)/len(combined)
    std = np.std(combined)
    # d1
    for i, raw_scores in enumerate(raw_scores_lst):
        final_score_nps = []
        for final_score in raw_scores:
            final_score_np = (final_score - avg) / std
            final_score_nps.append(final_score_np)
        dataset2final_scores[dataset_pair[i]] = final_score_nps
    # final_score_nps = []
    # for final_score in raw_scores1:
    #     final_score_np = (final_score - avg) / std
    #     final_score_nps.append(final_score_np)
    # dataset2final_scores[d1] = final_score_nps
    # # d2
    # final_score_nps = []
    # for final_score in raw_scores2:
    #     final_score_np = (final_score - avg) / std
    #     final_score_nps.append(final_score_np)
    # dataset2final_scores[d2] = final_score_nps
    return dataset2final_scores

    

def main():
    model_names, _ = parse_args()
    print(model_names)
    print(f"total {len(model_names)} models")
    # get scores
    dataset2raw_scores = {}
    dataset2final_scores = {}
    dataset2c1_scores = {}
    for dataset_names in [dataset_level_1_2, dataset_level_3_4]:
        for dataset_name in dataset_names:
            final_scores, final_score_nps, c1_scores = get_final_score(
                model_names, dataset_name)
            # print('%.2f'%final_score_np)
            # add to dict
            dataset2raw_scores[dataset_name] = final_scores
            dataset2final_scores[dataset_name] = final_score_nps
            dataset2c1_scores[dataset_name] = c1_scores
        # re_strandardize for r_* datasets
        if dataset_names == dataset_level_3_4:
            dataset2final_scores = re_strandardize(
                dataset2raw_scores, dataset2final_scores, ['r_KoRC++ood', 'KoRC++ood'])
            # dataset2final_scores = re_strandardize(dataset2raw_scores, dataset2final_scores, ['Creating', 'Rolling'])
        else:
            dataset2final_scores = re_strandardize(dataset2raw_scores, dataset2final_scores, [
                                                   '2_high_freq_ent', '1_low_freq_ent', 'r_1_simple_sample'])
            dataset2final_scores = re_strandardize(
                dataset2raw_scores, dataset2final_scores, ['DocRED', 'r_DocRED'])

    # re_strandardize rolling and creating
    C_R_dics = {
        'Creating': dataset2c1_scores['Creating'], 'Rolling': dataset2c1_scores['Rolling']}
    # C_R_df = pd.DataFrame(C_R_dics)
    # temp_final_C_R_df = pd.DataFrame(dataset2final_scores)[['Creating', 'Rolling']]
    # average_C_R_df = C_R_df.sum(axis=1) / 2 # calculate average for 20230605
    # average_C_R_df = (temp_final_C_R_df  + C_R_df * 2).sum(axis=1) / 6  # xiaozhi metric baodi
    # average_C_R_df = (temp_final_C_R_df + C_R_df).sum(axis=1) / 2  # without + stage2 C_R_df
    # total_C_R_rank = average_C_R_df.rank(method='average', ascending=False)
    for c_data in ['Creating', 'Rolling']:
        C_R_lst = C_R_dics[c_data]
        # print(C_R_lst)
        for i, model in enumerate(model_names):
            c1_raw_score = C_R_lst[i] * 2
            sum_score = c1_raw_score
            c2_raw_score = dataset2raw_scores[c_data][i]
            # sum_score = c1_raw_score + c2_raw_score
            dataset2raw_scores[c_data][i] = sum_score - c2_raw_score
    dataset2final_scores = re_strandardize(
        dataset2raw_scores, dataset2final_scores, ['Creating', 'Rolling'])
    deepcopy_dataset2final_scores = deepcopy(dataset2final_scores)
    # min max the scores
    values = []
    for v in dataset2final_scores.values():
        values += list(v)
    min_v = min(values)
    max_v = max(values)
    for k, v in dataset2final_scores.items():
        dataset2final_scores[k] = [100 * (i-min_v)/(max_v-min_v) for i in v]

    # show all datasets
    for dataset_names in [dataset_level_1_2, dataset_level_3_4]:
        if dataset_names == dataset_level_3_4:
            # for table level3-4
            create_standard_table(level_3_4, model_names, dataset2final_scores,
                                  deepcopy_dataset2final_scores, dataset2c1_scores, dataset2raw_scores, 6)
        else:
            # for table level1-2
            create_standard_table(level_1_2, model_names, dataset2final_scores,
                                  deepcopy_dataset2final_scores, dataset2c1_scores, dataset2raw_scores, 3)


def get_C_R_rank(dataset2c1_scores):
    C_R_dics = {
        'Creating': dataset2c1_scores['Creating'], 'Rolling': dataset2c1_scores['Rolling']}
    C_R_df = pd.DataFrame(C_R_dics)
    average_C_R_df = C_R_df.sum(axis=1) / 2
    total_C_R_rank = average_C_R_df.rank(method='average', ascending=False)
    return total_C_R_rank


def number_to_rank(number):
    if number == 1:
        return '1st'
    elif number == 2:
        return '2nd'
    elif number == 3:
        return '3rd'
    else:
        return '%dth' % number


def get_color(row_id, data_level='5'):
    row_id = int(row_id)
    print(row_id)
    if len(data_level) > 1:
        data_level = data_level[0]

    color_name = ''
    if row_id % 2 == 0:
        color_name += 'deep'
    else:
        color_name += 'shallow'
    color_name += data_level
    return color_name


def get_true_rank(i, rank_with_id):
    for rank_id in rank_with_id:
        if rank_id[0] == i:
            return rank_id[1]


def create_standard_table(levels, model_names, dataset2final_scores, deepcopy_dataset2final_scores, dataset2c1_scores, dataset2raw_scores, level1_len):
    # final_scores are freezed
    df = pd.DataFrame(dataset2final_scores)
    df.to_csv('./analyse/dataset_final_scores.csv')
    # get ranks
    # average_df = df.sum(axis=1) / len(df)
    std_df = pd.DataFrame(deepcopy_dataset2final_scores)
    # get rank by each layer's average
    id_starts = [0, 3, 11, 17]
    final_avg = []
    for i, id_start in enumerate(id_starts):
        if i == len(id_starts) - 1:
            id_end = len(model_names)
        else:
            id_end = id_starts[i+1]
        print(id_start, id_end)
        cols = std_df.columns[id_start:id_end]
        print(cols)
        average_df = std_df[cols].sum(axis=1) / len(cols)
        if i == 0:
            final_avg = average_df
        else:
            final_avg += average_df

    # 四类任务上最终分数
    final_avg /= 4

    total_rank = final_avg.rank(method='average', ascending=False)
    rank_with_id = [(i, rank) for i, rank in enumerate(total_rank)]
    rank_with_id.sort(key=lambda x: x[1], reverse=False)
    # print(rank_with_id)
    # level ranks

    # 所有任务上的分数
    if levels == level_1_2:
        local_df = df.iloc[:, :11]
    else:
        local_df = df.iloc[:, 11:]
    print(local_df)
    l1_cols = local_df.columns[0:level1_len]
    l2_cols = local_df.columns[level1_len:]
    l1_ranks = local_df[l1_cols].sum(axis=1).rank(
        method='average', ascending=False)
    l2_ranks = local_df[l2_cols].sum(axis=1).rank(
        method='average', ascending=False)
    # get table rows
    model2row = {}
    for i, model in enumerate(model_names):
        true_rank = get_true_rank(i, rank_with_id)
        true_name = std_model_names[i]
        if int(true_rank) % 2 == 0:
            model_color_name = 'gry'

        else:
            model_color_name = 'wit'
        model_row_str = f"\cellcolor{ {model_color_name} } {true_name} & "
        for j, data_id in enumerate(levels):

            if j == level1_len:
                true_color = str(int(data_id[0])-1)
                model_row_str += f"\cellcolor{ {get_color(true_rank, true_color)} }  {number_to_rank(l1_ranks[i])} & "
            dataset_name = id2dataset[data_id]
            final_scores = dataset2final_scores[dataset_name]
            # 如果 这个数据集上分数为0
            if dataset2raw_scores[dataset_name][i] == 0:
                if dataset_name in notfound_dict.keys() and model in notfound_dict[dataset_name]:
                    model_row_str += f"\cellcolor{ {get_color(true_rank, data_id)} } --- & "
                    continue

                model_row_str += f"\cellcolor{ {get_color(true_rank, data_id)} } {round(final_scores[i],1)} & "
                continue
            else:
                model_score = round(final_scores[i], 1)
            if dataset_name == 'Creating' or dataset_name == 'Rolling':
                c1_scores = dataset2c1_scores[dataset_name]
                c1_score = round(c1_scores[i], 1)
                # model_row_str += f" {model_score} ({c1_score}) & "
                model_row_str += f"\cellcolor{ {get_color(true_rank, data_id)} } {model_score} & "
            else:
                model_row_str += f"\cellcolor{ {get_color(true_rank, data_id)} } {model_score} & "

        if dataset_name == 'Rolling':
            avg_score = str(round(final_avg[i], 2))
            if len(avg_score.split(".")[1]) != 2:
                avg_score += "0"
            # avg_score = "%.2f" % avg_score
            model_row_str += f"\cellcolor{ {get_color(true_rank, data_id)} }  {number_to_rank(l2_ranks[i])} "
            # color_name1= 'deep5'
            # color_name2= 'shallow5'
            model_row_str += f"&\cellcolor{ {get_color(true_rank)} }  {avg_score} & \cellcolor{ {get_color(true_rank)} }  {number_to_rank(total_rank[i])} \\\\ "
            # model_row_str += f"& {round(average_C_R_df[i],1)} & {number_to_rank(total_C_R_rank[i])} \\\\ "
        else:
            model_row_str += f"\cellcolor{ {get_color(true_rank, data_id)} } {number_to_rank(l2_ranks[i])} \\\\ "
        model2row[i] = model_row_str

    # output table in the rank order
    for i, rank in rank_with_id:
        print(model2row[i])
    print('-------------------')





if __name__ == '__main__':
    main()
