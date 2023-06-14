from argparse import ArgumentParser
import numpy as np
import pandas as pd
import json
from utils import std_model_names,id2dataset, level_1_2, dataset_level_1_2, level_3_4, dataset_level_3_4

def load_dataset(model_name, dataset_name, seed):
    import os
    import json
    # find dataset
    dataset_path = f'/data2/cookie_results/{model_name}/{dataset_name}_inference.json'
    print(dataset_path)
    if not os.path.exists(dataset_path):
        print(f'Dataset {dataset_name} not found!')
        return None, None
    # load dataset
    with open(dataset_path, 'r') as f:
        dataset_test = json.load(f)
        request_states = dataset_test['request_states']
        # print(f'Load {len(request_states)} data from {dataset_name}')
        request_state = request_states[seed]
        instance = request_state['instance']
        print("-"*40)
        print(f"input: {instance['input']['text']}")
        print("-"*40)
        print(f"ref output: {instance['references'][0]['output']['text']}")
        print("-"*40)
        print(f"{model_name} output: {request_state['request']['result']['completions'][0]['text']}")



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
        # 'ChatGLM-2048'
    ]
    parser = ArgumentParser(description='Run inference on a dataset with a given model.')
    parser.add_argument('--models', dest='model_names', default=['flan-t5-xxl', 'flan-ul2'], nargs='+')
    parser.add_argument('--seed', default=56)

    args = parser.parse_args()
    return args.model_names, args.seed

def main():
    model_names, seed = parse_args()
    print(model_names)
    print(f"total {len(model_names)} models")
    
    layers = ['Rolling']
    for layer in layers:
        if layer == 'Rolling':
            dataset_names = ['r_with_triples', 'r_without_triples']
        else:
            dataset_names = ['with_triples', 'without_triples']
        
        for model_name in model_names:
            print("#"*40)
            print(f"model: {model_name}")
            for sub_task in dataset_names:
                print("-"*40)
                print(f"sub_task: {sub_task}")
                load_dataset(model_name, sub_task, seed)

if __name__ == '__main__':
    main()
