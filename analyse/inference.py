from argparse import ArgumentParser
from tqdm import tqdm
import os
import torch.nn.functional as F
import json
import time
import torch

from transformers import T5Tokenizer, AutoTokenizer, \
    AutoModelForCausalLM, T5ForConditionalGeneration, AutoModelForSeq2SeqLM, \
    StoppingCriteria, StoppingCriteriaList, \
    LlamaTokenizer, LlamaForCausalLM


class StopSequences(StoppingCriteria):
    def __init__(self, stop_sequences_set):
        super().__init__()
        self.stop_sequences_set = stop_sequences_set
    
    def __call__(self, input_ids, scores):
        if input_ids[0][-1].item() in self.stop_sequences_set:
            return True
        return False


def parse_args():
    parser = ArgumentParser(description='Run inference on a dataset with a given model.')
    parser.add_argument('--models', dest='model_name', required=True, nargs='+')
    parser.add_argument('--datasets', dest='dataset_names', required=True, nargs='+')
    parser.add_argument('--layer', required=True)
    parser.add_argument('--test_input', action='store_true')
    parser.add_argument('--num_shot', type=int, default=5)

    args = parser.parse_args()
    return args.model_name, args.dataset_names, args.layer, args.test_input, args.num_shot


def load_model(model_name, test_input):
    # only load tokenizer for test_input mode

    # find model
    model_paths = [i + model_name for i in [
        '/data1/zijun/instruction_distillation/model/',
        '/data2/cookie_huggingface_models/',
    ]]
    for model_path in model_paths:
        if os.path.exists(model_path):
            break
    else:
        print(f'Model {model_name} not found!')
        return None, None, None

    # load model from local files
    # all supported models are listed as following:
    if model_name == 'flan-t5-xxl':
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        model = T5ForConditionalGeneration.from_pretrained(
            model_path, device_map='auto', return_dict_in_generate=True, output_scores=True
        ) if not test_input else None
    elif model_name == 'flan-ul2':                                                               
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = T5ForConditionalGeneration.from_pretrained(
            model_path, device_map='auto', return_dict_in_generate=True, output_scores=True, torch_dtype=torch.bfloat16
        ) if not test_input else None
    elif model_name == 'ul2':
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = T5ForConditionalGeneration.from_pretrained(
            model_path, device_map='auto', return_dict_in_generate=True, output_scores=True
        ) if not test_input else None
    elif model_name == 'T0pp':
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path, device_map='auto', return_dict_in_generate=True, output_scores=True
        ) if not test_input else None
    # elif model_name == 'opt-iml-30b':  # TODO
    #     tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    #     model = AutoModelForCausalLM.from_pretrained(
    #         model_path, device_map='auto', return_dict_in_generate=True, output_scores=True, torch_dtype=torch.float16
    #     ) if not test_input else None
    elif model_name == 'alpaca-lora-7b':
        tokenizer = LlamaTokenizer.from_pretrained(model_path)
        model = LlamaForCausalLM.from_pretrained(
            model_path, device_map='auto', return_dict_in_generate=True, output_scores=True, torch_dtype=torch.float16, load_in_8bit=True
        ) if not test_input else None
    elif model_name in [
        'gpt-neox-20b',
        'GPT-JT-6B-v1',
        'gpt-j-6b',
        'bloom-7b1',
        'llama-65b-hf',
    ]:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map='auto', return_dict_in_generate=True, output_scores=True
        ) if not test_input else None
    else:
        print(f'Model {model_name} not supported!')
        tokenizer, model = None, None

    max_length = {
        'flan-t5-xxl': 512,
        'ul2': 512,
        'flan-ul2': 2048,
        'GPT-JT-6B-v1': 2048,
        'gpt-j-6b': 2048,
        'gpt-neox-20b': 2048,
        'bloom-7b1': 10000,  # no length overflow till now
        'T0pp': 512,
        'llama-65b-hf': 10000,
        'alpaca-lora-7b': 10000,
    }[model_name]

    return tokenizer, model, max_length


def load_dataset(dataset_name, layer):
    
    # find dataset
    dataset_path = f'/data2/cookie/input/{layer}/{dataset_name}/'
    if not os.path.exists(dataset_path):
        print(f'Dataset {dataset_name} not found!')
        return None, None
    
    # load dataset
    # all supported dataset are listed as following:
    
    # datasets with test.json and train.json, their inputs need to be assembled from the two files
    test_train_datasets = set([
        'MAVEN',
        'MAVEN-ERE',
        'DocRED',
        'COPEN',
        'FewNERD',
        '1_simple_sample',
        '2_1-N_sample',
        '3_low_freq_ent_sample',
        '4_low_freq_rel_sample',
        '5_complex_sample',
        'KoRC',
        '1_low_freq_ent',
        '2_high_freq_ent',
    ])
    # datasets with only one file which means examples have been attached to instances
    preprocessed_datasets_to_file = {
        'hotpotqa': 'hotpotqa_sample.json',
        'musique' : 'musique_sample.json',
        'kqapro'  : 'kqapro_sample.json',
        '2wikimultihopqa': '2WikiMultihopQA_sample.json',
    }

    if dataset_name.split('/')[0] in test_train_datasets:
        test_file = dataset_path + 'test.json'
    else:
        test_file = dataset_path + preprocessed_datasets_to_file[dataset_name]
    with open(test_file) as f:
        dataset_test = json.load(f)
    
    dataset_train = None
    if dataset_name.split('/')[0] in test_train_datasets:  # preprocessed datasets have no train.json
        with open(dataset_path + 'train.json') as f:
            dataset_train = json.load(f)

    return dataset_test, dataset_train


def main():

    model_names, dataset_names, layer, test_input, num_shot = parse_args()
    
    for model_name in model_names:

        tokenizer, model, max_length = load_model(model_name, test_input)
        if not tokenizer and not model:
            return
        
        if not test_input:
            result_path = '/data2/cookie_results/' + model_name + '/'
            if not os.path.exists(result_path):
                os.mkdir(result_path)

        # replace datasets to their sub tasks, if exist
        datasets_with_sub_tasks = {
            'COPEN': ['cic', 'cpj', 'csj'],
            'FewNERD': ['inter', 'intra', 'supervised'],
            'KoRC': ['iid', 'ood'],
        }
        dataset_names_temp = []
        for i in dataset_names:
            if i not in datasets_with_sub_tasks:
                dataset_names_temp.append(i)
            else:
                dataset_names_temp.extend((f'{i}/{j}' for j in datasets_with_sub_tasks[i]))  # 'task/subtask'
        dataset_names = dataset_names_temp

        for dataset_name in dataset_names:
            
            dataset_test, dataset_train = load_dataset(dataset_name, layer)
            if not dataset_test:
                return
            
            spec = dataset_test['adapter_spec']
            instruction = spec['instructions']
            input_prefix = spec['input_prefix']
            input_suffix = spec['input_suffix']
            output_prefix = spec['output_prefix']
            output_suffix = spec['output_suffix']
            instance_prefix = spec['instance_prefix'] if 'instance_prefix' in spec else ''  # some earlier datasets don't have this field

            # every query start with instruction
            query_prefix = instruction

            # instances in following datasets have been preprocessed, only instruction needs to be attached
            preprocessed_datasets = set([
                '2wikimultihopqa',
                'hotpotqa',
                'kqapro',
                'musique',
            ])
            
            if dataset_name not in preprocessed_datasets and dataset_train:  # if not preprocessed, examples need to be concatenated
                for i in dataset_train['request_states'][:num_shot]:
                    i_input = i['instance']['input']['text']
                    i_output = i['instance']['references'][0]['output']['text']
                    query_prefix += instance_prefix
                    query_prefix += f'{input_prefix}{i_input}{input_suffix}'
                    query_prefix += f'{output_prefix}{i_output}{output_suffix}'
                
            query_prefix += instance_prefix + input_prefix

            if not test_input:
                max_new_tokens = dataset_test['adapter_spec']['max_tokens']
                stop_sequences_set = set(tokenizer.encode(i)[0] for i in dataset_test['adapter_spec']['stop_sequences'])
                stop_criteria = StopSequences(stop_sequences_set)
                print(f'Running inference on {dataset_name} with {model_name}')

            if test_input:
                lens = []

            # run inference
            for request in tqdm(dataset_test['request_states']):
                
                input_text = query_prefix + request['instance']['input']['text']
                if dataset_name not in preprocessed_datasets:
                    input_text += input_suffix + output_prefix

                input_ids = tokenizer(input_text, return_tensors='pt').input_ids.to('cuda')
                
                if test_input:  # just print one input without running inference
                    lens.append(len(input_ids[0]))
                    continue
                
                if len(input_ids[0]) > max_length:  # mark and skip
                    print(f'skip overlong instances: {len(input_ids[0])}')
                    request['request'] = {
                        'result': {
                            'success': False,
                            'completions': [{
                                'text': 'null_overlength',
                                'logprob': 0.,
                                'tokens': [],
                            }],
                            'cached': True,
                            'request_time': 0.,
                            'request_datetime': int(time.time()),
                    }
                    }
                    continue
                
                start_time = time.time()
                # TODO: pad_token_id=tokenizer.eos_token_id
                outputs = model.generate(
                    input_ids, max_new_tokens=max_new_tokens,
                    stopping_criteria=StoppingCriteriaList([stop_criteria]),
                )
                request_time = time.time() - start_time
                request_datetime = int(time.time())

                # remove the attached input from output for some model
                scores = outputs.scores
                output_ids = outputs.sequences[0, -len(scores):]

                # remove the tail, if generation stops at any stop_sequences
                if output_ids[-1].item() in stop_sequences_set:
                    scores = scores[:-1]
                    output_ids = output_ids[:-1]

                # compute logprob for each token
                completions_tokens = []
                completions_logprob = 0

                for score, token in zip(scores, output_ids, strict=True):
                    logprobs = F.log_softmax(score[0], dim=-1)
                    logprob = logprobs[token].item()
                    completions_tokens.append({
                        'text': tokenizer.decode(token),
                        'logprob': logprob,
                    })
                    completions_logprob += logprob
                
                completions_text = tokenizer.decode(output_ids, skip_special_tokens=True)

                request['request'] = {
                    'result': {
                        'success': True,
                        'completions': [{
                            'text': completions_text,
                            'logprob': completions_logprob,
                            'tokens': completions_tokens,
                        }],
                        'cached': True,
                        'request_time': request_time,
                        'request_datetime': request_datetime,
                    }
                }
            
            dataset_name = dataset_name.replace('/', '++')  # rename sub task, e.g. KoRC/iid -> KoRC++iid, as filename
            if not test_input:
                prefix = 'r_' if layer == 'Rolling' else ''  # distinguish the datasets from Rolling, as they have the same names with the former ones 
                with open(result_path + f'{prefix}{dataset_name}_inference.json', 'w') as f:
                    json.dump(dataset_test, f, indent=2)
            else:
                print(input_text)
                print(sorted(lens))
            # else:
            #     with open(f'{dataset_name}_inputs.json', 'w') as f:
            #         json.dump(input_text, f, ensure_ascii=False)


if __name__ == '__main__':
    main()
