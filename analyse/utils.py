

std_model_names = ['FLAN-T5 (11B)',  'UL2 (20B)' ,'FLAN-UL2 (20B)', 'GPT-JT (6B)', 'GPT-J (6B)', 'GPT-NeoX (20B)','BLOOM (7B)','T0++ (11B)', 'LLaMa (65B)','Alpaca (7B)',
                    'J2-Jumbo-Instruct (178B*)','Cohere-command (52.4B)',  'GPT-3.5-turbo', 'GPT-3 curie v1 (6.7B)','GPT-3 davinci v1 (175B)', 'InstructGPT curie v1 (6.7B*)', 
                    'InstructGPT davinci v2 (175B*)','GLM (130B)', 'GPT-4', 'ChatGLM (130B)', 'ChatGLM (6B)']
"""
id2dataset ={
    '1-1': '2_high_freq_ent',
    '1-2': '1_low_freq_ent',
    '1-3': 'r_1_simple_sample',
    '2-1': 'FewNERD',
    '2-2': 'DocRED',
    '2-3': 'COPEN++csj',
    '2-4': 'COPEN++cpj',
    '2-5': 'COPEN++cic',
    '2-6': 'MAVEN',
    '2-7': 'MAVEN-ERE',
    '2-8': 'r_DocRED',
    '3-1': 'hotpotqa',
    '3-2': '2wikimultihopqa',
    '3-3': 'musique',
    '3-4': 'kqapro',
    '3-5': 'KoRC++ood',
    '3-6': 'r_KoRC++ood',
    '4-1': 'Creating',
    '4-2': 'Rolling',
}
"""
id2dataset ={
    '1-1': '2_high_freq_ent',
    '1-2': '1_low_freq_ent',
    '1-3': 'r_1_simple_sample',
    '2-1': 'COPEN++csj',
    '2-2': 'COPEN++cpj',
    '2-3': 'COPEN++cic',
    '2-4': 'FewNERD',
    '2-5': 'DocRED',
    '2-6': 'MAVEN',
    '2-7': 'MAVEN-ERE',
    '2-8': 'r_DocRED',
    '3-1': 'hotpotqa',
    '3-2': '2wikimultihopqa',
    '3-3': 'musique',
    '3-4': 'kqapro',
    '3-5': 'KoRC++ood',
    '3-6': 'r_KoRC++ood',
    '4-1': 'Creating',
    '4-2': 'Rolling',
}

id2level = {
    '1-1': 'KG',
    '1-2': 'KG',
    '1-3': 'Rolling',
    '2-1': 'IE',
    '2-2': 'IE',
    '2-3': 'IE',
    '2-4': 'IE',
    '2-5': 'IE',
    '2-6': 'IE',
    '2-7': 'IE',
    '2-8': 'Applying',
    '3-1': 'Applying',
    '3-2': 'Applying',
    '3-3': 'Applying',
    '3-4': 'Applying',
    '3-5': 'Applying',
    '3-6': 'Rolling',
    '4-1': 'Creating',
    '4-2': 'Rolling',
}

few_sub_tasks =  ['inter', 'intra', 'supervised']

level_1_2 = ['1-1', '1-2', '1-3', '2-1', '2-2', '2-3', '2-4', '2-5', '2-6', '2-7', '2-8']
dataset_level_1_2 = [id2dataset[i] for i in level_1_2]
level_3_4 = ['3-1', '3-2', '3-3', '3-4', '3-5', '3-6', '4-1', '4-2']
dataset_level_3_4 = [id2dataset[i] for i in level_3_4]


def load_dataset(dataset_name, layer):
    import os
    import json
    # find dataset
    dataset_path = f'/data2/cookie/input/{layer}/{dataset_name}/'
    print(dataset_path)
    if not os.path.exists(dataset_path):
        print(f'Dataset {dataset_name} not found!')
        return None, None
    # load dataset
    test_file = os.path.join(dataset_path, 'test.json')
    with open(test_file) as f:
        dataset_test = json.load(f)
        request_states = dataset_test['request_states']
        print(f'Load {len(request_states)} data from {dataset_name}')




def main():
    for k, v in id2dataset.items():
        print(k, v)
        if v.startswith('r_'):
            v = v[2:]
        if '++' in v:
            sub_task = v.split('++')[1]
            name = v.split('++')[0]
            # for sub_task in few_sub_tasks:
            load_dataset(f"{name}/{sub_task}", id2level[k])
        elif k == '2-1':
            for sub_task in few_sub_tasks:
                load_dataset(f"{v}/{sub_task}", id2level[k])
        elif v in ['Rolling', 'Creating']:
            for sub_task in ['with_triples', 'without_triples']:
                load_dataset(f"{sub_task}", id2level[k])
        else:
            try:
                load_dataset(v, id2level[k])
            except FileNotFoundError:
                pass


if __name__ == '__main__':
    main()


