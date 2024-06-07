from torch.utils.data import Dataset
import torch
import os
from dataclasses import dataclass
from tqdm import tqdm
import pandas as pd
import json
from transformers import AutoTokenizer
from accelerate.logging import get_logger
from sklearn.utils import resample
import random
from nltk.corpus import wordnet

logger = get_logger('my_logger')

def prepare_raw_data(data_dict, category, type):
    ''' 
    category: 'easy', 'medium', 'hard'
    type: 'train', 'test', 'validation'
    '''
    folder_path = os.path.join(data_dict, category, type)
    raw_data = []
    # get the file numbers in the folder
    file_numbers = len(os.listdir(folder_path)) // 2
    for i in tqdm(range(file_numbers)):
        # read json file
        json_file_path = os.path.join(folder_path, f'truth-problem-{str(i+1)}.json')
        with open(json_file_path, 'r') as f:
            json_data = json.load(f)
        # read txt file
        txt_file_path = os.path.join(folder_path, f'problem-{str(i+1)}.txt')
        with open(txt_file_path, 'r', newline="") as f:
            txt_data = f.readlines()
        # check the length of the data
        if len(txt_data) != len(json_data["changes"]) + 1:
            logger.info(f'Length of the data is not equal to the length of the changes in {txt_file_path}')
            continue
        # get the data
        pre_s = None
        for j, s in enumerate(txt_data):
            s = s.strip()
            if s == '':
                raise ValueError('Empty sentence')
            if pre_s is None:
                pre_s = s
                continue
            raw_data.append([pre_s, s, json_data["changes"][j-1]])
            pre_s = s
    # create the dataframe
    raw_data = pd.DataFrame(raw_data, columns=['sentence_1', 'sentence_2', 'label'])
    # save the data
    raw_data.to_csv(os.path.join(data_dict, f'{category}_{type}.csv'), index=False)
    return raw_data

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace('_', ' ')
            if synonym != word:
                synonyms.add(synonym)
    return list(synonyms)

def synonym_replacement(sentence, replace_prob=0.3):
    words = sentence.split()
    new_sentence = words.copy()
    for i, word in enumerate(words):
        if random.random() < replace_prob:
            synonyms = get_synonyms(word)
            if synonyms:
                synonym = random.choice(synonyms)
                new_sentence[i] = synonym
    return ' '.join(new_sentence)


def prepare_data(data_dict, category, type, tokenizer_type, max_length, augment_prob=0.3):
    ''' 
    category: 'easy', 'medium', 'hard'
    type: 'train', 'test', 'validation'
    augment_prob: probability of applying synonym replacement
    '''
    csv_path = os.path.join(data_dict, f'{category}_{type}.csv')
    if not os.path.exists(csv_path):
        raw_data = prepare_raw_data(data_dict, category, type)
    else:
        raw_data = pd.read_csv(csv_path)
    
    # Resample the data to ensure balanced distribution
    if category in ['easy', 'medium', 'hard']:
        # Resample data to balance classes
        majority = raw_data[raw_data.label == 1]
        minority = raw_data[raw_data.label == 0]
        upsampled_data = []
        if len(majority) > len(minority):
            for _, row in minority.iterrows():
                for _ in range(len(majority) // len(minority)):
                    new_row = row.copy()
                    new_row['sentence_1'] = synonym_replacement(new_row['sentence_1'], replace_prob=augment_prob)
                    new_row['sentence_2'] = synonym_replacement(new_row['sentence_2'], replace_prob=augment_prob)
                    upsampled_data.append(new_row)
            raw_data = pd.concat([majority, pd.DataFrame(upsampled_data)])
        else:
            for _, row in majority.iterrows():
                for _ in range(len(minority) // len(majority)):
                    new_row = row.copy()
                    new_row['sentence_1'] = synonym_replacement(new_row['sentence_1'], replace_prob=augment_prob)
                    new_row['sentence_2'] = synonym_replacement(new_row['sentence_2'], replace_prob=augment_prob)
                    upsampled_data.append(new_row)
            raw_data = pd.concat([minority, pd.DataFrame(upsampled_data)])
    
    # get the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)
    # tokenize the data
    tokenized_data = tokenizer(
        raw_data['sentence_1'].tolist(),
        raw_data['sentence_2'].tolist(),
        max_length=max_length,
        truncation=True,
        padding='max_length',
        return_tensors='pt',
        add_special_tokens=True
    )
    # data
    data = {
        'input_ids': tokenized_data['input_ids'].tolist(),
        'attention_mask': tokenized_data['attention_mask'].tolist(),
        'labels': raw_data['label'].tolist()
    }
    # save the data
    tokenizer_type_save = tokenizer_type.replace('/', '_')
    torch.save(data, os.path.join(data_dict, f'{tokenizer_type_save}_{category}_{type}.pt'))
    return data

class ClassificationDataset(Dataset):
    def __init__(self, data_dict, category, type, tokenizer_type, max_length):
        tokenizer_type_save = tokenizer_type.replace('/', '_')
        if not os.path.exists(os.path.join(data_dict, f'{tokenizer_type_save}_{category}_{type}.pt')):
            prepare_data(data_dict, category, type, tokenizer_type, max_length)
        self.data = torch.load(os.path.join(data_dict, f'{tokenizer_type_save}_{category}_{type}.pt'))

    def __len__(self):
        return len(self.data['input_ids'])

    def __getitem__(self, idx):
        return {
            'input_ids': self.data['input_ids'][idx],
            'attention_mask': self.data['attention_mask'][idx],
            'labels': self.data['labels'][idx]
        }

@dataclass
class ClassificationsCollator:
    def __call__(self, batch):
        input_ids = [item['input_ids'] for item in batch]
        attention_mask = [item['attention_mask'] for item in batch]
        labels = torch.tensor([item['labels'] for item in batch])
        return {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_mask),
            'labels': labels
        }
