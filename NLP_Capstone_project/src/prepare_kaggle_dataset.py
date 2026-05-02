import pandas as pd
import os
import json
import dotenv
import re

DATA_DIR = os.path.join('Tasks', 'project-sprints', 'Sprints-capstone-project', 'NLP_Capstone_project', 'data', 'raw')

def clean_text(text):
    text = text.lower()
    text = text.replace('\n', ' ')
    text = text.replace('\r', ' ')
    text = text.replace('\t', ' ')
    text = text.replace('\f', ' ')
    text = text.replace('\v', ' ')
    text = str.strip(text)
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('\\n', ' ')
    text = text.replace('<name>', '')
    text = text.replace('<email>', '')
    text = text.replace('[name]', '')
    text = text.replace('<tel_num>', '')
    return text

class PrepareKaggleDataset:
    def __init__(self):
        self.data_path = os.path.join(DATA_DIR, 'data.csv')
        self.data = pd.read_csv(self.data_path)
    def prepare_data(self):
        '''Select only English conversations'''
        self.data = self.data[self.data['language'] == 'en']
        
        self.data.rename(columns={'body': 'customer_issue', 'answer': 'reference_reply'}, inplace=True)
        self.data = self.data[['subject','customer_issue', 'reference_reply']]
        self.data.dropna(inplace=True)
        self.data.reset_index(drop=True, inplace=True)
        
        self.data['customer_issue'] = self.data['customer_issue'].apply(clean_text)
        self.data['reference_reply'] = self.data['reference_reply'].apply(clean_text)
        
        self.data.drop_duplicates(inplace=True)
        
        self.data['customer_issue'] = 'Subject :'+self.data['subject']+'\n'+self.data['customer_issue']
        self.data.drop(columns=['subject'], inplace=True)
        
        self.data = self.data.sample(1000, random_state=42)
        
        df = self.data.sample(20, random_state=42)
        
        os.makedirs('data', exist_ok=True)
        
        test_records = df[['customer_issue', 'reference_reply']].to_dict(orient='records')
        with open('data/test_subset.json', 'w', encoding='utf-8') as f:
            json.dump(test_records, f, indent=2, ensure_ascii=False)
        train_data = self.data.drop(df.index)
        train_data.to_csv('data/support_conversations.csv', index=False)
        
        
        
        