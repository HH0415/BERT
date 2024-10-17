import os
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import BertTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

def ensure_dir_exists(directory):
    """確保指定的目錄存在，如果不存在則創建它"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_keywords(keyword_file):
    """從外部檔案讀取關鍵詞辭庫"""
    with open(keyword_file, 'r', encoding='utf-8') as f:
        keywords = [line.strip() for line in f.readlines()]
    return keywords

def preprocess_function(examples, keywords):
    """對文本數據進行分詞和編碼，並增加TF-IDF特徵"""
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    tokenized_inputs = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)

    vectorizer = TfidfVectorizer(vocabulary=keywords)
    tfidf_features = vectorizer.fit_transform([examples['text']]).toarray()

    tokenized_inputs['tfidf'] = tfidf_features.tolist()
    
    return tokenized_inputs

def load_and_process_data(train_file, test_file, keyword_file):
    """加載數據集並進行預處理，包含TF-IDF與關鍵詞"""
    dataset = load_dataset('csv', data_files={'train': train_file, 'test': test_file})
    keywords = load_keywords(keyword_file)

    encoded_dataset = dataset.map(lambda examples: preprocess_function(examples, keywords))
    return encoded_dataset
