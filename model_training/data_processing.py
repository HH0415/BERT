from transformers import BertTokenizer
from datasets import load_dataset

def preprocess_function(examples):
    """對文本數據進行分詞和編碼"""
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)

def load_and_process_data(train_file, test_file):
    """加載數據集並進行預處理"""
    dataset = load_dataset('csv', data_files={'train': train_file, 'test': test_file})
    encoded_dataset = dataset.map(preprocess_function)
    return encoded_dataset

if __name__ == "__main__":
    encoded_dataset = load_and_process_data('./data/train.csv', './data/test.csv')
    print(encoded_dataset)
