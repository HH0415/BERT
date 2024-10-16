import os
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from model_training.data_processing import load_and_process_data

def train_model(train_file, test_file, keyword_file):
    # 加載數據
    dataset = load_and_process_data(train_file, test_file, keyword_file)  # 返回整個數據集

    # 使用 BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    # 將文本轉換為 token ids
    def preprocess_function(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)

    # 準備訓練數據
    train_dataset = dataset['train'].map(preprocess_function, batched=True)  # 使用 'train' 和 'test' 鍵
    test_dataset = dataset['test'].map(preprocess_function, batched=True)

    # 創建模型
    model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=2)

    # 設置訓練參數
    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    # 創建 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    # 開始訓練模型
    print("開始訓練模型...")
    trainer.train()

    # 保存模型
    model.save_pretrained('./bert_fraud_model')  # 保存模型權重
    tokenizer.save_pretrained('./bert_fraud_model')  # 保存 tokenizer
    print("模型和 tokenizer 已經成功保存！")


