import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from model_training.data_processing import load_and_process_data

def train_model(train_file, test_file, keyword_file):
    """訓練模型"""
    os.makedirs('./results', exist_ok=True)

    dataset = load_and_process_data(train_file, test_file, keyword_file)

    model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=2).to('cuda')
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
    )

    print("開始訓練模型...")
    trainer.train()

    model.save_pretrained('./bert_fraud_model')
    tokenizer.save_pretrained('./bert_fraud_model')
    print("模型和 tokenizer 已經成功保存！")
