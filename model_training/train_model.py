import os
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from model_training.data_processing import load_and_process_data

def ensure_dir_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def train_model(train_file, test_file):
    """訓練BERT模型以區分詐騙與非詐騙文本"""
    ensure_dir_exists('./results')  # 確保目錄存在
    encoded_dataset = load_and_process_data(train_file, test_file)
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=2)

    training_args = TrainingArguments(
        output_dir='./results',  # 訓練結果儲存的目錄
        evaluation_strategy="epoch",  # 每個 epoch 評估一次
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir='./logs',  # 日誌儲存的目錄
        logging_steps=10,  # 每 10 步記錄一次
        save_steps=500,  # 每 500 步保存一次模型
        save_total_limit=3,  # 限制保存的模型數量
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset['train'],
        eval_dataset=encoded_dataset['test'],
    )

    trainer.train()

    # 保存模型和分詞器
    model.save_pretrained('./bert_fraud_model')
    tokenizer.save_pretrained('./bert_fraud_model')

if __name__ == "__main__":
    train_model('./data/train.csv', './data/test.csv')
   
#loss (損失)：模型在訓練時的損失值，也就是模型預測與實際值之間的誤差。這個數值越小，表示模型的預測越準確。
# grad_norm (梯度範數)：指標表示的是梯度的大小，用來衡量模型更新參數的幅度。梯度太大可能導致不穩定的訓練，太小則可能導致學習過慢。
# learning_rate (學習率)：學習率決定了模型每次更新參數時的步伐大小。用於微調模型或訓練到較後期時，這樣可以避免模型大幅度更新參數導致不穩定。
# epoch (訓練週期)：訓練週期是指完整地用所有訓練數據更新一遍模型參數。
