import os
from transformers import BertTokenizer, BertForSequenceClassification, Trainer
from model_training.data_processing import load_and_process_data
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def ensure_dir_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def plot_confusion_matrix(cm, labels, save_path):
    """生成並保存混淆矩陣的圖表"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig(save_path)
    plt.close()

def evaluate_model(test_file):
    """評估訓練好的BERT模型"""
    ensure_dir_exists('./results')  # 確保目錄存在
    encoded_dataset = load_and_process_data('./data/train.csv', test_file)

    model = BertForSequenceClassification.from_pretrained('./bert_fraud_model')

    trainer = Trainer(
        model=model,
        eval_dataset=encoded_dataset['test'],
    )

    evaluation_results = trainer.evaluate()
    print(evaluation_results)

    predictions = trainer.predict(encoded_dataset['test'])
    predictions_labels = np.argmax(predictions.predictions, axis=1)
    
    true_labels = np.array(encoded_dataset['test']['label'])

    cm = confusion_matrix(true_labels, predictions_labels)
    report = classification_report(true_labels, predictions_labels, target_names=['Normal', 'Fraud'])

    # 保存文本報告
    with open('./results/confusion_matrix.txt', 'w') as f:
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(cm))

    with open('./results/classification_report.txt', 'w') as f:
        f.write("Classification Report:\n")
        f.write(report)

    plot_confusion_matrix(cm, ['Normal', 'Fraud'], './results/confusion_matrix.png')

if __name__ == "__main__":
    evaluate_model('./data/test.csv')
