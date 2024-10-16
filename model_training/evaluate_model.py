import numpy as np
from transformers import BertForSequenceClassification, Trainer
from sklearn.metrics import confusion_matrix, classification_report
from model_training.data_processing import ensure_dir_exists, load_and_process_data  
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(cm, labels, output_path):
    """繪製混淆矩陣並保存圖像"""
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(output_path)
    plt.close()  # 關閉圖形以釋放內存

def evaluate_model(test_dataset, keyword_file):
    """評估訓練好的 BERT 模型"""
    ensure_dir_exists('./results')  # 確保結果目錄存在

    encoded_dataset = load_and_process_data(test_dataset, keyword_file)

    model = BertForSequenceClassification.from_pretrained('./bert_fraud_model')

    trainer = Trainer(
        model=model,
        eval_dataset=encoded_dataset['test'],
    )

    evaluation_results = trainer.evaluate()
    print("Evaluation results:", evaluation_results)

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
    # 測試數據集和關鍵字文件的路徑
    test_dataset_path = './data/test.csv'
    keyword_file_path = './keywords.txt'
    
    evaluate_model(test_dataset_path, keyword_file_path)
