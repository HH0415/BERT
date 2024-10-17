import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from transformers import BertForSequenceClassification, Trainer
from model_training.data_processing import load_and_process_data, ensure_dir_exists
import matplotlib.pyplot as plt
import seaborn as sns
import torch

def plot_confusion_matrix(cm, labels, output_path):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(output_path)
    plt.close()

def evaluate_model(test_file, keyword_file):
    ensure_dir_exists('./results')
    
    encoded_dataset = load_and_process_data('./data/train.csv', test_file, keyword_file)

    model = BertForSequenceClassification.from_pretrained('./bert_fraud_model').to('cuda')

    trainer = Trainer(
        model=model,
        eval_dataset=encoded_dataset['test'],
    )

    evaluation_results = trainer.evaluate()
    print("評估結果:", evaluation_results)
    
    predictions = trainer.predict(encoded_dataset['test'])
    predictions_labels = np.argmax(predictions.predictions, axis=1)
    
    true_labels = np.array(encoded_dataset['test']['label'])

    cm = confusion_matrix(true_labels, predictions_labels)
    report = classification_report(true_labels, predictions_labels, target_names=['Normal', 'Fraud'])

    with open('./results/confusion_matrix.txt', 'w') as f:
        f.write("混淆矩陣:\n")
        f.write(np.array2string(cm))

    with open('./results/classification_report.txt', 'w') as f:
        f.write("分類報告:\n")
        f.write(report)

    plot_confusion_matrix(cm, ['Normal', 'Fraud'], './results/confusion_matrix.png')
