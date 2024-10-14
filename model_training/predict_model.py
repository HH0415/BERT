from transformers import BertTokenizer, BertForSequenceClassification
import torch

def predict(text):
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    model = BertForSequenceClassification.from_pretrained('./bert_fraud_model')

    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    predicted_class = torch.argmax(logits, dim=1).item()
    confidence = torch.softmax(logits, dim=1).max().item()

    if predicted_class == 1:
        print(f"該文本可能涉及詐騙，信心度：{confidence:.2f}")
    else:
        print(f"該文本無明顯詐騙風險，信心度：{confidence:.2f}")

if __name__ == "__main__":
    text = input("請輸入要檢測的文本：")
    predict(text)
