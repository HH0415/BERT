import torch
from transformers import BertTokenizer, BertForSequenceClassification
from model_training.data_processing import load_keywords

def predict(text, keyword_file):
    """對文本進行詐騙檢測"""
    # 載入 BERT 分詞器和模型
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    model = BertForSequenceClassification.from_pretrained('./bert_fraud_model')

    # 載入關鍵詞
    keywords = load_keywords(keyword_file)

    # 檢查文本中是否包含關鍵詞
    if any(keyword in text for keyword in keywords):
        print(f"該文本包含關鍵詞，直接判定為詐騙")
        return

    # 將文本編碼為 BERT 格式
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=512)

    # 使用 BERT 模型進行推理
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # 判斷分類結果
    predicted_class = torch.argmax(logits, dim=1).item()
    confidence = torch.softmax(logits, dim=1).max().item()

    if predicted_class == 1:
        print(f"該文本可能涉及詐騙，信心度：{confidence:.2f}")
    else:
        print(f"該文本無明顯詐騙風險，信心度：{confidence:.2f}")
