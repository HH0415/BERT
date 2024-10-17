import torch
from transformers import BertTokenizer, BertForSequenceClassification
import jieba
import jieba.analyse  
from keybert import KeyBERT

def predict(text):
    """對文本進行詐騙檢測"""
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    model = BertForSequenceClassification.from_pretrained('./bert_fraud_model').to('cuda')

    jieba_keywords = jieba.analyse.extract_tags(text, topK=5, withWeight=True)
    print(f"使用jieba提取到的關鍵詞: {jieba_keywords}")

    kw_model = KeyBERT()
    extracted_keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 1), stop_words='chinese', use_mmr=True, diversity=0.7)

    keywords = list(set(keyword for keyword, _ in jieba_keywords) | set(keyword for keyword, _ in extracted_keywords))
    
    weights = {keyword: weight for keyword, weight in jieba_keywords}

    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=512).to('cuda')

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    token_ids = inputs['input_ids'][0]
    weights_tensor = torch.tensor([weights.get(tokenizer.decode(token_id), 1.0) for token_id in token_ids], device='cuda')

    weighted_logits = logits * weights_tensor.mean() 

    predicted_class = torch.argmax(weighted_logits, dim=1).item()
    confidence = torch.softmax(weighted_logits, dim=1).max().item()

    if predicted_class == 1:
        print(f"該文本可能涉及詐騙，信心度：{confidence:.2f}")
    else:
        print(f"該文本無明顯詐騙風險，信心度：{confidence:.2f}")

