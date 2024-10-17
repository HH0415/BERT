from model_training.train_model import train_model
from model_training.evaluate_model import evaluate_model
from model_training.predict_model import predict

if __name__ == "__main__":
    print("開始訓練模型...")
    train_model('./data/train.csv', './data/test.csv', './keywords.txt')

    print("評估模型...")
    evaluate_model('./data/test.csv', './keywords.txt')

    print("進行文本推理...")
    text = input("請輸入要檢測的文本：")
    predict(text)
