import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "./model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    pred_class = torch.argmax(probs, dim=-1).item()
    confidence = probs[0][pred_class].item()
    return {"label": "positive" if pred_class == 1 else "negative", "confidence": confidence}

if __name__ == "__main__":
    while True:
        text = input("Enter review (or 'exit'): ")
        if text.lower() == "exit":
            break
        result = predict(text)
        print(result)
