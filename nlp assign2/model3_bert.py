from transformers import DistilBERTForSequenceClassification, DistilBERTTokenizer
from preprocess import preprocess_text

def load_bert_model():
    # Load the pre-trained DistilBERT model
    model = DistilBERTForSequenceClassification.from_pretrained('distilbert-base-uncased')
    tokenizer = DistilBERTTokenizer.from_pretrained('distilbert-base-uncased')
    return model, tokenizer

def predict_bert(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    return outputs.logits.argmax(dim=1).item()

def run_bert(text):
    model, tokenizer = load_bert_model()
    preprocessed_text = preprocess_text(text)
    prediction = predict_bert(preprocessed_text, model, tokenizer)
    return prediction
