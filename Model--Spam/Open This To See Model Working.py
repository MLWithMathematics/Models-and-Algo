import pickle
import re
import string

# ---------- same preprocess function as training ----------

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = ' '.join(text.split())
    return text

# ---------- load saved model and vectorizer ----------

with open('spam_detector_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# ---------- helper function to predict ----------

def predict_spam(message: str):
    clean_msg = preprocess_text(message)
    msg_vec = vectorizer.transform([clean_msg])      # no .fit here!
    pred = model.predict(msg_vec)[0]                 # 0 = ham, 1 = spam
    proba = model.predict_proba(msg_vec)[0]          # [P(ham), P(spam)]
    label = "SPAM" if pred == 1 else "HAM"
    confidence = proba[pred] * 100
    return label, confidence

# ---------- example usage ----------

if __name__ == "__main__":
    while True:
        text = input("\nEnter a message (or 'q' to quit): ")
        if text.lower() == 'q':
            break
        label, conf = predict_spam(text)
        print(f"Result: {label} (confidence: {conf:.2f}%)")
