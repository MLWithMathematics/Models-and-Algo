# %%
# Import required libraries
import pandas as pd              # For data manipulation (reading CSV, organizing data)
import numpy as np               # For numerical computations
from sklearn.model_selection import train_test_split  # For splitting data
from sklearn.feature_extraction.text import TfidfVectorizer  # For converting text to numbers
from sklearn.naive_bayes import MultinomialNB  # The classification algorithm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report  # For evaluation
import pickle                    # For saving/loading the trained model
import re                        # For text pattern matching and cleaning
import string                    # For punctuation removal


data = pd.read_csv('spam.csv', encoding='latin-1')
# Display basic information

print("Dataset shape:", data.shape)  # Shows number of rows and columns
print("\nFirst few rows:")
print(data.head) # Shows first 5 rows

# The CSV has columns named 'v1' (label) and 'v2' (message)
# Rename them to something meaningful
data = data[['v1', 'v2']]  # Keep only these two columns
data.columns = ['label', 'message']  # Rename to 'label' and 'message'

# Convert text labels to numbers (required for machine learning)
# 'spam' becomes 1, 'ham' (not spam) becomes 0
data['label'] = data['label'].map({'spam': 1, 'ham': 0})
 
print("Updated data:")
print(data.head())

# Check how many spam vs ham messages
print("\nClass distribution:")
print(data['label'].value_counts())




# Day 3
def preprocess_text(text):
    """
    Clean and normalize text for machine learning
    """
    
    # Step 1: Convert to lowercase
    # "CONGRATULATIONS! You've won" → "congratulations! you've won"
    text = text.lower()
    
    # Step 2: Remove URLs (they don't help predict spam)
    # "Check this http://example.com now" → "Check this  now"
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Step 3: Remove punctuation
    # "Hello! How are you?" → "Hello How are you"
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Step 4: Remove numbers (they don't add much value)
    # "Call 123456789" → "Call "
    text = re.sub(r'\d+', '', text)
    
    # Step 5: Remove extra spaces
    # "Hello  world   test" → "Hello world test"
    text = ' '.join(text.split())
    
    return text

# Test the preprocessing
test_message = "CONGRATULATIONS!!! You've won $1000! Visit http://example.com or call 1-800-123-4567"
cleaned = preprocess_text(test_message)
print(f"Original: {test_message}")
print(f"Cleaned: {cleaned}")



# Apply preprocessing to every message in the dataset
data['clean_message'] = data['message'].apply(preprocess_text)

# Check the results
print("Before and After:")
for i in range(3):
    print(f"Original: {data['message'].iloc[i]}")
    print(f"Cleaned: {data['clean_message'].iloc[i]}")
    print()

# DAy 4


# Create a vectorizer object
# max_features=3000: Use only the top 3000 most important words (reduces computation)
# stop_words='english': Ignore common English words (the, is, and, etc.)
vectorizer = TfidfVectorizer(max_features=3000, stop_words='english')

# Fit and transform the cleaned messages
# This converts text into a matrix of numbers (3000 features per message)
X = vectorizer.fit_transform(data['clean_message']).toarray()  # type: ignore

# Get the labels (0 for ham, 1 for spam)
y = data['label'].values

print(f"Feature matrix shape: {X.shape}")
# Output: Feature matrix shape: (5572, 3000)
# Meaning: 5572 messages, each represented by 3000 numerical features

print(f"First message features (first 10): {X[0][:10]}")
# Shows the numerical values for the first message's first 10 words

# See which words the vectorizer learned
feature_names = vectorizer.get_feature_names_out()
print(f"Total unique words: {len(feature_names)}")
print(f"Sample words: {feature_names[:20]}")

# Day 5
from sklearn.model_selection import train_test_split

# Why split data?
# 1. Training set: Learn patterns from this data (80%)
# 2. Testing set: Evaluate performance on unseen data (20%)
# This prevents overfitting (memorizing training data instead of learning patterns)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,                    # Feature matrix (all messages as numbers)
    y,                    # Labels (spam or ham)
    test_size=0.2,        # Use 20% for testing, 80% for training
    random_state=42,      # Use fixed seed for reproducibility (same split each run)
    stratify=y            # Maintain spam/ham ratio in both splits
)

print(f"Training set size: {X_train.shape[0]} messages")
print(f"Testing set size: {X_test.shape[0]} messages")


from sklearn.naive_bayes import MultinomialNB

# Create the Naive Bayes model
model = MultinomialNB()

# Train the model on training data
# The model learns word probabilities associated with spam vs ham
model.fit(X_train, y_train)

print("✓ Model trained successfully!")

# Day 6

# Predict on test set
y_pred = model.predict(X_test)

# y_pred contains 1115 predictions (0 for ham, 1 for spam)
print(f"Predictions made: {len(y_pred)}")
print(f"Sample predictions: {y_pred[:10]}")

# 1. ACCURACY
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# What it means: Of all 1115 test messages, how many did we classify correctly?
# Example: 96.5% means 1078 correct, 37 incorrect

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)

# Add this code after the confusion matrix section (Day 6)

# Import F1 score metric
from sklearn.metrics import f1_score

# Calculate F1 score
# F1 = 2 * (precision * recall) / (precision + recall)
# Balances precision (avoid false spam flags) and recall (catch spam)
f1 = f1_score(y_test, y_pred)

print(f"F1 Score: {f1:.4f} ({f1*100:.2f}%)")

# Detailed breakdown (optional)
from sklearn.metrics import precision_score, recall_score

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"Recall: {recall:.4f} ({recall*100:.2f}%)")
print(f"F1 Score: {f1:.4f} ({f1*100:.2f}%) [web:8][web:16]")

# Expected output with your data (tn=1030, fp=8, fn=29, tp=48):
# Precision: 0.8571 (85.71%)  → TP / (TP + FP) = 48 / (48 + 8)
# Recall: 0.6234 (62.34%)     → TP / (TP + FN) = 48 / (48 + 29)
# F1 Score: 0.7273 (72.73%)


# Interpretation:
#                Predicted Ham    Predicted Spam
# Actual Ham:         1030              8          (8 false positives)
# Actual Spam:         29              48          (29 false negatives)

# Breaking it down:
tn, fp, fn, tp = cm.ravel()  # tn=1030, fp=8, fn=29, tp=48

print(f"True Negatives (TN): {tn}   - Correctly identified as ham")
print(f"False Positives (FP): {fp} - Legitimate emails marked as spam (bad!)")
print(f"False Negatives (FN): {fn} - Spam emails marked as ham (bad!)")
print(f"True Positives (TP): {tp}   - Correctly identified as spam")


print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

# Day 7

def predict_spam(message):
    """
    Classify a new message as spam or ham
    """
    # Step 1: Clean the message using same preprocessing
    clean_msg = preprocess_text(message)
    
    # Step 2: Vectorize using the same vectorizer (IMPORTANT!)
    # Must use trained vectorizer to maintain same 3000 features
    msg_vector = vectorizer.transform([clean_msg]).toarray()
    
    # Step 3: Make prediction
    prediction = model.predict(msg_vector)[0]  # 0 or 1
    
    # Step 4: Get prediction probability
    probability = model.predict_proba(msg_vector)[0]  # [prob_ham, prob_spam]
    
    # Step 5: Format output
    result = "SPAM" if prediction == 1 else "HAM"
    confidence = probability[prediction] * 100
    
    return result, confidence

# Test it
test_msgs = [
    "Congratulations you've won $1000 click here now",
    "Hi, are we still on for tomorrow at 3pm?",
    "FREE prize claim now urgent action required",
    "Can you send me the project report?"
]

for msg in test_msgs:
    result, confidence = predict_spam(msg)
    print(f"Message: {msg}")
    print(f"Prediction: {result} (Confidence: {confidence:.2f}%)\n")

import pickle

# Save the trained model
with open('spam_detector_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("✓ Model saved")

# Save the vectorizer (CRITICAL - needed for preprocessing)
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
print("✓ Vectorizer saved")

# Later, load and use without retraining
def load_and_predict(message):
    """Load saved model and make predictions"""
    # Load saved model
    with open('spam_detector_model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)
    
    # Load saved vectorizer
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        loaded_vectorizer = pickle.load(f)
    
    # Preprocess
    clean_msg = preprocess_text(message)
    
    # Vectorize
    msg_vector = loaded_vectorizer.transform([clean_msg]).toarray()
    
    # Predict
    prediction = loaded_model.predict(msg_vector)[0]
    
    return "SPAM" if prediction == 1 else "HAM"

# Use it
result = load_and_predict("URGENT: Claim your prize now!")
print(f"Result: {result}")


# %%